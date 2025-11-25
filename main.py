

import os
import re
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from glob import glob
import signal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspedas
import cdflib
from scipy.interpolate import interp1d
from tqdm import tqdm
from pytplot import get_data, tplot_names

warnings.filterwarnings("ignore")

# ---------------------- Basic timeout + time helpers ---------------------- #

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("CDF read timed out")

def safe_cdf_read(cdf_file, timeout=10):
    """Read CDF with timeout protection."""
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        cdf = cdflib.CDF(cdf_file)
        signal.alarm(0)
        return cdf
    except TimeoutError:
        signal.alarm(0)
        print(f"    TIMEOUT reading {os.path.basename(cdf_file)}")
        return None
    except Exception as e:
        signal.alarm(0)
        print(f"    ERROR reading {os.path.basename(cdf_file)}: {e}")
        return None

def ensure_datetime_array(times):
    """
    Robustly convert any list/array of CDF epochs / numpy datetime / ints
    into an array of Python datetime.datetime.
    """
    out = []
    for t in times:
        if isinstance(t, datetime):
            out.append(t)
        elif isinstance(t, np.datetime64):
            out.append(pd.to_datetime(t).to_pydatetime())
        else:
            # Try CDF epoch
            try:
                dt = cdflib.cdfepoch.to_datetime(t)
                if isinstance(dt, (list, np.ndarray)):
                    out.append(dt[0])
                else:
                    out.append(dt)
                continue
            except Exception:
                pass
            # Fallback: treat as UNIX seconds
            try:
                out.append(datetime.utcfromtimestamp(float(t)))
            except Exception:
                # Last resort: sentinel date
                out.append(datetime(1970, 1, 1))
    return np.array(out, dtype=object)

# -------------------------- Debug + cleaning helpers ---------------------- #

def _dbg_stats(name, arr, units="", expected_min=None, expected_max=None):
    arr = np.asarray(arr, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        print(f"[DBG] {name}: no finite data.")
        return

    amin = float(np.min(finite))
    amax = float(np.max(finite))
    amed = float(np.median(finite))

    print(f"[DBG] {name}: N={finite.size}, min={amin:.3g} {units}, "
          f"median={amed:.3g} {units}, max={amax:.3g} {units}")

    if expected_min is not None or expected_max is not None:
        bad_low = bad_high = 0
        if expected_min is not None:
            bad_low = int(np.sum(finite < expected_min))
        if expected_max is not None:
            bad_high = int(np.sum(finite > expected_max))
        if bad_low or bad_high:
            print(f"[WARN] {name}: out-of-range => "
                  f"{bad_low} below {expected_min} {units}, "
                  f"{bad_high} above {expected_max} {units}")

def _clean_omni_array(arr, physical_max, name, units):
    """
    Mask OMNI sentinel / impossible values.
    """
    arr = np.asarray(arr, dtype=float)
    mask_sent = (
        (~np.isfinite(arr)) |
        (np.abs(arr) >= 1e29) |
        (arr == 99999) |
        (arr == 9999) |
        (arr > 999) |              # ← KEY FIX: catches 999.99 sentinels
        (arr == -999) |
        (arr < 0)                  # ← NEW: negative density impossible
    )
    mask_big = np.abs(arr) > physical_max
    bad = mask_sent | mask_big

    if np.any(bad):
        print(f"[WARN] {name}: masking {int(np.sum(bad))} bad points "
              f"(sentinels or |x|>{physical_max} {units})")

    cleaned = arr.copy()
    cleaned[bad] = np.nan
    return cleaned

# ------------------------------ SunPy import ------------------------------ #

try:
    from sunpy.coordinates import get_horizons_coord
    from astropy.time import Time
    import astropy.units as u
    SUNPY_AVAILABLE = True
except ImportError:
    SUNPY_AVAILABLE = False
    print("Warning: SunPy not available. Spacecraft ephemerides will not be loaded.")

# ------------------------------- Matplotlib ------------------------------- #

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "figure.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.25,
})

# ------------------------------ Constants -------------------------------- #

AU_KM            = 1.496e8
YEAR_DAYS        = 365.25
MEAN_MOTION      = 2 * np.pi / YEAR_DAYS

START_DATE       = datetime(2020, 1, 1)
END_DATE         = datetime(2024, 12, 31)
ECCENTRICITY     = 0.10
SUN_EARTH_TOL_AU = 0.08
DRO_TOL_AU       = 0.50
DT_HOURS         = 6
INSITU_WINDOW_HR = 72  # ± 3 days

# target cadence for MAG decimation (~1 point per minute per day-file)
MAG_TARGET_POINTS_PER_DAY = 1440

PLOT_OVERVIEW          = True
PLOT_HELIOCENTRIC      = True
PLOT_EARTH_FRAME       = True
PLOT_INSITU_DST        = True

EVENT_LIMIT = {
    "STEREO-A":      None,
    "Solar Orbiter": None,
}

WRITE_EVENTS_CSV   = True
CSV_FILENAME       = "henon_dro_crossover_events.csv"
TRACK_CONTEXT_DAYS = 42

# ------------------------------ Orbit model ------------------------------ #

class HenonDRO:
    def __init__(self, eccentricity=0.10, rotation_deg=0.0):
        self.e = eccentricity
        self.rotation_deg = rotation_deg
        self.rotation_rad = np.radians(rotation_deg)

    def solve_kepler(self, M):
        M = np.asarray(M)
        E = np.where(M < np.pi, M + self.e / 2.0, M - self.e / 2.0)
        for _ in range(32):
            f  = E - self.e * np.sin(E) - M
            fp = 1 - self.e * np.cos(E)
            dE = -f / fp
            E  = E + dE
            if np.max(np.abs(dE)) < 1e-10:
                break
        return E

    def generate_orbit(self, duration_days=365.0, dt_days=0.5):
        t = np.arange(0.0, duration_days + dt_days, dt_days)

        earth_x   = AU_KM * np.cos(MEAN_MOTION * t)
        earth_y   = AU_KM * np.sin(MEAN_MOTION * t)
        earth_pos = np.column_stack([earth_x, earth_y])

        M_sc = MEAN_MOTION * t
        E_sc = self.solve_kepler(M_sc)

        nu = 2 * np.arctan2(
            np.sqrt(1 + self.e) * np.sin(E_sc / 2),
            np.sqrt(1 - self.e) * np.cos(E_sc / 2)
        )

        r       = AU_KM * (1 - self.e**2) / (1 + self.e * np.cos(nu))
        x_orbit = r * np.cos(nu)
        y_orbit = r * np.sin(nu)

        cos_rot = np.cos(self.rotation_rad)
        sin_rot = np.sin(self.rotation_rad)

        x_helio = x_orbit * cos_rot - y_orbit * sin_rot
        y_helio = x_orbit * sin_rot + y_orbit * cos_rot

        dro_helio = np.column_stack([x_helio, y_helio])
        dro_earth = np.zeros_like(dro_helio)

        for i, ti in enumerate(t):
            theta     = MEAN_MOTION * ti
            rel       = dro_helio[i] - earth_pos[i]
            cos_theta = np.cos(-theta)
            sin_theta = np.sin(-theta)
            dro_earth[i, 0] = rel[0] * cos_theta - rel[1] * sin_theta
            dro_earth[i, 1] = rel[0] * sin_theta + rel[1] * cos_theta

        return {
            "times":       t,
            "earth_helio": earth_pos,
            "dro_helio":   dro_helio,
            "dro_earth":   dro_earth,
        }


class DROConstellation:
    def __init__(self, eccentricity=0.10):
        self.e = eccentricity
        self.satellites = []
        for i, angle in enumerate([0.0, 120.0, 240.0]):
            dro = HenonDRO(eccentricity, angle)
            orbit_data = dro.generate_orbit()
            self.satellites.append({
                "id":           i + 1,
                "rotation_deg": angle,
                "orbit":        orbit_data,
            })

# ----------------------- Ephemerides + geometry -------------------------- #

class SpacecraftData:
    HORIZONS_IDS = {
        "STEREO-A":      "-234",
        "Solar Orbiter": "-144",
    }

    @staticmethod
    def get_positions(spacecraft_name, start_date, end_date, dt_hours=6):
        if not SUNPY_AVAILABLE:
            return None
        if spacecraft_name not in SpacecraftData.HORIZONS_IDS:
            return None

        horizons_id   = SpacecraftData.HORIZONS_IDS[spacecraft_name]
        chunk_days    = 365
        all_times     = []
        all_positions = []

        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=chunk_days), end_date)
            times       = []
            current     = current_start
            while current <= current_end:
                times.append(current)
                current += timedelta(hours=dt_hours)
            if not times:
                break

            try:
                t_astropy = Time([t.isoformat() for t in times])
                coords    = get_horizons_coord(horizons_id, t_astropy)
                ecliptic  = coords.transform_to("heliocentriceclipticiau76")

                r_km = ecliptic.distance.to(u.km).value
                lon  = ecliptic.lon.to(u.rad).value
                lat  = ecliptic.lat.to(u.rad).value

                x = r_km * np.cos(lat) * np.cos(lon)
                y = r_km * np.cos(lat) * np.sin(lon)
                z = r_km * np.sin(lat)

                positions = np.column_stack([x, y, z])
                all_times.extend(times)
                all_positions.append(positions)
            except Exception:
                pass

            current_start = current_end + timedelta(hours=dt_hours)

        if not all_positions:
            return None

        combined_positions = np.vstack(all_positions)
        return {"times": all_times, "positions": combined_positions}


class SunEarthLineAnalyzer:
    @staticmethod
    def earth_position_at_time(time_dt, epoch_dt):
        days  = (time_dt - epoch_dt).total_seconds() / 86400
        theta = MEAN_MOTION * days
        return np.array([AU_KM * np.cos(theta), AU_KM * np.sin(theta), 0.0])

    @staticmethod
    def distance_to_sun_earth_line(spacecraft_pos, earth_pos):
        v      = earth_pos
        v_hat  = v / np.linalg.norm(v)
        w      = spacecraft_pos
        s      = np.dot(w, v_hat)
        proj   = s * v_hat
        perp   = w - proj
        is_between = 0 < s < np.linalg.norm(v)
        return np.linalg.norm(perp), s, is_between


class CrossoverFinder:
    def __init__(self, constellation, sun_earth_tolerance_au=0.05, dro_tolerance_au=0.05):
        self.constellation = constellation
        self.se_tol_km     = sun_earth_tolerance_au * AU_KM
        self.dro_tol_km    = dro_tolerance_au * AU_KM

    def find_events(self, spacecraft_data, epoch_date):
        events = []
        for time, pos in zip(spacecraft_data["times"], spacecraft_data["positions"]):
            earth_pos  = SunEarthLineAnalyzer.earth_position_at_time(time, epoch_date)
            perp, s, is_between = SunEarthLineAnalyzer.distance_to_sun_earth_line(pos, earth_pos)
            if perp > self.se_tol_km:
                continue
            for sat in self.constellation.satellites:
                dro_positions = sat["orbit"]["dro_helio"]
                dists         = np.linalg.norm(dro_positions - pos[:2], axis=1)
                j             = np.argmin(dists)
                dmin          = dists[j]
                if dmin < self.dro_tol_km:
                    events.append({
                        "time":                 time,
                        "spacecraft_pos":       pos,
                        "earth_pos":            earth_pos,
                        "dro_id":               sat["id"],
                        "dro_pos":              dro_positions[j],
                        "perp_to_se_line_au":   perp / AU_KM,
                        "along_se_line_au":     s / AU_KM,
                        "dist_to_dro_au":       dmin / AU_KM,
                        "is_between_sun_earth": is_between,
                    })
        return events

    def group_events_by_day(self, events):
        if not events:
            return []
        events.sort(key=lambda x: x["time"])
        groups = []
        cur    = [events[0]]
        for e in events[1:]:
            if (e["time"] - cur[0]["time"]).days < 7:
                cur.append(e)
            else:
                groups.append(cur)
                cur = [e]
        if cur:
            groups.append(cur)
        best = [min(g, key=lambda x: x["dist_to_dro_au"]) for g in groups]
        return best

# ------------------------- L1 OMNI loader + ballistic -------------------- #

class L1DataLoader:
    @staticmethod
    def load_omni_data(t0_dt, t1_dt):
        """
        Load OMNI high-res OMNI2 data and clean obvious sentinel/outlier values.
        """
        try:
            trange = [t0_dt.strftime("%Y-%m-%d/%H:%M:%S"),
                      t1_dt.strftime("%Y-%m-%d/%H:%M:%S")]
            files = pyspedas.omni.data(trange=trange,
                                       level='hro',
                                       time_clip=True,
                                       downloadonly=True)
            if not files:
                return None

            all_times = []
            all_B = []
            all_V = []
            all_n = []
            all_T = []

            for cdf_file in files:
                cdf = safe_cdf_read(cdf_file)
                if not cdf:
                    continue
                try:
                    cdf_info = cdf.cdf_info()
                    vars_list = cdf_info.zVariables

                    if 'Epoch' in vars_list:
                        epoch = cdf.varget('Epoch')
                        times_dt = cdflib.cdfepoch.to_datetime(epoch)
                        all_times.extend(times_dt)
                    else:
                        continue

                    # B-field
                    b_data = None
                    for b_var in ['BX_GSE', 'BY_GSE', 'BZ_GSE', 'B']:
                        if b_var in vars_list:
                            try:
                                if b_var == 'B':
                                    b_data = cdf.varget(b_var)
                                    if b_data.ndim == 1:
                                        b_data = b_data.reshape(-1, 1)
                                    break
                                else:
                                    bx = cdf.varget('BX_GSE')
                                    by = cdf.varget('BY_GSE')
                                    bz = cdf.varget('BZ_GSE')
                                    b_data = np.column_stack([bx, by, bz])
                                    break
                            except Exception:
                                continue

                    # Velocity
                    v_data = None
                    for v_var in ['flow_speed', 'V', 'Vx']:
                        if v_var in vars_list:
                            try:
                                v_data = cdf.varget(v_var)
                                if len(v_data.shape) == 1:
                                    v_data = v_data.reshape(-1, 1)
                                break
                            except Exception:
                                continue

                    # Density
                    n_data = None
                    for n_var in ['proton_density', 'Np', 'N']:
                        if n_var in vars_list:
                            try:
                                n_data = cdf.varget(n_var)
                                break
                            except Exception:
                                continue

                    # Temperature
                    t_data = None
                    for t_var in ['T', 'Tp', 'proton_temperature']:
                        if t_var in vars_list:
                            try:
                                t_data = cdf.varget(t_var)
                                break
                            except Exception:
                                continue

                    if b_data is not None:
                        all_B.append(b_data)
                    if v_data is not None:
                        all_V.append(v_data)
                    if n_data is not None:
                        all_n.append(n_data)
                    if t_data is not None:
                        all_T.append(t_data)

                except Exception as e:
                    print(f"    Err OMNI {os.path.basename(cdf_file)}: {e}")
                    continue

            if not all_times:
                return None

            time_arr = ensure_datetime_array(all_times)
            result = {"time": time_arr}

            # B
            if all_B:
                B_combined = np.vstack(all_B)
                B_clean = _clean_omni_array(B_combined, 200.0, "L1 B_gse", "nT")
                result["B_gse"] = B_clean
                if B_clean.ndim == 1 or B_clean.shape[1] == 1:
                    _dbg_stats("L1 B (clean)", B_clean, "nT",
                               expected_min=-100, expected_max=100)
                else:
                    Bmag_clean = np.linalg.norm(B_clean, axis=1)
                    _dbg_stats("L1 |B| (clean)", Bmag_clean, "nT",
                               expected_min=0, expected_max=100)

            # V
            if all_V:
                V_combined = np.vstack(all_V)
                V_clean = _clean_omni_array(V_combined, 3000.0, "L1 V", "km/s")
                result["V"] = V_clean
                Vmag_clean = np.linalg.norm(V_clean, axis=1) if V_clean.ndim > 1 else V_clean
                _dbg_stats("L1 V (clean)", Vmag_clean, "km/s",
                           expected_min=0, expected_max=2000)

            # n
            if all_n:
                n_combined = np.hstack(all_n).astype(float)
                n_clean = _clean_omni_array(n_combined, 1000.0, "L1 n", "cm^-3")
                result["n"] = n_clean
                _dbg_stats("L1 n (clean)", n_clean, "cm^-3",
                           expected_min=0, expected_max=100)

            # T
            if all_T:
                T_combined = np.hstack(all_T).astype(float)
                T_clean = _clean_omni_array(T_combined, 1e4, "L1 T", "eV")
                result["T"] = T_clean
                _dbg_stats("L1 T (clean)", T_clean, "eV",
                           expected_min=0, expected_max=2000)

            return result if len(result) > 1 else None

        except Exception as e:
            print(f"    OMNI error: {e}")
            return None

    @staticmethod
    def get_l1_position(time_dt, epoch_date):
        earth_pos = SunEarthLineAnalyzer.earth_position_at_time(time_dt, epoch_date)
        earth_sun_dir = -earth_pos / np.linalg.norm(earth_pos)
        l1_pos = earth_pos + earth_sun_dir * (0.01 * AU_KM)
        return l1_pos


class BallisticPropagation:
    @staticmethod
    def calculate_propagation_delay(sc_pos, l1_pos, v_sw):
        delta_r = l1_pos - sc_pos
        v_mag = np.linalg.norm(v_sw)
        if v_mag < 1.0:
            v_mag = 400.0
            v_sw = np.array([v_mag, 0, 0])

        v_hat = v_sw / v_mag
        # along_flow = np.dot(delta_r, v_hat)  # not used currently
        full_distance = np.linalg.norm(delta_r)
        delay_seconds = full_distance / v_mag
        return delay_seconds, full_distance

    @staticmethod
    def apply_ballistic_shift_to_l1(crossover_data, l1_data, sc_ephemeris, epoch_date):
        if not l1_data or not crossover_data:
            return None

        sc_times_sec = np.array([t.timestamp() for t in sc_ephemeris["times"]])
        sc_positions = sc_ephemeris["positions"]

        crossover_times_sec = np.array([t.timestamp() for t in crossover_data["time"]])

        sc_pos_interp = np.zeros((len(crossover_times_sec), 3))
        for i in range(3):
            f_pos = interp1d(sc_times_sec, sc_positions[:, i], kind="linear",
                             bounds_error=False, fill_value=np.nan)
            sc_pos_interp[:, i] = f_pos(crossover_times_sec)

        l1_positions = np.array([
            L1DataLoader.get_l1_position(t, epoch_date)
            for t in crossover_data["time"]
        ])

        propagation_delays = np.zeros(len(crossover_times_sec))
        propagation_distances = np.zeros(len(crossover_times_sec))

        for i in range(len(crossover_times_sec)):
            if np.any(np.isnan(sc_pos_interp[i])):
                propagation_delays[i] = np.nan
                continue
            v_sw = crossover_data["V"][i]
            delay_sec, dist_km = BallisticPropagation.calculate_propagation_delay(
                sc_pos_interp[i], l1_positions[i], v_sw
            )
            propagation_delays[i] = delay_sec
            propagation_distances[i] = dist_km

        l1_times_sec = np.array([t.timestamp() for t in l1_data["time"]])
        shifted_times_sec = crossover_times_sec + propagation_delays

        result = {
            "time": crossover_data["time"],
            "propagation_delay_hours": propagation_delays / 3600.0,
            "propagation_distance_au": propagation_distances / AU_KM,
        }

        if "B_gse" in l1_data and l1_data["B_gse"] is not None:
            B_l1 = l1_data["B_gse"]
            if len(B_l1.shape) == 1:
                B_l1 = B_l1.reshape(-1, 1)

            B_shifted = np.zeros((len(shifted_times_sec), B_l1.shape[1]))
            for j in range(B_l1.shape[1]):
                f_b = interp1d(l1_times_sec, B_l1[:, j], kind="linear",
                               bounds_error=False, fill_value=np.nan)
                B_shifted[:, j] = f_b(shifted_times_sec)
            result["B_gse_shifted"] = B_shifted

        if "V" in l1_data and l1_data["V"] is not None:
            V_l1 = l1_data["V"]
            if len(V_l1.shape) == 1:
                V_l1 = V_l1.reshape(-1, 1)

            V_shifted = np.zeros((len(shifted_times_sec), V_l1.shape[1]))
            for j in range(V_l1.shape[1]):
                f_v = interp1d(l1_times_sec, V_l1[:, j], kind="linear",
                               bounds_error=False, fill_value=np.nan)
                V_shifted[:, j] = f_v(shifted_times_sec)
            result["V_shifted"] = V_shifted

        if "n" in l1_data and l1_data["n"] is not None:
            f_n = interp1d(l1_times_sec, l1_data["n"], kind="linear",
                           bounds_error=False, fill_value=np.nan)
            result["n_shifted"] = f_n(shifted_times_sec)

        if "T" in l1_data and l1_data["T"] is not None:
            f_t = interp1d(l1_times_sec, l1_data["T"], kind="linear",
                           bounds_error=False, fill_value=np.nan)
            result["T_shifted"] = f_t(shifted_times_sec)

        has_b = "B_gse_shifted" in result and not np.all(np.isnan(result["B_gse_shifted"]))
        if not has_b:
            return None
        return result

# ------------------------ Solo in-situ + geomag -------------------------- #

class SoloDataLoader:
    @staticmethod
    def load_mag(t0_dt, t1_dt):
        """Solar Orbiter MAG with per-file decimation + debug on ranges."""
        try:
            trange = [t0_dt.strftime("%Y-%m-%d/%H:%M:%S"),
                      t1_dt.strftime("%Y-%m-%d/%H:%M:%S")]
            files = pyspedas.projects.solo.mag(
                trange=trange,
                datatype="rtn-normal",
                level="l2",
                time_clip=True,
                downloadonly=True,
            )
            if not files:
                print("    [DBG] MAG: no files returned")
                return None

            print(f"    Reading {len(files)} MAG files (decimating to ~{MAG_TARGET_POINTS_PER_DAY} pts/day)...")
            all_times = []
            all_B = []

            for cdf_file in files:
                cdf = safe_cdf_read(cdf_file)
                if not cdf:
                    continue
                try:
                    cdf_info  = cdf.cdf_info()
                    vars_list = cdf_info.zVariables

                    b_var = next((v for v in vars_list
                                  if 'B_RTN' in v or 'B_rtn' in v), None)

                    if b_var and 'EPOCH' in vars_list:
                        epoch = cdf.varget('EPOCH')
                        b_data = cdf.varget(b_var)

                        n = len(epoch)
                        if n == 0:
                            continue

                        # decimate per file
                        step = max(1, n // MAG_TARGET_POINTS_PER_DAY)
                        idx = np.arange(0, n, step, dtype=int)

                        epoch_sel = epoch[idx]
                        b_sel     = b_data[idx, :] if b_data.ndim > 1 else b_data[idx]

                        times_dt = cdflib.cdfepoch.to_datetime(epoch_sel)
                        all_times.extend(times_dt)
                        all_B.append(b_sel)

                        # per-file debug
                        if b_sel.ndim == 1:
                            bmag = np.abs(b_sel)
                        else:
                            bmag = np.linalg.norm(b_sel, axis=1)
                        print(
                            f"      [DBG] MAG {os.path.basename(cdf_file)}: "
                            f"{len(epoch)}→{len(epoch_sel)} pts, "
                            f"|B| range {np.nanmin(bmag):.2f}–{np.nanmax(bmag):.2f} nT"
                        )

                except Exception as e:
                    print(f"    Err MAG {os.path.basename(cdf_file)}: {e}")
                    continue

            if not all_B:
                print("    [DBG] MAG: all_B empty after reading files")
                return None

            B_combined = np.vstack(all_B)
            times_arr  = ensure_datetime_array(all_times)

            # global debug
            if B_combined.ndim == 1:
                Bmag = np.abs(B_combined)
            else:
                Bmag = np.linalg.norm(B_combined, axis=1)

            print(
                f"    ✓ MAG (decimated): {len(times_arr)} pts, B shape={B_combined.shape}"
            )
            print(
                f"    [DBG] MAG combined |B| range: {np.nanmin(Bmag):.2f}–{np.nanmax(Bmag):.2f} nT"
            )

            if np.nanmax(Bmag) > 1e3:
                print("    [WARN] MAG |B| > 1000 nT -> units/scale suspect.")

            return {"time": times_arr, "B": B_combined}

        except Exception as e:
            print(f"    MAG error: {e}")
            return None

    @staticmethod
    def load_swa_moments(t0_dt, t1_dt):
        """
        Load Solar Orbiter SWA PAS ground moments (L2 PAS-GRND-MOM).
        Reads CDF files directly since pyspedas.projects.solo.swa() doesn't load into pytplot.
        
        Returns:
            {
                "time": list of datetime,
                "n": 1D array (density),
                "V": 2D array [ntimes, 3] (velocity RTN),
                "T": 1D array (temperature)
            }
            or None on failure.
        """
        try:
            trange = [
                t0_dt.strftime("%Y-%m-%d/%H:%M:%S"),
                t1_dt.strftime("%Y-%m-%d/%H:%M:%S")
            ]

            # Download files (this part works)
            files = pyspedas.projects.solo.swa(
                trange=trange,
                datatype="pas-grnd-mom",
                level="l2",
                time_clip=True,
                downloadonly=True  # Just download, don't try to load
            )
            
            if not files:
                print("    [MOMENTS] No files downloaded")
                return None

            print(f"    Reading {len(files)} SWA-PAS-GRND-MOM files...")
            
            all_times = []
            all_n = []
            all_v = []
            all_t = []

            for cdf_file in files:
                cdf = safe_cdf_read(cdf_file)
                if not cdf:
                    continue
                
                try:
                    # Read variables directly from CDF
                    epoch = cdf.varget('Epoch')
                    times_dt = cdflib.cdfepoch.to_datetime(epoch)
                    
                    n_data = cdf.varget('N')          # Density
                    v_data = cdf.varget('V_RTN')      # Velocity [n×3]
                    t_data = cdf.varget('T')          # Temperature
                    
                    # Accumulate data
                    all_times.extend(times_dt)
                    all_n.extend(n_data)
                    all_v.extend(v_data)
                    all_t.extend(t_data)
                    
                except Exception as e:
                    print(f"    Warning: Could not read {os.path.basename(cdf_file)}: {e}")
                    continue

            if not all_times:
                print("    [MOMENTS] No data read from files")
                return None

            # Convert to arrays
            times_array = ensure_datetime_array(all_times)
            n_array = np.array(all_n, dtype=float)
            v_array = np.array(all_v, dtype=float)
            t_array = np.array(all_t, dtype=float)
            
            # Sort by time (in case files were out of order)
            sort_idx = np.argsort([t.timestamp() for t in times_array])
            times_array = times_array[sort_idx]
            n_array = n_array[sort_idx]
            v_array = v_array[sort_idx]
            t_array = t_array[sort_idx]
            
            print(f"    ✓ SWA moments: {len(times_array)} points")
            _dbg_stats("SWA n", n_array, "cm^-3", expected_min=0, expected_max=100)
            _dbg_stats("SWA |V|", np.linalg.norm(v_array, axis=1), "km/s", 
                    expected_min=0, expected_max=1000)
            _dbg_stats("SWA T", t_array, "eV", expected_min=0, expected_max=500)

            return {
                "time": times_array,
                "n": n_array,
                "V": v_array,
                "T": t_array
            }

        except Exception as e:
            print(f"    [MOMENTS] error: {e}")
            return None

    @staticmethod
    def load_swa_eflux(t0_dt, t1_dt):
        """
        Load Solar Orbiter SWA PAS energy flux (L2 PAS-EFLUX).
        Reads CDF files directly since pyspedas.projects.solo.swa() doesn't load into pytplot.
        
        Returns:
            {
                "time": list of datetime,
                "eflux": 2D array [ntimes, nenergies],
                "energy": 1D array [nenergies] (energy bin centers)
            }
            or None on failure.
        """
        try:
            trange = [
                t0_dt.strftime("%Y-%m-%d/%H:%M:%S"),
                t1_dt.strftime("%Y-%m-%d/%H:%M:%S")
            ]

            # Download files (this part works)
            files = pyspedas.projects.solo.swa(
                trange=trange,
                datatype="pas-eflux",
                level="l2",
                time_clip=True,
                downloadonly=True  # Just download, don't try to load
            )
            
            if not files:
                print("    [EFLUX] No files downloaded")
                return None

            print(f"    Reading {len(files)} SWA-PAS-EFLUX files...")
            
            all_times = []
            all_eflux = []
            energy_bins = None

            for cdf_file in files:
                cdf = safe_cdf_read(cdf_file)
                if not cdf:
                    continue
                
                try:
                    # Read variables directly from CDF
                    epoch = cdf.varget('Epoch')
                    times_dt = cdflib.cdfepoch.to_datetime(epoch)
                    
                    eflux_data = cdf.varget('eflux')  # Shape: [ntimes, nenergies]
                    
                    # Get energy bins (should be same for all files, so just grab once)
                    if energy_bins is None:
                        energy_bins = cdf.varget('Energy')
                    
                    # Accumulate data
                    all_times.extend(times_dt)
                    all_eflux.extend(eflux_data)
                    
                except Exception as e:
                    print(f"    Warning: Could not read {os.path.basename(cdf_file)}: {e}")
                    continue

            if not all_times:
                print("    [EFLUX] No data read from files")
                return None

            # Convert to arrays
            times_array = ensure_datetime_array(all_times)
            eflux_array = np.array(all_eflux, dtype=float)
            
            # Sort by time
            sort_idx = np.argsort([t.timestamp() for t in times_array])
            times_array = times_array[sort_idx]
            eflux_array = eflux_array[sort_idx]
            
            print(f"    ✓ SWA eflux: {len(times_array)} times × {len(energy_bins)} energies")
            
            return {
                "time": times_array,
                "eflux": eflux_array,
                "energy": energy_bins
            }

        except Exception as e:
            print(f"    [EFLUX] error: {e}")
            return None
        
    @staticmethod
    def load_kyoto_dst(t0_dt, t1_dt):
        """Dst via OMNI high-res."""
        try:
            trange = [t0_dt.strftime("%Y-%m-%d/%H:%M:%S"),
                      t1_dt.strftime("%Y-%m-%d/%H:%M:%S")]
            files = pyspedas.omni.data(trange=trange,
                                       level='hro',
                                       time_clip=True,
                                       downloadonly=True)
            if not files:
                return None

            all_times = []
            all_dst = []

            for cdf_file in files:
                cdf = safe_cdf_read(cdf_file)
                if not cdf:
                    continue
                try:
                    cdf_info = cdf.cdf_info()
                    vars_list = cdf_info.zVariables
                    dst_var = next((v for v in vars_list
                                    if 'DST' in v or 'Dst' in v or 'dst' in v), None)
                    if dst_var and 'Epoch' in vars_list:
                        epoch = cdf.varget('Epoch')
                        dst_data = cdf.varget(dst_var)
                        times_dt = cdflib.cdfepoch.to_datetime(epoch)
                        all_times.extend(times_dt)
                        all_dst.append(dst_data)
                except Exception:
                    continue

            if not all_dst:
                return None

            print("    ✓ Dst loaded")
            return {
                "time": ensure_datetime_array(all_times),
                "dst":  np.hstack(all_dst)
            }

        except Exception as e:
            print(f"    Dst error: {e}")
            return None

    @staticmethod
    def load_kp_index(t0_dt, t1_dt):
        """Kp via OMNI (if present)."""
        try:
            trange = [t0_dt.strftime("%Y-%m-%d/%H:%M:%S"),
                      t1_dt.strftime("%Y-%m-%d/%H:%M:%S")]
            files = pyspedas.omni.data(trange=trange,
                                       level='hro',
                                       time_clip=True,
                                       downloadonly=True)
            if not files:
                return None

            all_times = []
            all_kp = []

            for cdf_file in files:
                cdf = safe_cdf_read(cdf_file)
                if not cdf:
                    continue
                try:
                    cdf_info = cdf.cdf_info()
                    vars_list = cdf_info.zVariables
                    kp_var = next((v for v in vars_list
                                   if 'kp' in v.lower() or 'Kp' in v or 'KP' in v), None)
                    if kp_var and 'Epoch' in vars_list:
                        epoch = cdf.varget('Epoch')
                        kp_data = cdf.varget(kp_var)
                        times_dt = cdflib.cdfepoch.to_datetime(epoch)
                        all_times.extend(times_dt)
                        all_kp.append(kp_data)
                except Exception:
                    continue

            if not all_kp:
                return None

            print("    ✓ Kp loaded")
            return {
                "time": ensure_datetime_array(all_times),
                "kp":   np.hstack(all_kp).flatten()
            }

        except Exception as e:
            print(f"    Kp error: {e}")
            return None

    @staticmethod
    def interpolate_to_mag_time(mag_data, swa_moments):
        """Interpolate SWA moments to MAG times + debug ranges."""
        mag_t = np.array([t.timestamp() for t in mag_data["time"]])
        swa_t = np.array([t.timestamp() for t in swa_moments["time"]])

        if mag_t.size == 0 or swa_t.size == 0:
            print("    [DBG] interpolate_to_mag_time: empty time arrays")
            return None

        if mag_t[0] > swa_t[-1] or mag_t[-1] < swa_t[0]:
            print("    [DBG] interpolate_to_mag_time: no time overlap between MAG and SWA")
            return None

        print(
            "    [DBG] interpolate_to_mag_time:"
            f" MAG {len(mag_t)} pts ({datetime.utcfromtimestamp(mag_t[0])} → {datetime.utcfromtimestamp(mag_t[-1])}),"
            f" SWA {len(swa_t)} pts ({datetime.utcfromtimestamp(swa_t[0])} → {datetime.utcfromtimestamp(swa_t[-1])})"
        )

        V_interp = np.zeros((len(mag_t), 3))
        for i in range(3):
            f = interp1d(
                swa_t, swa_moments["V"][:, i], kind="linear",
                bounds_error=False, fill_value=np.nan
            )
            V_interp[:, i] = f(mag_t)

        f_n = interp1d(
            swa_t, swa_moments["n"], kind="linear",
            bounds_error=False, fill_value=np.nan
        )
        n_interp = f_n(mag_t)

        valid = ~(np.isnan(V_interp).any(axis=1) | np.isnan(n_interp))
        if not valid.any():
            print("    [DBG] interpolate_to_mag_time: no valid points after interpolation (all NaN)")
            return None

        # trim to valid indices
        mag_times_valid = [mag_data["time"][i] for i in range(len(mag_t)) if valid[i]]
        B_valid         = mag_data["B"][valid]
        V_valid         = V_interp[valid]
        n_valid         = n_interp[valid]

        # debug ranges: MAG B, SWA V, n
        if B_valid.ndim == 1:
            Bmag = np.abs(B_valid)
        else:
            Bmag = np.linalg.norm(B_valid, axis=1)
        Vmag = np.linalg.norm(V_valid, axis=1)

        print(
            f"    [DBG] common (MAG+SWA) points: {len(mag_times_valid)}"
        )
        print(
            f"    [DBG] MAG |B| range on common grid: {np.nanmin(Bmag):.2f}–{np.nanmax(Bmag):.2f} nT"
        )
        print(
            f"    [DBG] SWA V_mag range on MAG grid: {np.nanmin(Vmag):.1f}–{np.nanmax(Vmag):.1f} km/s"
        )
        print(
            f"    [DBG] SWA n range on MAG grid: {np.nanmin(n_valid):.2f}–{np.nanmax(n_valid):.2f} cm^-3"
        )

        if np.nanmax(Vmag) > 2_000:
            print("    [WARN] SWA V_mag on MAG grid > 2000 km/s -> units suspect.")
        if np.nanmax(n_valid) > 1_000:
            print("    [WARN] SWA n on MAG grid > 1000 cm^-3 -> units/flags suspect.")

        return {
            "time": mag_times_valid,
            "B":    B_valid,
            "V":    V_valid,
            "n":    n_valid,
        }

    @staticmethod
    def transform_rtn_to_gse(B_rtn, spacecraft_positions, times_dt, epoch_date):
        N = len(B_rtn)
        B_gse = np.zeros_like(B_rtn)

        for i in range(N):
            sc_pos = spacecraft_positions[i]
            earth_pos = SunEarthLineAnalyzer.earth_position_at_time(times_dt[i], epoch_date)

            sc_xy = sc_pos[:2]
            theta_sc = np.arctan2(sc_xy[1], sc_xy[0])
            R_hat = np.array([np.cos(theta_sc), np.sin(theta_sc), 0])
            T_hat = np.array([-np.sin(theta_sc), np.cos(theta_sc), 0])
            N_hat = np.array([0, 0, 1])

            earth_xy = earth_pos[:2]
            theta_earth = np.arctan2(earth_xy[1], earth_xy[0])
            X_hat = np.array([-np.cos(theta_earth), -np.sin(theta_earth), 0])
            Y_hat = np.array([np.sin(theta_earth), -np.cos(theta_earth), 0])
            Z_hat = np.array([0, 0, 1])

            R = np.array([
                [np.dot(X_hat, R_hat), np.dot(X_hat, T_hat), np.dot(X_hat, N_hat)],
                [np.dot(Y_hat, R_hat), np.dot(Y_hat, T_hat), np.dot(Y_hat, N_hat)],
                [np.dot(Z_hat, R_hat), np.dot(Z_hat, T_hat), np.dot(Z_hat, N_hat)],
            ])

            B_gse[i] = R @ B_rtn[i]

        return B_gse

    @staticmethod
    def average_to_3hour_bins(times_dt, values):
        if len(times_dt) == 0:
            return {"time": [], "values": np.array([])}

        start_time = times_dt[0].replace(hour=(times_dt[0].hour // 3) * 3,
                                         minute=0, second=0, microsecond=0)
        end_time = times_dt[-1]

        bin_times = []
        bin_values = []

        current_bin_start = start_time
        while current_bin_start <= end_time:
            current_bin_end = current_bin_start + timedelta(hours=3)
            mask = [(current_bin_start <= t < current_bin_end) for t in times_dt]
            if any(mask):
                bin_values.append(np.mean(values[mask]))
                bin_times.append(current_bin_start + timedelta(hours=1.5))
            current_bin_start = current_bin_end

        return {"time": bin_times, "values": np.array(bin_values)}

    @staticmethod
    def predict_dst_obrien(B_rtn, V_rtn, n, times_dt):
        a        = -4.4
        b        = 0.5
        tau_main = 7.7 * 3600

        V_mag = np.linalg.norm(V_rtn, axis=1)
        Bn    = B_rtn[:, 2]
        Bs    = np.where(Bn < 0, -Bn, 0)
        Ey    = V_mag * Bs / 1000.0

        Q      = np.where(Ey > b, a * (Ey - b), 0.0)
        Q_sec  = Q / 3600.0

        m_p  = 1.67e-27
        n_si = n * 1e6
        V_si = V_mag * 1000.0
        Pdyn = m_p * n_si * V_si**2 * 1e9

        P0            = 2.0
        P_correction  = 7.26 * np.sqrt(Pdyn) - 7.26 * np.sqrt(P0)

        tsec = np.diff([t.timestamp() for t in times_dt])
        tsec = np.insert(tsec, 0, tsec[0] if len(tsec) > 0 else 60.0)

        Dst = np.zeros(len(times_dt))
        for i in range(1, len(times_dt)):
            dDst  = (Q_sec[i] - Dst[i - 1] / tau_main) * tsec[i]
            Dst[i] = Dst[i - 1] + dDst

        Dst_pred = Dst + P_correction
        return {"time": times_dt, "dst_predicted": Dst_pred}

    @staticmethod
    def predict_kp_newell(B_rtn, V_rtn, times_dt, spacecraft_positions, epoch_date):
        B_gse = SoloDataLoader.transform_rtn_to_gse(B_rtn, spacecraft_positions,
                                                    times_dt, epoch_date)

        B_Y = B_gse[:, 1]
        B_Z = B_gse[:, 2]
        theta_clock = np.arctan2(B_Y, B_Z)
        B_T = np.sqrt(B_Y**2 + B_Z**2)

        V_mag = np.linalg.norm(V_rtn, axis=1)

        sin_half_theta = np.abs(np.sin(theta_clock / 2.0))
        V_safe = np.maximum(V_mag, 1.0)
        B_T_safe = np.maximum(B_T, 0.1)

        dPhi_dt = (V_safe**(4.0/3.0)) * (B_T_safe**(2.0/3.0)) * (sin_half_theta**8) * 1e-4

        a_kp = 0.3
        b_kp = 2.5

        Kp_raw = a_kp + b_kp * np.log10(dPhi_dt + 1e-10)
        Kp_instantaneous = np.clip(Kp_raw, 0, 9)

        Kp_binned = SoloDataLoader.average_to_3hour_bins(times_dt, Kp_instantaneous)

        return {
            "time": Kp_binned["time"],
            "kp_predicted": Kp_binned["values"]
        }

    @staticmethod
    def apply_propagation_delay(common_data, dst_measured, spacecraft_ephemeris, epoch_date):
        if not dst_measured or not common_data:
            return None

        sc_times_sec = np.array([t.timestamp() for t in spacecraft_ephemeris["times"]])
        sc_positions = spacecraft_ephemeris["positions"]

        common_times_sec = np.array([t.timestamp() for t in common_data["time"]])

        sc_pos_interp = np.zeros((len(common_times_sec), 3))
        for i in range(3):
            f_pos = interp1d(sc_times_sec, sc_positions[:, i], kind="linear",
                             bounds_error=False, fill_value=np.nan)
            sc_pos_interp[:, i] = f_pos(common_times_sec)

        earth_positions = np.array([
            SunEarthLineAnalyzer.earth_position_at_time(t, epoch_date)
            for t in common_data["time"]
        ])

        distances = np.linalg.norm(sc_pos_interp - earth_positions, axis=1)
        V_mag = np.linalg.norm(common_data["V"], axis=1)
        propagation_delays_sec = distances / V_mag
        propagation_delays_hours = propagation_delays_sec / 3600.0

        dst_times_sec = np.array([t.timestamp() for t in dst_measured["time"]])
        dst_values = dst_measured["dst"].flatten() if len(dst_measured["dst"].shape) > 1 else dst_measured["dst"]

        f_dst = interp1d(dst_times_sec, dst_values, kind="linear",
                         bounds_error=False, fill_value=np.nan)

        shifted_times_sec = common_times_sec + propagation_delays_sec
        dst_shifted = f_dst(shifted_times_sec)

        valid = ~np.isnan(dst_shifted)
        if not valid.any():
            return None

        return {
            "time": [common_data["time"][i] for i in range(len(valid)) if valid[i]],
            "dst_shifted": dst_shifted[valid],
            "propagation_delay_hours": propagation_delays_hours[valid]
        }

    @staticmethod
    def apply_propagation_delay_kp(common_data, kp_measured, spacecraft_ephemeris, epoch_date):
        if not kp_measured or not common_data:
            return None

        sc_times_sec = np.array([t.timestamp() for t in spacecraft_ephemeris["times"]])
        sc_positions = spacecraft_ephemeris["positions"]

        common_times_sec = np.array([t.timestamp() for t in common_data["time"]])

        sc_pos_interp = np.zeros((len(common_times_sec), 3))
        for i in range(3):
            f_pos = interp1d(sc_times_sec, sc_positions[:, i], kind="linear",
                             bounds_error=False, fill_value=np.nan)
            sc_pos_interp[:, i] = f_pos(common_times_sec)

        earth_positions = np.array([
            SunEarthLineAnalyzer.earth_position_at_time(t, epoch_date)
            for t in common_data["time"]
        ])

        distances = np.linalg.norm(sc_pos_interp - earth_positions, axis=1)
        V_mag = np.linalg.norm(common_data["V"], axis=1)
        propagation_delays_sec = distances / V_mag

        kp_times_sec = np.array([t.timestamp() for t in kp_measured["time"]])
        kp_values = kp_measured["kp"].flatten() if len(kp_measured["kp"].shape) > 1 else kp_measured["kp"]

        f_kp = interp1d(kp_times_sec, kp_values, kind="linear",
                        bounds_error=False, fill_value=np.nan)

        shifted_times_sec = common_times_sec + propagation_delays_sec
        kp_shifted = f_kp(shifted_times_sec)

        valid = ~np.isnan(kp_shifted)
        if not valid.any():
            return None

        return {
            "time": [common_data["time"][i] for i in range(len(valid)) if valid[i]],
            "kp_shifted": kp_shifted[valid]
        }

    @staticmethod
    def preload_all(events, spacecraft_ephemeris, epoch_date, window_hours=72):
        all_data = {}
        for event in tqdm(events, desc="Loading", ncols=80):
            t_ev  = event["time"]
            t0_dt = t_ev - timedelta(hours=window_hours)
            t1_dt = t_ev + timedelta(hours=window_hours)

            print(f"\n  {t_ev}:")
            print(f"    In-situ window: {t0_dt} → {t1_dt}")

            mag  = SoloDataLoader.load_mag(t0_dt, t1_dt)
            swam = SoloDataLoader.load_swa_moments(t0_dt, t1_dt)
            swae = SoloDataLoader.load_swa_eflux(t0_dt, t1_dt)

            ext_t0 = t0_dt
            ext_t1 = t1_dt + timedelta(days=5)
            print(f"    Extended window for Dst/Kp/L1: {ext_t0} → {ext_t1}")

            dst = SoloDataLoader.load_kyoto_dst(ext_t0, ext_t1)
            kp  = SoloDataLoader.load_kp_index(ext_t0, ext_t1)
            l1  = L1DataLoader.load_omni_data(ext_t0, ext_t1)

            dst_pred = None
            dst_shifted = None
            kp_pred = None
            kp_shifted = None
            l1_shifted = None

            if mag and swam and ("V" in swam) and ("n" in swam):
                common = SoloDataLoader.interpolate_to_mag_time(mag, swam)
                if common:
                    dst_pred = SoloDataLoader.predict_dst_obrien(
                        common["B"], common["V"], common["n"], common["time"]
                    )
                    dst_shifted = SoloDataLoader.apply_propagation_delay(
                        common, dst, spacecraft_ephemeris, epoch_date
                    )

                    sc_times_sec = np.array([t.timestamp() for t in spacecraft_ephemeris["times"]])
                    sc_positions = spacecraft_ephemeris["positions"]
                    common_times_sec = np.array([t.timestamp() for t in common["time"]])

                    sc_pos_interp = np.zeros((len(common_times_sec), 3))
                    for i in range(3):
                        f_pos = interp1d(sc_times_sec, sc_positions[:, i],
                                         kind="linear", bounds_error=False,
                                         fill_value=np.nan)
                        sc_pos_interp[:, i] = f_pos(common_times_sec)

                    kp_pred = SoloDataLoader.predict_kp_newell(
                        common["B"], common["V"], common["time"],
                        sc_pos_interp, epoch_date
                    )
                    kp_shifted = SoloDataLoader.apply_propagation_delay_kp(
                        common, kp, spacecraft_ephemeris, epoch_date
                    )

                    if l1:
                        l1_shifted = BallisticPropagation.apply_ballistic_shift_to_l1(
                            common, l1, spacecraft_ephemeris, epoch_date
                        )

            all_data[t_ev] = {
                "mag":           mag,
                "swa":           swae,
                "swa_moments":   swam,
                "dst_measured":  dst,
                "dst_predicted": dst_pred,
                "dst_shifted":   dst_shifted,
                "kp_measured":   kp,
                "kp_predicted":  kp_pred,
                "kp_shifted":    kp_shifted,
                "l1_data":       l1,
                "l1_shifted":    l1_shifted,
            }

        return all_data

# ------------------------------- Plotting -------------------------------- #

class Plotter:
    @staticmethod
    def plot_constellation_overview(events, constellation):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_aspect("equal")

        ax.plot(0, 0, "o", ms=16, color="#FDB813", mec="#F37021", mew=2,
                label="Sun", zorder=10)

        theta = np.linspace(0, 2 * np.pi, 1000)
        ax.plot(np.cos(theta), np.sin(theta), "b--", lw=1.5, alpha=0.4,
                label="Earth orbit")

        colors = ["#E74C3C", "#3498DB", "#2ECC71"]
        for i, sat in enumerate(constellation.satellites):
            orbit = sat["orbit"]["dro_helio"] / AU_KM
            ax.plot(orbit[:, 0], orbit[:, 1], color=colors[i], lw=2, alpha=0.7,
                    label=f"DRO-{sat['id']} ({sat['rotation_deg']}°)")

        sc_colors  = {"STEREO-A": "#FF6B6B", "Solar Orbiter": "#45B7D1"}
        plotted_sc = set()

        for sc_name, sc_events in events.items():
            if not sc_events:
                continue
            color = sc_colors.get(sc_name, "#888888")
            for ev in sc_events:
                label = sc_name if sc_name not in plotted_sc else None
                plotted_sc.add(sc_name)
                pos = ev["spacecraft_pos"][:2] / AU_KM
                ax.plot(pos[0], pos[1], "o", ms=8, color=color, mec="white", mew=1.5,
                        alpha=0.9, label=label)
                text = f"{ev['time'].strftime('%Y-%m-%d')}\nDRO-{ev['dro_id']}"
                ax.annotate(text, xy=(pos[0], pos[1]), xytext=(5, 5),
                            textcoords="offset points",
                            fontsize=8, alpha=0.8,
                            bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="white", alpha=0.8))

        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_xlabel("X [AU]")
        ax.set_ylabel("Y [AU]")
        ax.set_title("Hénon DRO Constellation — Crossover Overview")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_event_detail(event, spacecraft_name, constellation,
                          spacecraft_data, epoch_date,
                          context_days=TRACK_CONTEXT_DAYS):
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
        ax1.set_aspect("equal")

        dro_sat = None
        for sat in constellation.satellites:
            if sat["id"] == event["dro_id"]:
                dro_sat = sat
                break
        if not dro_sat:
            return None

        dro_orbit = dro_sat["orbit"]["dro_helio"] / AU_KM

        ax1.plot(0, 0, "o", ms=12, color="#FDB813", mec="#F37021", mew=1.5,
                 label="Sun")
        theta = np.linspace(0, 2 * np.pi, 1000)
        ax1.plot(np.cos(theta), np.sin(theta), "b--", lw=1, alpha=0.2,
                 label="Earth orbit")

        earth_pos = event["earth_pos"] / AU_KM
        sc_pos    = event["spacecraft_pos"][:2] / AU_KM

        half = context_days / 2.0
        t0   = event["time"] - timedelta(days=half)
        t1   = event["time"] + timedelta(days=half)

        times = spacecraft_data["times"]
        pos   = spacecraft_data["positions"]
        mask  = [(t0 <= t <= t1) for t in times]

        radii = []

        if any(mask):
            sc_xy = (pos[mask] / AU_KM)[:, :2]
            ax1.plot(sc_xy[:, 0], sc_xy[:, 1], "-", lw=2, alpha=0.9,
                     label=f"{spacecraft_name} (±{half:.0f} d)")
            radii.extend(np.linalg.norm(sc_xy, axis=1).tolist())

            earth_xy = []
            for t in np.array(times)[mask]:
                epos = SunEarthLineAnalyzer.earth_position_at_time(t, epoch_date) / AU_KM
                earth_xy.append(epos[:2])
            earth_xy = np.array(earth_xy)
            ax1.plot(earth_xy[:, 0], earth_xy[:, 1], "-", lw=2, alpha=0.9,
                     label=f"Earth (±{half:.0f} d)")
            radii.extend(np.linalg.norm(earth_xy, axis=1).tolist())

        ax1.plot(dro_orbit[:, 0], dro_orbit[:, 1], "r-", lw=2, alpha=0.7,
                 label=f"DRO-{event['dro_id']}")
        radii.extend(np.linalg.norm(dro_orbit[:, :2], axis=1).tolist())

        ax1.plot(earth_pos[0], earth_pos[1], "o", ms=10, color="blue",
                 mec="white", mew=1.5, label="Earth @ event")
        ax1.plot(sc_pos[0], sc_pos[1], "*", ms=15, color="gold",
                 mec="black", mew=1.5, label=f"{spacecraft_name} @ event")
        ax1.plot([0, earth_pos[0]], [0, earth_pos[1]], "g--", lw=1.5,
                 alpha=0.5, label="Sun–Earth line")

        rmax = max(radii) if radii else max(np.linalg.norm(earth_pos[:2]),
                                            np.linalg.norm(sc_pos), 1.2)
        pad  = 0.10 * rmax
        span = rmax + pad

        ax1.set_xlim(-span, span)
        ax1.set_ylim(-span, span)
        ax1.set_xlabel("X [AU]")
        ax1.set_ylabel("Y [AU]")
        ax1.set_title(f"{spacecraft_name} — Heliocentric — {event['time'].strftime('%Y-%m-%d')}")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_event_detail_earth_frame(event, spacecraft_name,
                                      constellation, spacecraft_data,
                                      epoch_date,
                                      context_days=TRACK_CONTEXT_DAYS):
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
        ax1.set_aspect("equal")

        dro_sat = None
        for sat in constellation.satellites:
            if sat["id"] == event["dro_id"]:
                dro_sat = sat
                break
        if not dro_sat:
            return None

        dro_1           = constellation.satellites[0]
        dro_orbit_earth = dro_1["orbit"]["dro_earth"] / AU_KM

        ax1.plot(0, 0, "o", ms=12, color="#4A90E2", mec="white", mew=1.5,
                 label="Earth", zorder=10)

        half = context_days / 2.0
        t0   = event["time"] - timedelta(days=half)
        t1   = event["time"] + timedelta(days=half)

        times     = spacecraft_data["times"]
        pos_helio = spacecraft_data["positions"]
        mask      = [(t0 <= t <= t1) for t in times]

        radii = []

        if any(mask):
            sc_e = []
            for t, pxyz in zip(np.array(times)[mask], pos_helio[mask]):
                days  = (t - epoch_date).total_seconds() / 86400
                theta = MEAN_MOTION * days
                epos  = SunEarthLineAnalyzer.earth_position_at_time(t, epoch_date)
                rel   = pxyz - epos
                c, s  = np.cos(-theta), np.sin(-theta)
                x_e   = rel[0] * c - rel[1] * s
                y_e   = rel[0] * s + rel[1] * c
                sc_e.append([x_e, y_e])
            sc_e = np.array(sc_e) / AU_KM
            ax1.plot(sc_e[:, 0], sc_e[:, 1], "-", lw=2, color="#45B7D1",
                     alpha=0.9, label=f"{spacecraft_name} (±{half:.0f} d)")
            radii.extend(np.linalg.norm(sc_e, axis=1).tolist())

        ax1.plot(dro_orbit_earth[:, 0], dro_orbit_earth[:, 1], "r-", lw=2,
                 alpha=0.7, label=f"DRO-{event['dro_id']} ({dro_sat['rotation_deg']}°)")
        radii.extend(np.linalg.norm(dro_orbit_earth, axis=1).tolist())

        days_ev  = (event["time"] - epoch_date).total_seconds() / 86400
        theta_ev = MEAN_MOTION * days_ev

        sc_h    = event["spacecraft_pos"]
        e_h     = event["earth_pos"]
        rel_ev  = sc_h - e_h
        c_ev, s_ev = np.cos(-theta_ev), np.sin(-theta_ev)
        sc_x_e  = rel_ev[0] * c_ev - rel_ev[1] * s_ev
        sc_y_e  = rel_ev[0] * s_ev + rel_ev[1] * c_ev
        sc_e_pt = np.array([sc_x_e, sc_y_e]) / AU_KM

        ax1.plot(sc_e_pt[0], sc_e_pt[1], "*", ms=15, color="gold",
                 mec="black", mew=1.5, label=f"{spacecraft_name} @ event")

        sun_rel = -e_h
        sun_x_e = sun_rel[0] * c_ev - sun_rel[1] * s_ev
        sun_y_e = sun_rel[0] * s_ev + sun_rel[1] * c_ev
        sun_e   = np.array([sun_x_e, sun_y_e]) / AU_KM

        ax1.plot(sun_e[0], sun_e[1], "o", ms=14, color="#FDB813",
                 mec="#F37021", mew=2, label="Sun @ event", zorder=9)
        radii.append(np.linalg.norm(sun_e))

        ax1.plot([0, sun_e[0]], [0, sun_e[1]], "g--", lw=1.5,
                 alpha=0.5, label="Sun–Earth line")

        rmax = max(radii) if radii else 1.2
        pad  = 0.10 * rmax
        span = rmax + pad

        ax1.set_xlim(-span, span)
        ax1.set_ylim(-span, span)
        ax1.set_xlabel("X [AU] (Earth Frame)")
        ax1.set_ylabel("Y [AU] (Earth Frame)")
        ax1.set_title(f"{spacecraft_name} — Earth-Centered — {event['time'].strftime('%Y-%m-%d')}")
        ax1.legend(loc="best", fontsize=10)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_solo_insitu_with_dst(event, preloaded, window_hours=INSITU_WINDOW_HR):
        center = event.get("time")
        if not isinstance(center, datetime):
            print("[DBG] plot_solo_insitu_with_dst: event has no valid 'time'")
            return None

        d = preloaded.get(center)
        if not d:
            print(f"[DBG] plot_solo_insitu_with_dst: no preloaded data for {center}")
            return None

        mag          = d.get("mag")
        swa          = d.get("swa")
        swa_moments  = d.get("swa_moments")
        dst_measured = d.get("dst_measured")
        dst_pred     = d.get("dst_predicted")
        dst_shifted  = d.get("dst_shifted")
        kp_measured  = d.get("kp_measured")
        kp_pred      = d.get("kp_predicted")
        kp_shifted   = d.get("kp_shifted")
        l1_shifted   = d.get("l1_shifted")

        if not mag or "time" not in mag or "B" not in mag:
            print(f"[DBG] plot_solo_insitu_with_dst: MAG missing for {center}")
            return None

        # enforce datetimes
        T_mag_full = ensure_datetime_array(mag["time"])
        B_full     = mag["B"]

        t0 = center - timedelta(hours=window_hours)
        t1 = center + timedelta(hours=window_hours)

        mask_mag = (T_mag_full >= t0) & (T_mag_full <= t1)
        if not np.any(mask_mag):
            print(f"[DBG] plot_solo_insitu_with_dst: no MAG points in window for {center}")
            return None

        T_mag = T_mag_full[mask_mag]
        B     = B_full[mask_mag]

        max_points = 200_000
        if len(T_mag) > max_points:
            step = int(np.ceil(len(T_mag) / float(max_points)))
            T_mag = T_mag[::step]
            B     = B[::step]
            print(f"[DBG] plot_solo_insitu_with_dst: decimated MAG to {len(T_mag)} points (step={step})")
        else:
            print(f"[DBG] plot_solo_insitu_with_dst: using {len(T_mag)} MAG points")

        fig = plt.figure(figsize=(14, 22))
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(9, 1, figure=fig, hspace=0.3)
        axes = [fig.add_subplot(gs[i, 0]) for i in range(9)]

        # B components
        labels_B = ["Br", "Bt", "Bn"]
        colors_B = ["red", "green", "blue"]

        for i in range(min(3, B.shape[1])):
            axes[i].plot(T_mag, B[:, i], linewidth=1.2, color=colors_B[i],
                         label=f"Solar Orbiter B{labels_B[i]}")
            if l1_shifted and "B_gse_shifted" in l1_shifted:
                B_l1 = l1_shifted["B_gse_shifted"]
                T_l1_full = ensure_datetime_array(l1_shifted["time"])
                mask_l1   = (T_l1_full >= t0) & (T_l1_full <= t1)
                if np.any(mask_l1) and B_l1.shape[1] > i:
                    T_l1 = T_l1_full[mask_l1]
                    B_l1_i = B_l1[mask_l1, i]
                    axes[i].plot(T_l1, B_l1_i, linewidth=1.0,
                                 color=colors_B[i], linestyle='--', alpha=0.7,
                                 label=f"L1 B{labels_B[i]} (shifted)")
            axes[i].set_ylabel(f"B{labels_B[i]} [nT]")
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc="upper right", fontsize=8)
            axes[i].axhline(0, color="black", linestyle=":", alpha=0.5, linewidth=0.8)

        # ------------------------------------------------------------------
        # EFLUX PANEL – HEATMAP (PAS-EFLUX), OLD STYLE
        # ------------------------------------------------------------------
        if swa and "eflux" in swa and swa.get("eflux") is not None:
            from matplotlib.dates import date2num
            import matplotlib.colors as mcolors

            T_ef_full = ensure_datetime_array(swa["time"])
            mask_ef   = (T_ef_full >= t0) & (T_ef_full <= t1)

            if np.any(mask_ef):
                T_ef   = T_ef_full[mask_ef]
                eflux  = np.array(swa["eflux"])[mask_ef]
                energy = swa.get("energy")

                if eflux.ndim == 2 and energy is not None:
                    times_mpl = date2num(T_ef)
                    T_mesh, E_mesh = np.meshgrid(times_mpl, energy)

                    pcm = axes[3].pcolormesh(
                        T_mesh,
                        E_mesh,
                        eflux.T,
                        shading="auto",
                        cmap="jet",
                        norm=mcolors.LogNorm(vmin=1e5, vmax=1e10),
                    )
                    axes[3].set_yscale("log")
                    axes[3].set_ylabel("Energy [eV]")
                    axes[3].set_ylim([100, 2e4])

                    position = axes[3].get_position()
                    cax = fig.add_axes([position.x1 + 0.01,
                                        position.y0,
                                        0.015,
                                        position.height])
                    cbar = fig.colorbar(pcm, cax=cax)
                    cbar.set_label(
                        r"eflux [cm$^{-2}$ s$^{-1}$ sr$^{-1}$ eV$^{-1}$]",
                        rotation=270, labelpad=20
                    )
                else:
                    axes[3].text(0.5, 0.5, "No 2D eflux data",
                                 ha="center", va="center",
                                 transform=axes[3].transAxes)
            else:
                axes[3].text(0.5, 0.5, "No Eflux data in window",
                             ha="center", va="center",
                             transform=axes[3].transAxes)
        else:
            axes[3].text(0.5, 0.5, "No Eflux data",
                         ha="center", va="center",
                         transform=axes[3].transAxes)
        axes[3].grid(True, alpha=0.3)

        # Temperature
        if swa_moments and "T" in swa_moments:
            T_swa_full = ensure_datetime_array(swa_moments["time"])
            mask_swa   = (T_swa_full >= t0) & (T_swa_full <= t1)
            if np.any(mask_swa):
                T_swa = T_swa_full[mask_swa]
                T_data = np.array(swa_moments["T"])[mask_swa]
                axes[4].plot(T_swa, T_data, linewidth=1.2, color="purple",
                             label="Solar Orbiter T")
                if l1_shifted and "T_shifted" in l1_shifted:
                    T_l1_full = ensure_datetime_array(l1_shifted["time"])
                    mask_l1   = (T_l1_full >= t0) & (T_l1_full <= t1)
                    if np.any(mask_l1):
                        T_l1 = T_l1_full[mask_l1]
                        T_l1_data = np.array(l1_shifted["T_shifted"])[mask_l1]
                        axes[4].plot(T_l1, T_l1_data, linewidth=1.0,
                                     color="purple", linestyle='--', alpha=0.7,
                                     label="L1 T (shifted)")
                axes[4].set_ylabel("T [eV]")
                axes[4].legend(loc="upper right", fontsize=8)
            else:
                axes[4].text(0.5, 0.5, "No temperature data in window",
                             ha="center", va="center", transform=axes[4].transAxes)
        else:
            axes[4].text(0.5, 0.5, "No temperature data",
                         ha="center", va="center", transform=axes[4].transAxes)
        axes[4].grid(True, alpha=0.3)

        # Density
        if swa_moments and "n" in swa_moments:
            T_swa_full = ensure_datetime_array(swa_moments["time"])
            mask_swa   = (T_swa_full >= t0) & (T_swa_full <= t1)
            if np.any(mask_swa):
                T_swa = T_swa_full[mask_swa]
                n_data = np.array(swa_moments["n"])[mask_swa]
                axes[5].plot(T_swa, n_data, linewidth=1.2, color="orange",
                             label="Solar Orbiter n")
                if l1_shifted and "n_shifted" in l1_shifted:
                    T_l1_full = ensure_datetime_array(l1_shifted["time"])
                    mask_l1   = (T_l1_full >= t0) & (T_l1_full <= t1)
                    if np.any(mask_l1):
                        T_l1 = T_l1_full[mask_l1]
                        n_l1_data = np.array(l1_shifted["n_shifted"])[mask_l1]
                        axes[5].plot(T_l1, n_l1_data, linewidth=1.0,
                                     color="orange", linestyle='--', alpha=0.7,
                                     label="L1 n (shifted)")
                axes[5].set_ylabel("n [cm⁻³]")
                axes[5].legend(loc="upper right", fontsize=8)
            else:
                axes[5].text(0.5, 0.5, "No density data in window",
                             ha="center", va="center", transform=axes[5].transAxes)
        else:
            axes[5].text(0.5, 0.5, "No density data",
                         ha="center", va="center", transform=axes[5].transAxes)
        axes[5].grid(True, alpha=0.3)

        # Velocity magnitude
        if swa_moments and "V" in swa_moments:
            T_swa_full = ensure_datetime_array(swa_moments["time"])
            mask_swa   = (T_swa_full >= t0) & (T_swa_full <= t1)
            if np.any(mask_swa):
                T_swa = T_swa_full[mask_swa]
                V_data = np.array(swa_moments["V"])[mask_swa]
                V_mag  = np.linalg.norm(V_data, axis=1)
                axes[6].plot(T_swa, V_mag, linewidth=1.2, color="cyan",
                             label="Solar Orbiter V")
                if l1_shifted and "V_shifted" in l1_shifted:
                    T_l1_full = ensure_datetime_array(l1_shifted["time"])
                    mask_l1   = (T_l1_full >= t0) & (T_l1_full <= t1)
                    if np.any(mask_l1):
                        T_l1 = T_l1_full[mask_l1]
                        V_l1 = np.array(l1_shifted["V_shifted"])[mask_l1]
                        V_l1_mag = np.linalg.norm(V_l1, axis=1)
                        axes[6].plot(T_l1, V_l1_mag, linewidth=1.0,
                                     color="cyan", linestyle='--', alpha=0.7,
                                     label="L1 V (shifted)")
                axes[6].set_ylabel("V [km/s]")
                axes[6].legend(loc="upper right", fontsize=8)
            else:
                axes[6].text(0.5, 0.5, "No velocity data in window",
                             ha="center", va="center", transform=axes[6].transAxes)
        else:
            axes[6].text(0.5, 0.5, "No velocity data",
                         ha="center", va="center", transform=axes[6].transAxes)
        axes[6].grid(True, alpha=0.3)

        # Dst panel
        has_dst_data = False
        if dst_shifted:
            axes[7].plot(dst_shifted["time"], dst_shifted["dst_shifted"],
                         linewidth=1.5, color="blue",
                         label="Measured Dst (Time-Shifted)", alpha=0.8)
            has_dst_data = True
        if dst_pred:
            axes[7].plot(dst_pred["time"], dst_pred["dst_predicted"],
                         linewidth=1.5, color="red", linestyle="--",
                         label="Predicted Dst (Crossover)", alpha=0.8)
            has_dst_data = True

        if has_dst_data:
            axes[7].set_ylabel("Dst [nT]")
            axes[7].axhline(0, color="black", linestyle=":", alpha=0.5, linewidth=0.8)
            axes[7].axhline(-50, color="orange", linestyle="--", alpha=0.3, linewidth=0.8,
                            label="Moderate storm")
            axes[7].axhline(-100, color="red", linestyle="--", alpha=0.3, linewidth=0.8,
                            label="Strong storm")
            axes[7].legend(loc="upper right", fontsize=8)
        else:
            axes[7].text(0.5, 0.5, "No Dst data",
                         ha="center", va="center", transform=axes[7].transAxes)
        axes[7].grid(True, alpha=0.3)

        # Kp panel
        has_kp_data = False
        if kp_shifted:
            axes[8].plot(kp_shifted["time"], kp_shifted["kp_shifted"],
                         linewidth=1.5, color="blue", marker='o', markersize=4,
                         label="Measured Kp (Time-Shifted)", alpha=0.8)
            has_kp_data = True
        if kp_pred:
            axes[8].plot(kp_pred["time"], kp_pred["kp_predicted"],
                         linewidth=1.5, color="red", linestyle="--",
                         marker='s', markersize=4,
                         label="Predicted Kp (Crossover)", alpha=0.8)
            has_kp_data = True

        if has_kp_data:
            axes[8].set_ylabel("Kp")
            axes[8].set_ylim([0, 9])
            axes[8].axhline(5, color="orange", linestyle="--", alpha=0.3, linewidth=0.8,
                            label="Storm threshold")
            axes[8].axhline(7, color="red", linestyle="--", alpha=0.3, linewidth=0.8,
                            label="Strong storm")
            axes[8].legend(loc="upper right", fontsize=8)
        else:
            axes[8].text(0.5, 0.5, "No Kp data",
                         ha="center", va="center", transform=axes[8].transAxes)
        axes[8].grid(True, alpha=0.3)

        from matplotlib.dates import DateFormatter, AutoDateLocator
        axes[-1].set_xlabel("Time (UTC)")
        locator = AutoDateLocator()
        formatter = DateFormatter("%Y-%b-%d\n%H:%M")
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(formatter)

        axes[-1].set_xlim(T_mag[0], T_mag[-1])
        for i in range(8):
            axes[i].sharex(axes[8])
            axes[i].tick_params(labelbottom=False)

        plt.suptitle(
            f"Crossover Event with L1 Comparison — {center.strftime('%Y-%m-%d %H:%M:%S')}",
            fontsize=14, y=0.995
        )
        fig.autofmt_xdate()
        return fig
    
# ------------------------------- Main driver ------------------------------ #

def main():
    print("Hénon DRO Constellation Crossover Analysis with L1 Comparison")
    print(f"Period: {START_DATE:%Y-%m-%d} → {END_DATE:%Y-%m-%d}")
    print(f"DRO e = {ECCENTRICITY}, tolerances: SE={SUN_EARTH_TOL_AU} AU, "
          f"DRO={DRO_TOL_AU} AU, dt={DT_HOURS} h")

    print("\nBuilding DRO constellation...")
    constellation = DROConstellation(eccentricity=ECCENTRICITY)
    print("Constellation ready (3 sats).")

    print("\nLoading spacecraft ephemerides...")
    spacecraft_list = ["STEREO-A", "Solar Orbiter"]
    spacecraft_data = {}
    for sc in spacecraft_list:
        print(f"[DBG] main: loading positions for {sc}")
        data = SpacecraftData.get_positions(sc, START_DATE, END_DATE,
                                            dt_hours=DT_HOURS)
        if data:
            spacecraft_data[sc] = data
            print(f"{sc}: {len(data['times'])} samples")
        else:
            print(f"{sc}: no data")

    if not spacecraft_data:
        print("No spacecraft data available.")
        return

    print("\nFinding crossover events...")
    finder = CrossoverFinder(constellation, SUN_EARTH_TOL_AU, DRO_TOL_AU)
    all_events = {}
    for sc, sc_data in spacecraft_data.items():
        print(f"[DBG] main: finding events for {sc}")
        ev   = finder.find_events(sc_data, START_DATE)
        uniq = finder.group_events_by_day(ev)
        limit = EVENT_LIMIT.get(sc)
        if isinstance(limit, int) and limit >= 0:
            uniq = uniq[:limit]
        all_events[sc] = uniq
        print(f"{sc}: {len(uniq)} events")

    total_events = sum(len(v) for v in all_events.values())
    if total_events == 0:
        print("\nNo crossover events found with current tolerances.")
        return

    print("\nPreparing outputs...")
    records = []
    for sc, evs in all_events.items():
        for e in evs:
            records.append({
                "Spacecraft":           sc,
                "Date":                 e["time"].strftime("%Y-%m-%d"),
                "Time_UTC":             e["time"].strftime("%H:%M:%S"),
                "DRO_ID":               e["dro_id"],
                "Perp_to_SE_Line_AU":   round(e["perp_to_se_line_au"], 5),
                "Along_SE_Line_AU":     round(e["along_se_line_au"], 5),
                "Distance_to_DRO_AU":   round(e["dist_to_dro_au"], 5),
                "Between_Sun_Earth":    e["is_between_sun_earth"],
                "SC_X_AU":              round(e["spacecraft_pos"][0] / AU_KM, 5),
                "SC_Y_AU":              round(e["spacecraft_pos"][1] / AU_KM, 5),
                "SC_Z_AU":              round(e["spacecraft_pos"][2] / AU_KM, 5),
            })

    if records and WRITE_EVENTS_CSV:
        df = pd.DataFrame(records)
        df.sort_values(["Date", "Spacecraft"], inplace=True)
        df.to_csv(CSV_FILENAME, index=False)
        print(f"Events CSV: {CSV_FILENAME}  (rows: {len(df)})")

    if PLOT_OVERVIEW:
        print("\nPlotting: overview...")
        Plotter.plot_constellation_overview(all_events, constellation)

    if PLOT_HELIOCENTRIC:
        print("Plotting: heliocentric tracks (±21 d)...")
        for sc, evs in all_events.items():
            for e in evs:
                Plotter.plot_event_detail(e, sc, constellation,
                                          spacecraft_data[sc], START_DATE,
                                          context_days=TRACK_CONTEXT_DAYS)

    if PLOT_EARTH_FRAME:
        print("Plotting: Earth-centered tracks (±21 d)...")
        for sc, evs in all_events.items():
            for e in evs:
                Plotter.plot_event_detail_earth_frame(
                    e, sc, constellation, spacecraft_data[sc],
                    START_DATE, context_days=TRACK_CONTEXT_DAYS
                )

    if PLOT_INSITU_DST and all_events.get("Solar Orbiter"):
        print("\nLoading in-situ data...")
        solo_preloaded = SoloDataLoader.preload_all(
            all_events["Solar Orbiter"],
            spacecraft_data["Solar Orbiter"],
            START_DATE,
            window_hours=INSITU_WINDOW_HR
        )

        print("\nGenerating plots...")
        for e in all_events["Solar Orbiter"]:
            fig = Plotter.plot_solo_insitu_with_dst(e, solo_preloaded)
            if fig is None:
                print(f"⚠️  Could not create plot for {e['time']}")

    print("\nDone.")
    plt.show()


if __name__ == "__main__":
    main()
