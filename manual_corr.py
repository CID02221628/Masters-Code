"""
manual_corr.py

Manual window alignment with TWO correlation objectives:
  1) corr(|B|)  (binned, within user window)
  2) corr(d|B|/dt) (binned, within user window)

Window can be:
  - prompted from user, OR
  - defined in config (toggle)

Then applies BOTH lags to the full ballistic L1 series (native cadence) and
plots a final 7-panel stack plot overlaying:
  - Spacecraft (solid)
  - L1 ballistic (optional)
  - L1 shifted by |B| lag (optional)
  - L1 shifted by dB/dt lag (optional)

Binning affects ONLY correlation computation, not final plot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.dates import DateFormatter, AutoDateLocator


# =============================================================================
# CONFIG
# =============================================================================

@dataclass(frozen=True)
class Config:
    CSV_FILE: str = '/Users/henryhodges/Documents/Year 4/Masters/Code/figures/big file/csv/event_STEREOA_20240509.csv'

    # Max additional lag searched (seconds) around current ballistic alignment
    MAX_LAG_SECONDS: int = 20 * 3600  # +/- 20 hours

    # Correlation computed on z-scored signals within the selected window
    ZSCORE: bool = True

    # Minimum overlap points (after binning) required to accept a lag
    MIN_OVERLAP_POINTS: int = 10

    # ONLY for correlation: bin/average the windowed data.
    CORR_BIN_MINUTES: int = 30

    # Final plot overlays (toggles)
    PLOT_L1_BALLISTIC: bool = False
    PLOT_L1_SHIFT_BMAG: bool = True
    PLOT_L1_SHIFT_DBDT: bool = False

    # ---- Window selection toggle ----
    # If True, uses the two UTC strings below. If False, prompts user.
    USE_CONFIG_WINDOW: bool = True
    WINDOW_START_UTC: str = "2024-05-09 00:00"
    WINDOW_END_UTC: str = "2024-05-13 00:00"
    

# =============================================================================
# BASIC HELPERS
# =============================================================================

def _as_float_array(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _parse_time_seconds(df: pd.DataFrame) -> np.ndarray:
    if "unix_timestamp" in df.columns:
        return pd.to_numeric(df["unix_timestamp"], errors="coerce").to_numpy(dtype=float)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        return (ts.view("int64") / 1e9).to_numpy(dtype=float)
    raise ValueError("CSV must contain 'unix_timestamp' or 'timestamp'.")


def _median_dt_seconds(t: np.ndarray) -> float:
    t = t[np.isfinite(t)]
    if t.size < 2:
        return np.nan
    dt = np.diff(np.sort(t))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.median(dt)) if dt.size else np.nan


def _interp_at_times(t_src: np.ndarray, y_src: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    mask = np.isfinite(t_src) & np.isfinite(y_src)
    if np.sum(mask) < 2:
        return np.full_like(t_new, np.nan, dtype=float)

    tt = t_src[mask]
    yy = y_src[mask]
    order = np.argsort(tt)
    tt = tt[order]
    yy = yy[order]

    out = np.interp(t_new, tt, yy, left=np.nan, right=np.nan)
    out[(t_new < tt[0]) | (t_new > tt[-1])] = np.nan
    return out


def _zscore(x: np.ndarray) -> np.ndarray:
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s == 0:
        return x * np.nan
    return (x - m) / s


def _corrcoef_nan(x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(np.sum(mask))
    if n < 2:
        return np.nan, n
    xx = x[mask]
    yy = y[mask]
    if np.std(xx) == 0 or np.std(yy) == 0:
        return np.nan, n
    return float(np.corrcoef(xx, yy)[0, 1]), n


def _parse_utc_to_unix_seconds(s: str) -> float:
    s = s.strip().replace("T", " ")
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Could not parse time: {s}")
    return float(ts.value / 1e9)


# =============================================================================
# BINNING ONLY FOR CORRELATION
# =============================================================================

def _bin_means_by_time(
    t: np.ndarray,
    x: np.ndarray,
    t_start: float,
    t_end: float,
    bin_seconds: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if bin_seconds <= 0:
        raise ValueError("bin_seconds must be > 0")

    wmask = (t >= t_start) & (t <= t_end) & np.isfinite(t)
    if np.sum(wmask) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    tt = t[wmask]
    xx = x[wmask]

    bin_id = np.floor((tt - t_start) / bin_seconds).astype(int)
    nbins = int(np.max(bin_id)) + 1
    if nbins <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    valid = np.isfinite(xx)
    sums = np.bincount(bin_id[valid], weights=xx[valid], minlength=nbins).astype(float)
    counts = np.bincount(bin_id[valid], minlength=nbins).astype(float)

    means = np.full(nbins, np.nan, dtype=float)
    nz = counts > 0
    means[nz] = sums[nz] / counts[nz]

    centers = t_start + (np.arange(nbins, dtype=float) + 0.5) * bin_seconds
    cmask = (centers >= t_start) & (centers <= t_end)
    return centers[cmask], means[cmask]


def _derivative(x: np.ndarray, dt: float) -> np.ndarray:
    if x.size < 2 or not np.isfinite(dt) or dt <= 0:
        return np.full_like(x, np.nan, dtype=float)
    return np.gradient(x, dt)


# =============================================================================
# PLOTTING (7-panel)
# =============================================================================
def _get_spacecraft_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    b_r = _as_float_array(df.get("B_r", pd.Series(np.nan, index=df.index)))
    b_t = _as_float_array(df.get("B_t", pd.Series(np.nan, index=df.index)))
    b_n = _as_float_array(df.get("B_n", pd.Series(np.nan, index=df.index)))

    b_mag = _as_float_array(df.get("B_mag", pd.Series(np.nan, index=df.index)))
    if (not np.isfinite(b_mag).any()) and (np.isfinite(b_r).any() or np.isfinite(b_t).any() or np.isfinite(b_n).any()):
        b_mag = np.linalg.norm(np.vstack([b_r, b_t, b_n]).T, axis=1)

    swa_n = _as_float_array(df.get("swa_n", pd.Series(np.nan, index=df.index)))

    # ---- V magnitude: prefer swa_V_mag, else compute from components ----
    swa_vmag = _as_float_array(df.get("swa_V_mag", pd.Series(np.nan, index=df.index)))
    if not np.isfinite(swa_vmag).any():
        if all(c in df.columns for c in ["swa_V_r", "swa_V_t", "swa_V_n"]):
            v_r = _as_float_array(df["swa_V_r"])
            v_t = _as_float_array(df["swa_V_t"])
            v_n_ = _as_float_array(df["swa_V_n"])
            swa_vmag = np.linalg.norm(np.vstack([v_r, v_t, v_n_]).T, axis=1)

    swa_T = _as_float_array(df.get("swa_T", pd.Series(np.nan, index=df.index)))

    return {
        "B_r": b_r, "B_t": b_t, "B_n": b_n, "B_mag": b_mag,
        "n": swa_n, "V_mag": swa_vmag, "T": swa_T
    }

def _get_l1_arrays(df: pd.DataFrame, mode: str) -> Dict[str, np.ndarray]:
    if mode == "ballistic":
        bx, by, bz = "l1_B_x_gse_ballistic", "l1_B_y_gse_ballistic", "l1_B_z_gse_ballistic"
        bm = "l1_B_mag_ballistic"
        nn = "l1_n_ballistic"
        vx, vy, vz = "l1_V_x_ballistic", "l1_V_y_ballistic", "l1_V_z_ballistic"
        vm = "l1_V_mag_ballistic"
        tt = "l1_T_ballistic"

    elif mode == "bmag":
        bx, by, bz = "l1_B_x_gse_shifted_bmag", "l1_B_y_gse_shifted_bmag", "l1_B_z_gse_shifted_bmag"
        bm = "l1_B_mag_shifted_bmag"
        nn = "l1_n_shifted_bmag"
        vx, vy, vz = "l1_V_x_shifted_bmag", "l1_V_y_shifted_bmag", "l1_V_z_shifted_bmag"
        vm = "l1_V_mag_shifted_bmag"
        tt = "l1_T_shifted_bmag"

    elif mode == "dbdt":
        bx, by, bz = "l1_B_x_gse_shifted_dbdt", "l1_B_y_gse_shifted_dbdt", "l1_B_z_gse_shifted_dbdt"
        bm = "l1_B_mag_shifted_dbdt"
        nn = "l1_n_shifted_dbdt"
        vx, vy, vz = "l1_V_x_shifted_dbdt", "l1_V_y_shifted_dbdt", "l1_V_z_shifted_dbdt"
        vm = "l1_V_mag_shifted_dbdt"
        tt = "l1_T_shifted_dbdt"

    else:
        raise ValueError(f"Unknown L1 mode: {mode}")

    Bx = _as_float_array(df.get(bx, pd.Series(np.nan, index=df.index)))
    By = _as_float_array(df.get(by, pd.Series(np.nan, index=df.index)))
    Bz = _as_float_array(df.get(bz, pd.Series(np.nan, index=df.index)))
    Bm = _as_float_array(df.get(bm, pd.Series(np.nan, index=df.index)))

    n = _as_float_array(df.get(nn, pd.Series(np.nan, index=df.index)))

    Vx = _as_float_array(df.get(vx, pd.Series(np.nan, index=df.index)))
    Vy = _as_float_array(df.get(vy, pd.Series(np.nan, index=df.index)))
    Vz = _as_float_array(df.get(vz, pd.Series(np.nan, index=df.index)))
    Vm = _as_float_array(df.get(vm, pd.Series(np.nan, index=df.index)))

    # ---- V magnitude: prefer *_V_mag_*, else compute from components ----
    if not np.isfinite(Vm).any():
        if np.isfinite(Vx).any() or np.isfinite(Vy).any() or np.isfinite(Vz).any():
            Vm = np.linalg.norm(np.vstack([Vx, Vy, Vz]).T, axis=1)

    T = _as_float_array(df.get(tt, pd.Series(np.nan, index=df.index)))

    return {
        "B_x": Bx, "B_y": By, "B_z": Bz, "B_mag": Bm,
        "n": n, "V_mag": Vm, "T": T
    }

def make_stackplot_multi(
    df: pd.DataFrame,
    title_extra: str,
    show_ballistic: bool,
    show_bmag: bool,
    show_dbdt: bool,
) -> plt.Figure:
    t_sec = _parse_time_seconds(df)
    t_dt = pd.to_datetime(t_sec, unit="s", utc=True).to_pydatetime()

    spacecraft_name = str(df["spacecraft"].iloc[0]) if "spacecraft" in df.columns else "Spacecraft"
    crossover_date = str(df["crossover_date"].iloc[0]) if "crossover_date" in df.columns else ""
    crossover_time = str(df["crossover_time"].iloc[0]) if "crossover_time" in df.columns else ""
    center_str = f"{crossover_date} {crossover_time}".strip()

    sc = _get_spacecraft_arrays(df)

    overlays = []
    if show_ballistic:
        overlays.append(("L1 ballistic", "ballistic", ":", "0.6"))
    if show_bmag:
        overlays.append(("L1 shifted (|B|)", "bmag", ":", "k"))
    if show_dbdt:
        overlays.append(("L1 shifted (d|B|/dt)", "dbdt", "--", "k"))

    fig = plt.figure(figsize=(14, 14))
    gs = GridSpec(7, 1, figure=fig, hspace=0.08)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(7)]

    # |B|
    axes[0].plot(t_dt, sc["B_mag"], linewidth=1.2, color="black", label=f"{spacecraft_name} |B|")
    for lab, mode, ls, col in overlays:
        l1 = _get_l1_arrays(df, mode)
        if np.isfinite(l1["B_mag"]).any():
            axes[0].plot(t_dt, l1["B_mag"], linewidth=1.0, linestyle=ls, color=col, alpha=0.9, label=f"{lab} |B|")
    axes[0].set_ylabel("|B| [nT]", fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_ylim(bottom=0)

    # components
    labels = ["r", "t", "n"]
    colors = ["red", "green", "blue"]
    sc_list = [sc["B_r"], sc["B_t"], sc["B_n"]]

    for i in range(3):
        ax = axes[i + 1]
        ax.plot(t_dt, sc_list[i], linewidth=1.2, color=colors[i], label=f"{spacecraft_name} B{labels[i]}")
        for lab, mode, ls, col in overlays:
            l1 = _get_l1_arrays(df, mode)
            comp = {"r": "B_x", "t": "B_y", "n": "B_z"}[labels[i]]
            if np.isfinite(l1[comp]).any():
                ax.plot(t_dt, l1[comp], linewidth=1.0, linestyle=ls, color=col, alpha=0.9, label=f"{lab} B{labels[i]}")
        ax.set_ylabel(f"B{labels[i]} [nT]", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        ax.axhline(0, color="black", linestyle=":", alpha=0.5, linewidth=0.8)

    # n
    ax = axes[4]
    if np.isfinite(sc["n"]).any():
        ax.plot(t_dt, sc["n"], linewidth=1.2, color="orange", label=f"{spacecraft_name} n")
        for lab, mode, ls, col in overlays:
            l1 = _get_l1_arrays(df, mode)
            if np.isfinite(l1["n"]).any():
                ax.plot(t_dt, l1["n"], linewidth=1.0, linestyle=ls, color=col, alpha=0.9, label=f"{lab} n")
        ax.set_ylabel("n [cm⁻³]", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(bottom=0)
    else:
        ax.text(0.5, 0.5, "No n", ha="center", va="center", transform=ax.transAxes, fontsize=10)
    ax.grid(True, alpha=0.3)

    # V
    ax = axes[5]
    if np.isfinite(sc["V_mag"]).any():
        ax.plot(t_dt, sc["V_mag"], linewidth=1.2, color="cyan", label=f"{spacecraft_name} V")
        for lab, mode, ls, col in overlays:
            l1 = _get_l1_arrays(df, mode)
            if np.isfinite(l1["V_mag"]).any():
                ax.plot(t_dt, l1["V_mag"], linewidth=1.0, linestyle=ls, color=col, alpha=0.9, label=f"{lab} V")
        ax.set_ylabel("V [km/s]", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(bottom=0)
    else:
        ax.text(0.5, 0.5, "No V", ha="center", va="center", transform=ax.transAxes, fontsize=10)
    ax.grid(True, alpha=0.3)

    # T
    ax = axes[6]
    if np.isfinite(sc["T"]).any():
        ax.plot(t_dt, sc["T"], linewidth=1.2, color="purple", label=f"{spacecraft_name} T")
        for lab, mode, ls, col in overlays:
            l1 = _get_l1_arrays(df, mode)
            if np.isfinite(l1["T"]).any():
                ax.plot(t_dt, l1["T"], linewidth=1.0, linestyle=ls, color=col, alpha=0.9, label=f"{lab} T")
        ax.set_ylabel("T [eV]", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(bottom=0)
    else:
        ax.text(0.5, 0.5, "No T", ha="center", va="center", transform=ax.transAxes, fontsize=10)
    ax.grid(True, alpha=0.3)

    locator = AutoDateLocator()
    formatter = DateFormatter("%Y-%b-%d\n%H:%M")
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)
    axes[-1].set_xlim(t_dt[0], t_dt[-1])
    axes[-1].set_xlabel("Time (UTC)", fontsize=11)

    for i in range(6):
        axes[i].sharex(axes[6])
        axes[i].tick_params(labelbottom=False)

    title = f"{spacecraft_name} In-Situ – {center_str}".strip()
    if title_extra:
        title += f"\n{title_extra}"
    plt.suptitle(title, fontsize=14, y=0.998)

    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


# =============================================================================
# TWO-OBJECTIVE CORRELATION (BINNED)
# =============================================================================

def _best_lag_for_signal(
    t: np.ndarray,
    sc_signal: np.ndarray,
    l1_ballistic_signal: np.ndarray,
    t_start: float,
    t_end: float,
    cfg: Config,
    use_derivative: bool,
) -> Tuple[float, float, int]:
    dt_native = _median_dt_seconds(t)
    if not np.isfinite(dt_native) or dt_native <= 0:
        raise ValueError("Bad/unknown cadence (dt).")

    bin_seconds = int(cfg.CORR_BIN_MINUTES * 60)
    if bin_seconds <= 0:
        raise ValueError("CORR_BIN_MINUTES must be >= 1")

    wmask = (t >= t_start) & (t <= t_end) & np.isfinite(t)
    if np.sum(wmask) < 2:
        raise ValueError("Selected window contains too few samples.")
    t_w = t[wmask]

    _, sc_b = _bin_means_by_time(t, sc_signal, t_start, t_end, bin_seconds)
    if use_derivative:
        sc_b = _derivative(sc_b, float(bin_seconds))
    if cfg.ZSCORE:
        sc_b = _zscore(sc_b)

    max_steps = int(round(cfg.MAX_LAG_SECONDS / dt_native))
    max_steps = max(1, max_steps)

    best_lag = 0.0
    best_corr = -np.inf
    best_n = 0

    for steps in range(-max_steps, max_steps + 1):
        lag = steps * dt_native

        l1_shifted_w = _interp_at_times(t, l1_ballistic_signal, t_w - lag)

        tmp = np.full_like(t, np.nan, dtype=float)
        tmp[wmask] = l1_shifted_w

        _, l1_b = _bin_means_by_time(t, tmp, t_start, t_end, bin_seconds)
        if use_derivative:
            l1_b = _derivative(l1_b, float(bin_seconds))
        if cfg.ZSCORE:
            l1_b = _zscore(l1_b)

        c, n = _corrcoef_nan(sc_b, l1_b)
        if n < cfg.MIN_OVERLAP_POINTS or not np.isfinite(c):
            continue

        if c > best_corr:
            best_corr = c
            best_lag = float(lag)
            best_n = int(n)

    if best_corr == -np.inf:
        raise ValueError("No valid lag found (NaNs too high or MIN_OVERLAP too strict).")

    return best_lag, float(best_corr), int(best_n)


def compute_two_lags(
    df: pd.DataFrame,
    t_start: float,
    t_end: float,
    cfg: Config,
) -> Dict[str, Dict[str, float]]:
    t = _parse_time_seconds(df)
    sc_bmag = _as_float_array(df["B_mag"])
    l1_bmag_ball = _as_float_array(df["l1_B_mag_ballistic"])

    lag_bmag, corr_bmag, n_bmag = _best_lag_for_signal(t, sc_bmag, l1_bmag_ball, t_start, t_end, cfg, use_derivative=False)
    lag_dbdt, corr_dbdt, n_dbdt = _best_lag_for_signal(t, sc_bmag, l1_bmag_ball, t_start, t_end, cfg, use_derivative=True)

    return {
        "bmag": {"lag_seconds": lag_bmag, "corr": corr_bmag, "n": float(n_bmag)},
        "dbdt": {"lag_seconds": lag_dbdt, "corr": corr_dbdt, "n": float(n_dbdt)},
    }


def apply_lag_to_all_l1_columns(df: pd.DataFrame, lag_seconds: float, tag: str) -> pd.DataFrame:
    out = df.copy()
    t = _parse_time_seconds(out)

    def _mk(src: str, dst: str) -> None:
        if src not in out.columns:
            out[dst] = np.full(len(out), np.nan, dtype=float)
            return
        y = _as_float_array(out[src])
        out[dst] = _interp_at_times(t, y, t - lag_seconds)

    # B
    _mk("l1_B_mag_ballistic", f"l1_B_mag_shifted_{tag}")
    _mk("l1_B_x_gse_ballistic", f"l1_B_x_gse_shifted_{tag}")
    _mk("l1_B_y_gse_ballistic", f"l1_B_y_gse_shifted_{tag}")
    _mk("l1_B_z_gse_ballistic", f"l1_B_z_gse_shifted_{tag}")

    # n, T
    _mk("l1_n_ballistic", f"l1_n_shifted_{tag}")
    _mk("l1_T_ballistic", f"l1_T_shifted_{tag}")

    # V: shift components AND mag (if present), then compute mag if needed
    _mk("l1_V_x_ballistic", f"l1_V_x_shifted_{tag}")
    _mk("l1_V_y_ballistic", f"l1_V_y_shifted_{tag}")
    _mk("l1_V_z_ballistic", f"l1_V_z_shifted_{tag}")
    _mk("l1_V_mag_ballistic", f"l1_V_mag_shifted_{tag}")

    # compute V_mag if missing/NaN
    vmag_col = f"l1_V_mag_shifted_{tag}"
    if vmag_col not in out.columns or not np.isfinite(_as_float_array(out[vmag_col])).any():
        vx = _as_float_array(out.get(f"l1_V_x_shifted_{tag}", pd.Series(np.nan, index=out.index)))
        vy = _as_float_array(out.get(f"l1_V_y_shifted_{tag}", pd.Series(np.nan, index=out.index)))
        vz = _as_float_array(out.get(f"l1_V_z_shifted_{tag}", pd.Series(np.nan, index=out.index)))
        if np.isfinite(vx).any() or np.isfinite(vy).any() or np.isfinite(vz).any():
            out[vmag_col] = np.linalg.norm(np.vstack([vx, vy, vz]).T, axis=1)

    return out

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    cfg = Config()
    csv_path = Path(cfg.CSV_FILE).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV_FILE not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "B_mag" not in df.columns:
        raise ValueError("Missing spacecraft column: B_mag")
    if "l1_B_mag_ballistic" not in df.columns:
        raise ValueError("Missing L1 ballistic column: l1_B_mag_ballistic")

    print("[DEBUG] SC V_mag finite:", np.isfinite(_as_float_array(df.get("swa_V_mag", pd.Series(np.nan, index=df.index)))).sum())
    print("[DEBUG] SC V components present:", all(c in df.columns for c in ["swa_V_r","swa_V_t","swa_V_n"]))
    print("[DEBUG] L1 V_mag ballistic finite:", np.isfinite(_as_float_array(df.get("l1_V_mag_ballistic", pd.Series(np.nan, index=df.index)))).sum())
    print("[DEBUG] L1 V components present:", all(c in df.columns for c in ["l1_V_x_ballistic","l1_V_y_ballistic","l1_V_z_ballistic"]))
    
    
    # Plot 1: SC + L1 ballistic (for context)
    fig1 = make_stackplot_multi(
        df,
        title_extra="(initial overlay)",
        show_ballistic=True,
        show_bmag=False,
        show_dbdt=False,
    )
    plt.show()
    plt.close(fig1)

    # Window selection (config OR prompt)
    if cfg.USE_CONFIG_WINDOW:
        t_start = _parse_utc_to_unix_seconds(cfg.WINDOW_START_UTC)
        t_end = _parse_utc_to_unix_seconds(cfg.WINDOW_END_UTC)
        print("\n[INFO] Using CONFIG window:")
        print(f"  start: {cfg.WINDOW_START_UTC} UTC")
        print(f"  end  : {cfg.WINDOW_END_UTC} UTC")
    else:
        print("\nEnter CME interval (UTC) to use for correlation.")
        print("Format: YYYY-MM-DD HH:MM   (or include seconds)")
        t_start = _parse_utc_to_unix_seconds(input("Start time (UTC): "))
        t_end = _parse_utc_to_unix_seconds(input("End time   (UTC): "))

    if t_end <= t_start:
        raise ValueError("End time must be after start time.")

    dt_native = _median_dt_seconds(_parse_time_seconds(df))
    print("\n[INFO]")
    print(f"  dt ~ {dt_native:.3f} s")
    print(f"  search: +/- {cfg.MAX_LAG_SECONDS} s")
    print(f"  corr bin: {cfg.CORR_BIN_MINUTES} min (correlation only)")
    print(f"  min overlap bins: {cfg.MIN_OVERLAP_POINTS}")
    print(f"  zscore={cfg.ZSCORE}")

    # Compute both lags
    res = compute_two_lags(df=df, t_start=t_start, t_end=t_end, cfg=cfg)

    # Apply both lags to full series
    df2 = apply_lag_to_all_l1_columns(df, res["bmag"]["lag_seconds"], tag="bmag")
    df2 = apply_lag_to_all_l1_columns(df2, res["dbdt"]["lag_seconds"], tag="dbdt")

    print("\n[RESULTS]")
    print(f"  window: {pd.to_datetime(t_start, unit='s', utc=True)} -> {pd.to_datetime(t_end, unit='s', utc=True)}")
    print(f"  |B|   : lag={res['bmag']['lag_seconds']:.1f} s  ({res['bmag']['lag_seconds']/3600:.3f} h)  corr={res['bmag']['corr']:.3f}  n_bins={int(res['bmag']['n'])}")
    print(f"  dB/dt : lag={res['dbdt']['lag_seconds']:.1f} s  ({res['dbdt']['lag_seconds']/3600:.3f} h)  corr={res['dbdt']['corr']:.3f}  n_bins={int(res['dbdt']['n'])}")

    title_extra = (
        f"Windowed lags (bin={cfg.CORR_BIN_MINUTES}min): "
        f"|B| lag={res['bmag']['lag_seconds']:.0f}s (corr={res['bmag']['corr']:.2f}), "
        f"dB/dt lag={res['dbdt']['lag_seconds']:.0f}s (corr={res['dbdt']['corr']:.2f})"
    )

    fig2 = make_stackplot_multi(
        df2,
        title_extra=title_extra,
        show_ballistic=cfg.PLOT_L1_BALLISTIC,
        show_bmag=cfg.PLOT_L1_SHIFT_BMAG,
        show_dbdt=cfg.PLOT_L1_SHIFT_DBDT,
    )
    plt.show()
    plt.close(fig2)


if __name__ == "__main__":
    main()