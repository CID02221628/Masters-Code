"""
save_corr.py

Batch-run the |B| (B_mag) correlation lag from manual_corr.py using a params CSV.

For each params row (in params order):
  - Load the event CSV (by File name)
  - ALWAYS add a page to the XCORR PDF, in the SAME ORDER as params:
      * If Corr_prop_needed? == TRUE:
          - compute lag using window in params
          - apply lag -> l1_*_shifted_bmag
          - write xcorr columns -> l1_*_xcorr
          - save CSV in place
          - plot spacecraft + shifted overlay
      * Else:
          - do NOT compute
          - plot spacecraft + ballistic overlay (so the PDF order still matches main.py’s ballistic PDF)

This produces ONE PDF (XCORR) that you can compare page-by-page with the ballistic PDF already produced by main.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.dates import DateFormatter, AutoDateLocator
from matplotlib.backends.backend_pdf import PdfPages


# =============================================================================
# FIXED CONFIG (your always-the-same settings)
# =============================================================================

@dataclass(frozen=True)
class FixedConfig:
    EVENTS_DIR: str = "/Users/henryhodges/Documents/Year 4/Masters/Code/figures/primary/csv"
    PARAMS_FILE: str = "/Users/henryhodges/Documents/Year 4/Masters/Code/corr_prop_results.csv"

    MAX_LAG_SECONDS: int = 20 * 3600
    ZSCORE: bool = False
    MIN_OVERLAP_POINTS: int = 10
    CORR_BIN_MINUTES: int = 30

    # Save behavior
    SAVE_IN_PLACE: bool = True

    # Only ONE PDF produced here (XCORR). Ballistic PDF is produced by main.py already.
    SAVE_XCORR_PDF: bool = True
    XCORR_PDF_PATH: str = "/Users/henryhodges/Documents/Year 4/Masters/Code/figures/primary/plots/xcorr_all_events.pdf"

    # Optional: show one random sanity plot INLINE (no saving)
    SHOW_ONE_SANITY_PLOT_INLINE: bool = False
    SANITY_RANDOM_SEED: Optional[int] = None  # None = truly random each run

    # Debug
    PRINT_LAG_SEARCH_PROGRESS: bool = True
    LAG_PROGRESS_EVERY_STEPS: int = 300


# =============================================================================
# BASIC HELPERS (manual_corr core)
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
    s = str(s).strip().replace("T", " ")
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Could not parse time: {s}")
    return float(ts.value / 1e9)


def _norm_bool(x: Any) -> Optional[bool]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    return None


# =============================================================================
# BINNING ONLY FOR CORRELATION (manual_corr core)
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


# =============================================================================
# ONE-OBJECTIVE LAG SEARCH (|B| only)
# =============================================================================

def _best_lag_for_bmag(
    t: np.ndarray,
    sc_bmag: np.ndarray,
    l1_bmag_ballistic: np.ndarray,
    t_start: float,
    t_end: float,
    cfg: FixedConfig,
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

    _, sc_b = _bin_means_by_time(t, sc_bmag, t_start, t_end, bin_seconds)
    if cfg.ZSCORE:
        sc_b = _zscore(sc_b)

    max_steps = int(round(cfg.MAX_LAG_SECONDS / dt_native))
    max_steps = max(1, max_steps)

    best_lag = 0.0
    best_corr = -np.inf
    best_n = 0

    if cfg.PRINT_LAG_SEARCH_PROGRESS:
        print(f"    [LagSearch] dt_native ~ {dt_native:.3f} s | bin={bin_seconds}s | zscore={cfg.ZSCORE}")
        print(f"    [LagSearch] searching steps [-{max_steps}, +{max_steps}] (~+/- {max_steps*dt_native/3600:.2f} h)")

    for steps in range(-max_steps, max_steps + 1):
        lag = steps * dt_native

        l1_shifted_w = _interp_at_times(t, l1_bmag_ballistic, t_w - lag)

        tmp = np.full_like(t, np.nan, dtype=float)
        tmp[wmask] = l1_shifted_w

        _, l1_b = _bin_means_by_time(t, tmp, t_start, t_end, bin_seconds)
        if cfg.ZSCORE:
            l1_b = _zscore(l1_b)

        c, n = _corrcoef_nan(sc_b, l1_b)
        if n < cfg.MIN_OVERLAP_POINTS or not np.isfinite(c):
            continue

        if c > best_corr:
            best_corr = c
            best_lag = float(lag)
            best_n = int(n)

        if cfg.PRINT_LAG_SEARCH_PROGRESS and cfg.LAG_PROGRESS_EVERY_STEPS > 0:
            if (steps + max_steps) % cfg.LAG_PROGRESS_EVERY_STEPS == 0:
                print(f"    [LagSearch] step={steps:+6d} lag={lag/3600:+7.3f} h  corr={c:+.4f}  n={n}")

    if best_corr == -np.inf:
        raise ValueError("No valid lag found (NaNs too high or MIN_OVERLAP too strict).")

    return best_lag, float(best_corr), int(best_n)


# =============================================================================
# APPLY LAG / WRITE OUTPUT COLUMNS
# =============================================================================

def apply_lag_to_all_l1_columns(df: pd.DataFrame, lag_seconds: float, tag: str) -> pd.DataFrame:
    out = df.copy()
    t = _parse_time_seconds(out)

    def _mk(src: str, dst: str) -> None:
        if src not in out.columns:
            out[dst] = np.full(len(out), np.nan, dtype=float)
            return
        y = _as_float_array(out[src])
        out[dst] = _interp_at_times(t, y, t - lag_seconds)

    _mk("l1_B_mag_ballistic", f"l1_B_mag_shifted_{tag}")
    _mk("l1_B_x_gse_ballistic", f"l1_B_x_gse_shifted_{tag}")
    _mk("l1_B_y_gse_ballistic", f"l1_B_y_gse_shifted_{tag}")
    _mk("l1_B_z_gse_ballistic", f"l1_B_z_gse_shifted_{tag}")

    _mk("l1_n_ballistic", f"l1_n_shifted_{tag}")
    _mk("l1_T_ballistic", f"l1_T_shifted_{tag}")

    _mk("l1_V_x_ballistic", f"l1_V_x_shifted_{tag}")
    _mk("l1_V_y_ballistic", f"l1_V_y_shifted_{tag}")
    _mk("l1_V_z_ballistic", f"l1_V_z_shifted_{tag}")
    _mk("l1_V_mag_ballistic", f"l1_V_mag_shifted_{tag}")

    vmag_col = f"l1_V_mag_shifted_{tag}"
    if vmag_col not in out.columns or not np.isfinite(_as_float_array(out[vmag_col])).any():
        vx = _as_float_array(out.get(f"l1_V_x_shifted_{tag}", pd.Series(np.nan, index=out.index)))
        vy = _as_float_array(out.get(f"l1_V_y_shifted_{tag}", pd.Series(np.nan, index=out.index)))
        vz = _as_float_array(out.get(f"l1_V_z_shifted_{tag}", pd.Series(np.nan, index=out.index)))
        if np.isfinite(vx).any() or np.isfinite(vy).any() or np.isfinite(vz).any():
            out[vmag_col] = np.linalg.norm(np.vstack([vx, vy, vz]).T, axis=1)

    return out


def write_xcorr_columns_from_lag(df: pd.DataFrame, lag_seconds: float) -> pd.DataFrame:
    out = df.copy()
    t = _parse_time_seconds(out)

    def _mk(src: str, dst: str) -> None:
        if src not in out.columns:
            out[dst] = np.full(len(out), np.nan, dtype=float)
            return
        y = _as_float_array(out[src])
        out[dst] = _interp_at_times(t, y, t - lag_seconds)

    _mk("l1_B_x_gse_ballistic", "l1_B_x_gse_xcorr")
    _mk("l1_B_y_gse_ballistic", "l1_B_y_gse_xcorr")
    _mk("l1_B_z_gse_ballistic", "l1_B_z_gse_xcorr")
    _mk("l1_B_mag_ballistic", "l1_B_mag_xcorr")

    _mk("l1_V_x_ballistic", "l1_V_x_xcorr")
    _mk("l1_V_y_ballistic", "l1_V_y_xcorr")
    _mk("l1_V_z_ballistic", "l1_V_z_xcorr")
    _mk("l1_V_mag_ballistic", "l1_V_mag_xcorr")

    _mk("l1_n_ballistic", "l1_n_xcorr")
    _mk("l1_T_ballistic", "l1_T_xcorr")

    if "l1_V_mag_xcorr" in out.columns and not np.isfinite(_as_float_array(out["l1_V_mag_xcorr"])).any():
        vx = _as_float_array(out.get("l1_V_x_xcorr", pd.Series(np.nan, index=out.index)))
        vy = _as_float_array(out.get("l1_V_y_xcorr", pd.Series(np.nan, index=out.index)))
        vz = _as_float_array(out.get("l1_V_z_xcorr", pd.Series(np.nan, index=out.index)))
        if np.isfinite(vx).any() or np.isfinite(vy).any() or np.isfinite(vz).any():
            out["l1_V_mag_xcorr"] = np.linalg.norm(np.vstack([vx, vy, vz]).T, axis=1)

    return out


def drop_old_shift_and_xcorr(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    drop: List[str] = []
    for c in cols:
        if c.startswith("xcorr_"):
            drop.append(c)
        if c.startswith("l1_") and c.endswith("_xcorr"):
            drop.append(c)
        if c.startswith("l1_") and ("_shifted_bmag" in c or "_shifted_dbdt" in c):
            drop.append(c)
    drop = sorted(set(drop))
    if drop:
        print(f"  [Drop] Removing {len(drop)} old xcorr/shifted columns")
        return df.drop(columns=drop, errors="ignore")
    print("  [Drop] No old xcorr/shifted columns found")
    return df


# =============================================================================
# PLOTTING (manual_corr 7-panel style)
# =============================================================================

def _get_spacecraft_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    b_r = _as_float_array(df.get("B_r", pd.Series(np.nan, index=df.index)))
    b_t = _as_float_array(df.get("B_t", pd.Series(np.nan, index=df.index)))
    b_n = _as_float_array(df.get("B_n", pd.Series(np.nan, index=df.index)))

    b_mag = _as_float_array(df.get("B_mag", pd.Series(np.nan, index=df.index)))
    if (not np.isfinite(b_mag).any()) and (np.isfinite(b_r).any() or np.isfinite(b_t).any() or np.isfinite(b_n).any()):
        b_mag = np.linalg.norm(np.vstack([b_r, b_t, b_n]).T, axis=1)

    swa_n = _as_float_array(df.get("swa_n", pd.Series(np.nan, index=df.index)))

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

    if not np.isfinite(Vm).any():
        if np.isfinite(Vx).any() or np.isfinite(Vy).any() or np.isfinite(Vz).any():
            Vm = np.linalg.norm(np.vstack([Vx, Vy, Vz]).T, axis=1)

    T = _as_float_array(df.get(tt, pd.Series(np.nan, index=df.index)))

    return {"B_x": Bx, "B_y": By, "B_z": Bz, "B_mag": Bm, "n": n, "V_mag": Vm, "T": T}


def make_stackplot_multi(
    df: pd.DataFrame,
    title_extra: str,
    show_ballistic: bool,
    show_bmag: bool,
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

    fig = plt.figure(figsize=(14, 14))
    gs = GridSpec(7, 1, figure=fig, hspace=0.08)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(7)]

    axes[0].plot(t_dt, sc["B_mag"], linewidth=1.2, color="black", label=f"{spacecraft_name} |B|")
    for lab, mode, ls, col in overlays:
        l1 = _get_l1_arrays(df, mode)
        if np.isfinite(l1["B_mag"]).any():
            axes[0].plot(t_dt, l1["B_mag"], linewidth=1.0, linestyle=ls, color=col, alpha=0.9, label=f"{lab} |B|")
    axes[0].set_ylabel("|B| [nT]", fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_ylim(bottom=0)

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
    fig.tight_layout()
    return fig


# =============================================================================
# PDF PATH HELPERS (avoid directory path crashes)
# =============================================================================

def _resolve_pdf_path(p: str, default_name: str) -> Path:
    path = Path(p).expanduser().resolve()

    if path.exists() and path.is_dir():
        path = path / default_name

    if path.suffix.lower() != ".pdf":
        path = path.with_suffix(".pdf")

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    cfg = FixedConfig()

    if cfg.SANITY_RANDOM_SEED is None:
        random.seed()
    else:
        random.seed(cfg.SANITY_RANDOM_SEED)

    events_dir = Path(cfg.EVENTS_DIR).expanduser().resolve()
    params_path = Path(cfg.PARAMS_FILE).expanduser().resolve()

    print("\n" + "=" * 90)
    print("XCORR BUILDER START")
    print("=" * 90)
    print(f"[Config] EVENTS_DIR = {events_dir}")
    print(f"[Config] PARAMS_FILE = {params_path}")
    print(f"[Config] MAX_LAG_SECONDS={cfg.MAX_LAG_SECONDS}  BIN_MIN={cfg.CORR_BIN_MINUTES}  MIN_OVERLAP={cfg.MIN_OVERLAP_POINTS}  ZSCORE={cfg.ZSCORE}")
    print(f"[Config] SAVE_IN_PLACE={cfg.SAVE_IN_PLACE}")
    print(f"[Config] SAVE_XCORR_PDF={cfg.SAVE_XCORR_PDF}  XCORR_PDF_PATH={cfg.XCORR_PDF_PATH}")
    print("=" * 90)

    if not events_dir.exists():
        raise FileNotFoundError(f"EVENTS_DIR not found: {events_dir}")
    if not params_path.exists():
        raise FileNotFoundError(f"PARAMS_FILE not found: {params_path}")

    params = pd.read_csv(params_path)
    params.columns = [str(c).strip() for c in params.columns]
    print(f"[Params] Loaded rows={len(params)} cols={len(params.columns)}")
    print(f"[Params] Columns: {list(params.columns)}")

    for col in ["File", "Corr_prop_needed?", "WINDOW_START_UTC", "EndWINDOW_END_UTC"]:
        if col not in params.columns:
            raise ValueError(f"Params CSV missing required column '{col}'. Found: {list(params.columns)}")

    xcorr_pdf: Optional[PdfPages] = None
    if cfg.SAVE_XCORR_PDF:
        xcorr_path = _resolve_pdf_path(cfg.XCORR_PDF_PATH, "xcorr_all_events.pdf")
        print(f"[PDF] XCORR -> {xcorr_path}")
        xcorr_pdf = PdfPages(xcorr_path)

    processed_for_sanity: List[Tuple[str, pd.DataFrame, str]] = []
    n_updated = 0
    n_skipped_compute = 0

    # Iterate in PARAMS order to guarantee plot order matches your existing ballistic PDF from main.py
    for idx in range(len(params)):
        row = params.iloc[idx]
        fname = str(row["File"]).strip()
        needed = _norm_bool(row["Corr_prop_needed?"])

        print("\n" + "-" * 90)
        print(f"[Row {idx}] {fname}")
        print(f"[Row {idx}] Corr_prop_needed? = {needed}")

        csv_path = events_dir / fname
        if not csv_path.exists():
            print(f"[Row {idx}] [ERROR] Missing event CSV: {csv_path} -> skipping page")
            continue

        df = pd.read_csv(csv_path)
        print(f"[Row {idx}] [Load] rows={len(df)} cols={len(df.columns)}")

        # Default plot behavior for the XCORR PDF:
        # if no xcorr computed, show ballistic overlay (so your pages still align with main.py’s ballistic PDF)
        plot_df = df
        title_extra = "No xcorr applied (showing ballistic)"
        show_ballistic = True
        show_bmag = False

        if needed is True:
            try:
                t_start = _parse_utc_to_unix_seconds(row["WINDOW_START_UTC"])
                t_end = _parse_utc_to_unix_seconds(row["EndWINDOW_END_UTC"])
            except Exception as e:
                print(f"[Row {idx}] [SkipCompute] Window parse failed: {e}")
                n_skipped_compute += 1
            else:
                if t_end <= t_start:
                    print(f"[Row {idx}] [SkipCompute] Bad window ordering (end <= start)")
                    n_skipped_compute += 1
                else:
                    repeated = _norm_bool(row["REPEATED MEASUREMENT?"]) if "REPEATED MEASUREMENT?" in params.columns else None
                    clear_cme = _norm_bool(row["CLEAR CME?"]) if "CLEAR CME?" in params.columns else None

                    print(f"[Row {idx}] [Window] {pd.to_datetime(t_start, unit='s', utc=True)} -> {pd.to_datetime(t_end, unit='s', utc=True)}")
                    print(f"[Row {idx}] [Flags] repeated_measurements={repeated}  clear_cme={clear_cme}")

                    if "B_mag" not in df.columns or "l1_B_mag_ballistic" not in df.columns:
                        print(f"[Row {idx}] [SkipCompute] Missing B_mag or l1_B_mag_ballistic")
                        n_skipped_compute += 1
                    else:
                        df_work = drop_old_shift_and_xcorr(df)

                        print(f"[Row {idx}] [Lag] Computing best |B| lag (manual_corr core, derivative OFF)")
                        t = _parse_time_seconds(df_work)
                        sc_bmag = _as_float_array(df_work["B_mag"])
                        l1_bmag_ball = _as_float_array(df_work["l1_B_mag_ballistic"])

                        try:
                            lag_bmag, corr_bmag, n_bins = _best_lag_for_bmag(
                                t=t,
                                sc_bmag=sc_bmag,
                                l1_bmag_ballistic=l1_bmag_ball,
                                t_start=t_start,
                                t_end=t_end,
                                cfg=cfg,
                            )
                        except Exception as e:
                            print(f"[Row {idx}] [SkipCompute] Lag search failed: {e}")
                            n_skipped_compute += 1
                        else:
                            print(f"[Row {idx}] [LagResult] lag={lag_bmag:.3f}s ({lag_bmag/3600:.3f}h) corr={corr_bmag:.4f} n_bins={n_bins}")

                            print(f"[Row {idx}] [Shift] Applying lag -> l1_*_shifted_bmag")
                            df2 = apply_lag_to_all_l1_columns(df_work, lag_bmag, tag="bmag")

                            print(f"[Row {idx}] [XCORR] Writing l1_*_xcorr (same lag)")
                            df2 = write_xcorr_columns_from_lag(df2, lag_bmag)

                            print(f"[Row {idx}] [Meta] Writing xcorr_* metadata + flags")
                            df2["xcorr_lag_seconds"] = float(lag_bmag)
                            df2["xcorr_corr_coeff"] = float(corr_bmag)
                            df2["xcorr_n_overlap"] = int(n_bins)
                            df2["xcorr_bin_minutes"] = int(cfg.CORR_BIN_MINUTES)
                            df2["xcorr_min_overlap_points"] = int(cfg.MIN_OVERLAP_POINTS)
                            df2["xcorr_zscore"] = int(bool(cfg.ZSCORE))
                            df2["xcorr_window_start_utc"] = str(pd.to_datetime(t_start, unit="s", utc=True))
                            df2["xcorr_window_end_utc"] = str(pd.to_datetime(t_end, unit="s", utc=True))

                            if "repeated_measurements" not in df2.columns:
                                df2["repeated_measurements"] = np.nan
                            if "clear_cme" not in df2.columns:
                                df2["clear_cme"] = np.nan
                            if repeated is not None:
                                df2["repeated_measurements"] = bool(repeated)
                            if clear_cme is not None:
                                df2["clear_cme"] = bool(clear_cme)

                            out_path = csv_path if cfg.SAVE_IN_PLACE else csv_path.with_name(csv_path.stem + "_xcorr" + csv_path.suffix)
                            df2.to_csv(out_path, index=False)
                            print(f"[Row {idx}] [Save] Wrote: {out_path}")

                            plot_df = df2
                            title_extra = (
                                f"XCORR applied: |B| lag={lag_bmag:.0f}s ({lag_bmag/3600:.2f}h), corr={corr_bmag:.2f}, bins={n_bins} | "
                                f"Window: {row['WINDOW_START_UTC']} -> {row['EndWINDOW_END_UTC']}"
                            )
                            show_ballistic = False
                            show_bmag = True

                            processed_for_sanity.append((fname, df2, f"lag={lag_bmag:.0f}s corr={corr_bmag:.2f}"))
                            n_updated += 1

        if xcorr_pdf is not None:
            fig = make_stackplot_multi(
                plot_df,
                title_extra=title_extra,
                show_ballistic=show_ballistic,
                show_bmag=show_bmag,
            )
            xcorr_pdf.savefig(fig)
            plt.close(fig)
            print(f"[Row {idx}] [PDF] Page added")

    if xcorr_pdf is not None:
        xcorr_pdf.close()
        print("\n" + "=" * 90)
        print(f"[PDF] XCORR complete: {_resolve_pdf_path(cfg.XCORR_PDF_PATH, 'xcorr_all_events.pdf')}")
        print("=" * 90)

    print("\n" + "=" * 90)
    print("XCORR BUILDER SUMMARY")
    print("=" * 90)
    print(f"[XCORR] Computed+Saved: {n_updated} | ComputeSkipped: {n_skipped_compute} | Folder: {events_dir}")
    print("=" * 90)

    if cfg.SHOW_ONE_SANITY_PLOT_INLINE and processed_for_sanity:
        pick = random.choice(processed_for_sanity)
        fname, dfp, extra = pick
        print("\n" + "-" * 90)
        print(f"[SanityPlot] Inline random: {fname} ({extra})")
        fig = make_stackplot_multi(
            dfp,
            title_extra=f"XCORR sanity: {extra}",
            show_ballistic=False,
            show_bmag=True,
        )
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()