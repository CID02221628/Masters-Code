import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CSV_FOLDER = '/Users/henryhodges/Documents/Year 4/Masters/Code/figures/primary/csv'

PLOT_FIGSIZE_TRIPLE = (18, 6)
PLOT_FIGSIZE_SINGLE = (9, 5)

SCATTER_SIZE = 55
SCATTER_ALPHA = 0.75
GRID_ALPHA = 0.25
GRID_STYLE = "--"
LINE_WIDTH = 2.0
FIT_LINE_STYLE = "--"

FONT_LABEL = 11
FONT_TITLE = 12
FONT_SUPTITLE = 14


PLOT_COLORS = {
    "rmse": "#1f77b4",
    "peak_sym": "#ff7f0e",
    "dv": "#2ca02c",
    "dt": "#d62728",
    "dst": "#9467bd",
}


# --------------------------------- Utilities -------------------------------- #

def _list_event_csvs(csv_folder: str):
    if not os.path.exists(csv_folder):
        print(f"❌ CSV folder not found: {csv_folder}")
        return []
    csv_files = glob.glob(os.path.join(csv_folder, "event_*.csv"))
    if not csv_files:
        print(f"❌ No CSV files found in {csv_folder}")
    return csv_files


def _compute_distance_metrics(df: pd.DataFrame):
    if "sc_distance_to_earth_au" not in df.columns:
        return None

    abs_dist = pd.to_numeric(df["sc_distance_to_earth_au"], errors="coerce").to_numpy(dtype=float)

    if "sc_angle_from_sun_earth_line_deg" in df.columns:
        angle_deg = pd.to_numeric(df["sc_angle_from_sun_earth_line_deg"], errors="coerce").to_numpy(dtype=float)
        perp = abs_dist * np.sin(np.radians(angle_deg))
        downstream = abs_dist * np.cos(np.radians(angle_deg))
    else:
        needed = ["sc_x_au", "sc_y_au", "earth_x_au", "earth_y_au"]
        if not all(c in df.columns for c in needed):
            return None

        sc_x = pd.to_numeric(df["sc_x_au"], errors="coerce").to_numpy(dtype=float)
        sc_y = pd.to_numeric(df["sc_y_au"], errors="coerce").to_numpy(dtype=float)
        earth_x = pd.to_numeric(df["earth_x_au"], errors="coerce").to_numpy(dtype=float)
        earth_y = pd.to_numeric(df["earth_y_au"], errors="coerce").to_numpy(dtype=float)

        dx = sc_x - earth_x
        dy = sc_y - earth_y

        sun_earth_x = -earth_x
        sun_earth_y = -earth_y
        sun_earth_norm = np.sqrt(sun_earth_x**2 + sun_earth_y**2)

        with np.errstate(invalid="ignore", divide="ignore"):
            hat_x = sun_earth_x / sun_earth_norm
            hat_y = sun_earth_y / sun_earth_norm

        downstream = dx * hat_x + dy * hat_y
        perp = np.abs(dx * hat_y - dy * hat_x)

    mean_perp = float(np.nanmean(perp))
    mean_downstream = float(np.nanmean(downstream))
    mean_abs = float(np.nanmean(abs_dist))

    if not (np.isfinite(mean_perp) and np.isfinite(mean_downstream) and np.isfinite(mean_abs)):
        return None

    return mean_perp, mean_downstream, mean_abs


def _scatter_with_linear_fit(
    ax,
    x,
    y,
    xlabel,
    ylabel,
    title,
    color,
    plot_fit: bool = True,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ax.scatter(
        x,
        y,
        s=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
        c=color,
        edgecolors="black",
        linewidth=0.7,
    )

    mask = np.isfinite(x) & np.isfinite(y)
    x_m = x[mask]
    y_m = y[mask]

    if plot_fit and len(x_m) >= 3 and np.nanmax(x_m) > np.nanmin(x_m):
        z = np.polyfit(x_m, y_m, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(np.nanmin(x_m), np.nanmax(x_m), 200)
        ax.plot(x_fit, p(x_fit), FIT_LINE_STYLE, linewidth=LINE_WIDTH, alpha=0.85, color=color)

        residuals = y_m - p(x_m)
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((y_m - np.mean(y_m))**2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        ax.text(
            0.05,
            0.95,
            f"y = {z[0]:.3f}x + {z[1]:.3f}\nR² = {r2:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.85),
        )

    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE)
    ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_STYLE)

# --------------------------------- Analysis --------------------------------- #

def plot_distance_vs_rmse_ballistic(
    csv_folder: str = CSV_FOLDER,
    min_points: int = 10,
    exclude_l1_zeros: bool = True,
    enforce_l1_available_flag: bool = True,
):
    csv_files = _list_event_csvs(csv_folder)
    if not csv_files:
        return None

    data = {"perp_dist": [], "downstream_dist": [], "abs_dist": [], "rmse": []}
    n_events = 0

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            if "B_mag" not in df.columns or "l1_B_mag_ballistic" not in df.columns:
                continue

            if enforce_l1_available_flag and "l1_ballistic_available" in df.columns:
                avail = pd.to_numeric(df["l1_ballistic_available"], errors="coerce").to_numpy(dtype=float)
                df = df[np.isfinite(avail) & (avail == 1.0)]
                if df.empty:
                    continue

            b_sc = pd.to_numeric(df["B_mag"], errors="coerce").to_numpy(dtype=float)
            b_l1 = pd.to_numeric(df["l1_B_mag_ballistic"], errors="coerce").to_numpy(dtype=float)

            valid = np.isfinite(b_sc) & np.isfinite(b_l1)
            if exclude_l1_zeros:
                valid = valid & (b_l1 != 0.0)

            if valid.sum() < min_points:
                continue

            rmse = float(np.sqrt(np.mean((b_sc[valid] - b_l1[valid]) ** 2)))

            dist = _compute_distance_metrics(df)
            if dist is None:
                continue

            mean_perp, mean_downstream, mean_abs = dist

            data["perp_dist"].append(mean_perp)
            data["downstream_dist"].append(mean_downstream)
            data["abs_dist"].append(mean_abs)
            data["rmse"].append(rmse)
            n_events += 1

        except Exception:
            continue

    if n_events == 0:
        print("❌ No valid events for RMSE plot.")
        return None

    for k in data:
        data[k] = np.array(data[k], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=PLOT_FIGSIZE_TRIPLE)

    distance_types = [
        ("perp_dist", "Distance from Sun-Earth Line [AU]", "Perpendicular Distance"),
        ("downstream_dist", "Distance Downstream from Earth [AU]", "Downstream Distance"),
        ("abs_dist", "Absolute Distance from Earth [AU]", "Absolute Distance"),
    ]

    color = PLOT_COLORS["rmse"]
    for ax, (dist_key, xlabel, suffix) in zip(axes, distance_types):
        _scatter_with_linear_fit(
            ax=ax,
            x=data[dist_key],
            y=data["rmse"],
            xlabel=xlabel,
            ylabel="RMSE [nT]",
            title=f"vs {suffix}",
            color=color,
            plot_fit=True,
        )

    fig.suptitle("Ballistic: RMSE vs Distance Metrics", fontsize=FONT_SUPTITLE)
    fig.tight_layout()
    print(f"✓ RMSE: processed {n_events} events")
    return fig


def plot_normalized_bmag_peak_sym_error_vs_distance_ballistic(
    csv_folder: str = CSV_FOLDER,
    min_points: int = 10,
    exclude_l1_zeros: bool = True,
    background_quantile: float = 20.0,
    peak_quantile: float = 95.0,
    eps: float = 1e-6,
    enforce_l1_available_flag: bool = True,
):
    csv_files = _list_event_csvs(csv_folder)
    if not csv_files:
        return None

    data = {"perp_dist": [], "downstream_dist": [], "abs_dist": [], "sym_norm_peak_err": []}
    n_events = 0

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            if "B_mag" not in df.columns or "l1_B_mag_ballistic" not in df.columns:
                continue

            if enforce_l1_available_flag and "l1_ballistic_available" in df.columns:
                avail = pd.to_numeric(df["l1_ballistic_available"], errors="coerce").to_numpy(dtype=float)
                df = df[np.isfinite(avail) & (avail == 1.0)]
                if df.empty:
                    continue

            b_sc = pd.to_numeric(df["B_mag"], errors="coerce").to_numpy(dtype=float)
            b_l1 = pd.to_numeric(df["l1_B_mag_ballistic"], errors="coerce").to_numpy(dtype=float)

            sc_mask = np.isfinite(b_sc)
            l1_mask = np.isfinite(b_l1)
            if exclude_l1_zeros:
                l1_mask = l1_mask & (b_l1 != 0.0)

            if sc_mask.sum() < min_points or l1_mask.sum() < min_points:
                continue

            b_sc_v = b_sc[sc_mask]
            b_l1_v = b_l1[l1_mask]

            sc_q = np.nanpercentile(b_sc_v, background_quantile)
            l1_q = np.nanpercentile(b_l1_v, background_quantile)

            sc_bg = np.nanmedian(b_sc_v[b_sc_v <= sc_q]) if np.any(b_sc_v <= sc_q) else np.nanmedian(b_sc_v)
            l1_bg = np.nanmedian(b_l1_v[b_l1_v <= l1_q]) if np.any(b_l1_v <= l1_q) else np.nanmedian(b_l1_v)

            if not np.isfinite(sc_bg) or not np.isfinite(l1_bg):
                continue

            sc_peak = float(np.nanpercentile(b_sc_v, peak_quantile))
            l1_peak = float(np.nanpercentile(b_l1_v, peak_quantile))

            p_sc = (sc_peak - sc_bg) / (sc_bg + eps)
            p_l1 = (l1_peak - l1_bg) / (l1_bg + eps)

            y_abs = abs(p_sc - p_l1)
            y_sym = (2.0 * y_abs) / (abs(p_sc) + abs(p_l1) + eps)

            dist = _compute_distance_metrics(df)
            if dist is None:
                continue

            mean_perp, mean_downstream, mean_abs = dist

            data["perp_dist"].append(mean_perp)
            data["downstream_dist"].append(mean_downstream)
            data["abs_dist"].append(mean_abs)
            data["sym_norm_peak_err"].append(y_sym)

            n_events += 1

        except Exception:
            continue

    if n_events == 0:
        print("❌ No valid events for normalized |B| peak symmetric error plot.")
        return None

    for k in data:
        data[k] = np.array(data[k], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=PLOT_FIGSIZE_TRIPLE)

    distance_types = [
        ("perp_dist", "Distance from Sun-Earth Line [AU]", "Perpendicular Distance"),
        ("downstream_dist", "Distance Downstream from Earth [AU]", "Downstream Distance"),
        ("abs_dist", "Absolute Distance from Earth [AU]", "Absolute Distance"),
    ]

    color = PLOT_COLORS["peak_sym"]
    for ax, (dist_key, xlabel, suffix) in zip(axes, distance_types):
        _scatter_with_linear_fit(
            ax=ax,
            x=data[dist_key],
            y=data["sym_norm_peak_err"],
            xlabel=xlabel,
            ylabel="Difference in |B| Peak",
            title=f"vs {suffix}",
            color=color,
            plot_fit=True,
        )

    fig.suptitle("Ballistic: Difference in |B| Peak vs Distance", fontsize=FONT_SUPTITLE)
    fig.tight_layout()
    print(f"✓ Peak symmetric error: processed {n_events} events")
    return fig


def plot_avg_rmse_vs_resolution_ballistic(
    csv_folder: str = CSV_FOLDER,
    resolutions_minutes=None,
    min_points_per_event: int = 10,
    exclude_l1_zeros: bool = True,
    enforce_l1_available_flag: bool = True,
):
    if resolutions_minutes is None:
        resolutions_minutes = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180]

    csv_files = _list_event_csvs(csv_folder)
    if not csv_files:
        return None

    rmse_by_res = []

    for res_min in resolutions_minutes:
        total_sq_err = 0.0
        total_n = 0

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)

                if "B_mag" not in df.columns or "l1_B_mag_ballistic" not in df.columns:
                    continue

                if enforce_l1_available_flag and "l1_ballistic_available" in df.columns:
                    avail = pd.to_numeric(df["l1_ballistic_available"], errors="coerce").to_numpy(dtype=float)
                    df = df[np.isfinite(avail) & (avail == 1.0)]
                    if df.empty:
                        continue

                if "timestamp" in df.columns:
                    t = pd.to_datetime(df["timestamp"], errors="coerce")
                elif "unix_timestamp" in df.columns:
                    t = pd.to_datetime(pd.to_numeric(df["unix_timestamp"], errors="coerce"), unit="s", errors="coerce")
                else:
                    continue

                sub = pd.DataFrame(
                    {
                        "t": t,
                        "B_mag": pd.to_numeric(df["B_mag"], errors="coerce"),
                        "l1_B_mag_ballistic": pd.to_numeric(df["l1_B_mag_ballistic"], errors="coerce"),
                    }
                ).dropna(subset=["t"]).set_index("t").sort_index()

                if sub.empty:
                    continue

                sub_r = sub.resample(f"{int(res_min)}min").mean()

                x = sub_r["B_mag"].to_numpy(dtype=float)
                y = sub_r["l1_B_mag_ballistic"].to_numpy(dtype=float)

                valid = np.isfinite(x) & np.isfinite(y)
                if exclude_l1_zeros:
                    valid = valid & (y != 0.0)

                if valid.sum() < min_points_per_event:
                    continue

                err = x[valid] - y[valid]
                total_sq_err += float(np.sum(err**2))
                total_n += int(valid.sum())

            except Exception:
                continue

        rmse_by_res.append(float(np.sqrt(total_sq_err / total_n)) if total_n > 0 else np.nan)

    resolutions = np.array(resolutions_minutes, dtype=float)
    rmse_vals = np.array(rmse_by_res, dtype=float)

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_SINGLE)
    ax.plot(resolutions, rmse_vals, marker="o", linewidth=LINE_WIDTH, color="#444444")
    ax.set_xlabel("Resolution (minutes)", fontsize=FONT_LABEL)
    ax.set_ylabel("Global RMSE [nT]", fontsize=FONT_LABEL)
    ax.set_title("Ballistic: Global RMSE vs Resolution", fontsize=FONT_TITLE)
    ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_STYLE)
    fig.tight_layout()
    return fig


def plot_velocity_jump_sym_error_vs_distance_ballistic(
    csv_folder: str = CSV_FOLDER,
    smooth_window: str = "30min",
    baseline_window: str = "60min",
    min_points_per_window: int = 6,
    exclude_l1_zeros: bool = True,
    enforce_l1_available_flag: bool = True,
    eps: float = 1e-6,
):
    csv_files = _list_event_csvs(csv_folder)
    if not csv_files:
        return None

    data = {"perp_dist": [], "downstream_dist": [], "abs_dist": [], "sym_dv_err": []}
    n_events = 0

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            if enforce_l1_available_flag and "l1_ballistic_available" in df.columns:
                avail = pd.to_numeric(df["l1_ballistic_available"], errors="coerce").to_numpy(dtype=float)
                df = df[np.isfinite(avail) & (avail == 1.0)]
                if df.empty:
                    continue

            if "timestamp" in df.columns:
                t = pd.to_datetime(df["timestamp"], errors="coerce")
            elif "unix_timestamp" in df.columns:
                t = pd.to_datetime(pd.to_numeric(df["unix_timestamp"], errors="coerce"), unit="s", errors="coerce")
            else:
                continue

            if "swa_V_mag" in df.columns:
                v_sc = pd.to_numeric(df["swa_V_mag"], errors="coerce")
            else:
                needed_sc = ["swa_V_r", "swa_V_t", "swa_V_n"]
                if not all(c in df.columns for c in needed_sc):
                    continue
                v_r = pd.to_numeric(df["swa_V_r"], errors="coerce")
                v_t = pd.to_numeric(df["swa_V_t"], errors="coerce")
                v_n = pd.to_numeric(df["swa_V_n"], errors="coerce")
                v_sc = np.sqrt(v_r**2 + v_t**2 + v_n**2)

            if "l1_V_mag_ballistic" in df.columns:
                v_l1 = pd.to_numeric(df["l1_V_mag_ballistic"], errors="coerce")
            else:
                needed_l1 = ["l1_V_x_ballistic", "l1_V_y_ballistic", "l1_V_z_ballistic"]
                if not all(c in df.columns for c in needed_l1):
                    continue
                vx = pd.to_numeric(df["l1_V_x_ballistic"], errors="coerce")
                vy = pd.to_numeric(df["l1_V_y_ballistic"], errors="coerce")
                vz = pd.to_numeric(df["l1_V_z_ballistic"], errors="coerce")
                v_l1 = np.sqrt(vx**2 + vy**2 + vz**2)

            sub = pd.DataFrame({"t": t, "v_sc": v_sc, "v_l1": v_l1}).dropna(subset=["t"]).set_index("t").sort_index()
            if sub.empty:
                continue

            if exclude_l1_zeros:
                sub.loc[sub["v_l1"] == 0.0, "v_l1"] = np.nan

            if sub["v_sc"].notna().sum() < (2 * min_points_per_window) or sub["v_l1"].notna().sum() < (2 * min_points_per_window):
                continue

            vsc_s = sub["v_sc"].rolling(smooth_window, min_periods=min_points_per_window).median()
            vl1_s = sub["v_l1"].rolling(smooth_window, min_periods=min_points_per_window).median()

            dt_s = vsc_s.index.to_series().diff().dt.total_seconds()
            dVdt_sc = vsc_s.diff() / dt_s
            dVdt_l1 = vl1_s.diff() / dt_s

            def _pick_jump_time(dvdt: pd.Series):
                dvdt = dvdt.replace([np.inf, -np.inf], np.nan).dropna()
                dvdt_pos = dvdt[dvdt > 0]
                if dvdt_pos.empty:
                    return None
                return dvdt_pos.idxmax()

            t0_sc = _pick_jump_time(dVdt_sc)
            t0_l1 = _pick_jump_time(dVdt_l1)
            if t0_sc is None or t0_l1 is None:
                continue

            def _delta_v(v_smooth: pd.Series, t0: pd.Timestamp):
                pre = v_smooth.loc[(v_smooth.index >= (t0 - pd.Timedelta(baseline_window))) & (v_smooth.index < t0)].dropna()
                post = v_smooth.loc[(v_smooth.index > t0) & (v_smooth.index <= (t0 + pd.Timedelta(baseline_window)))].dropna()
                if len(pre) < min_points_per_window or len(post) < min_points_per_window:
                    return None
                dv = float(np.nanmedian(post) - np.nanmedian(pre))
                if not np.isfinite(dv) or dv <= 0:
                    return None
                return dv

            dv_sc = _delta_v(vsc_s, t0_sc)
            dv_l1 = _delta_v(vl1_s, t0_l1)
            if dv_sc is None or dv_l1 is None:
                continue

            sym_err = (2.0 * abs(dv_sc - dv_l1)) / (abs(dv_sc) + abs(dv_l1) + eps)

            dist = _compute_distance_metrics(df)
            if dist is None:
                continue
            mean_perp, mean_downstream, mean_abs = dist

            data["perp_dist"].append(mean_perp)
            data["downstream_dist"].append(mean_downstream)
            data["abs_dist"].append(mean_abs)
            data["sym_dv_err"].append(sym_err)
            n_events += 1

        except Exception:
            continue

    if n_events == 0:
        print("❌ No valid events for ΔV symmetric error plot.")
        return None

    for k in data:
        data[k] = np.array(data[k], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=PLOT_FIGSIZE_TRIPLE)

    distance_types = [
        ("perp_dist", "Distance from Sun-Earth Line [AU]", "Perpendicular Distance"),
        ("downstream_dist", "Distance Downstream from Earth [AU]", "Downstream Distance"),
        ("abs_dist", "Absolute Distance from Earth [AU]", "Absolute Distance"),
    ]

    color = PLOT_COLORS["dv"]
    for ax, (dist_key, xlabel, suffix) in zip(axes, distance_types):
        _scatter_with_linear_fit(
            ax=ax,
            x=data[dist_key],
            y=data["sym_dv_err"],
            xlabel=xlabel,
            ylabel="Difference in ΔV",
            title=f"vs {suffix}",
            color=color,
            plot_fit=False,
        )

    fig.suptitle("Ballistic: Δ(ΔV) vs Distance", fontsize=FONT_SUPTITLE)
    fig.tight_layout()
    print(f"✓ ΔV symmetric error: processed {n_events} events")
    return fig


def plot_optimal_time_offset_vs_distance_ballistic(
    csv_folder: str = CSV_FOLDER,
    min_overlap_points: int = 30,
    exclude_l1_zeros: bool = True,
    enforce_l1_available_flag: bool = True,
    resample_rule: str = "1min",
    lag_search_hours: float = 6.0,
    lag_step_minutes: int = 5,
    error_metric: str = "mae",
):
    csv_files = _list_event_csvs(csv_folder)
    if not csv_files:
        return None

    if error_metric not in {"rmse", "mae"}:
        raise ValueError("error_metric must be one of: 'rmse', 'mae'")

    lags_min = np.arange(
        -int(lag_search_hours * 60),
        int(lag_search_hours * 60) + lag_step_minutes,
        lag_step_minutes,
        dtype=int,
    )

    data = {
        "perp_dist": [],
        "downstream_dist": [],
        "abs_dist": [],
        "delta_t_hours": [],
        "err_ballistic": [],
        "err_opt": [],
    }

    n_events = 0

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            if "B_mag" not in df.columns or "l1_B_mag_ballistic" not in df.columns:
                continue

            if enforce_l1_available_flag and "l1_ballistic_available" in df.columns:
                avail = pd.to_numeric(df["l1_ballistic_available"], errors="coerce").to_numpy(dtype=float)
                df = df[np.isfinite(avail) & (avail == 1.0)]
                if df.empty:
                    continue

            if "timestamp" in df.columns:
                t = pd.to_datetime(df["timestamp"], errors="coerce")
            elif "unix_timestamp" in df.columns:
                t = pd.to_datetime(pd.to_numeric(df["unix_timestamp"], errors="coerce"), unit="s", errors="coerce")
            else:
                continue

            sub = pd.DataFrame(
                {
                    "t": t,
                    "B_sc": pd.to_numeric(df["B_mag"], errors="coerce"),
                    "B_l1": pd.to_numeric(df["l1_B_mag_ballistic"], errors="coerce"),
                }
            ).dropna(subset=["t"]).set_index("t").sort_index()

            if sub.empty:
                continue

            if exclude_l1_zeros:
                sub.loc[sub["B_l1"] == 0.0, "B_l1"] = np.nan

            if resample_rule is not None:
                sub = sub.resample(resample_rule).mean()

            if sub["B_sc"].notna().sum() < min_overlap_points or sub["B_l1"].notna().sum() < min_overlap_points:
                continue

            def _err_for_lag_minutes(lag_m: int):
                l1_shifted = sub["B_l1"].shift(freq=pd.Timedelta(minutes=int(lag_m)))
                aligned = pd.concat([sub["B_sc"], l1_shifted.rename("B_l1s")], axis=1, join="inner")
                x = aligned["B_sc"].to_numpy(dtype=float)
                y = aligned["B_l1s"].to_numpy(dtype=float)
                valid = np.isfinite(x) & np.isfinite(y)
                if valid.sum() < min_overlap_points:
                    return np.nan
                d = x[valid] - y[valid]
                if error_metric == "mae":
                    return float(np.mean(np.abs(d)))
                return float(np.sqrt(np.mean(d**2)))

            err0 = _err_for_lag_minutes(0)
            if not np.isfinite(err0):
                continue

            errs = np.array([_err_for_lag_minutes(int(lm)) for lm in lags_min], dtype=float)
            if not np.any(np.isfinite(errs)):
                continue

            i_best = int(np.nanargmin(errs))
            best_lag_min = int(lags_min[i_best])
            best_err = float(errs[i_best])

            dist = _compute_distance_metrics(df)
            if dist is None:
                continue
            mean_perp, mean_downstream, mean_abs = dist

            data["perp_dist"].append(mean_perp)
            data["downstream_dist"].append(mean_downstream)
            data["abs_dist"].append(mean_abs)
            data["delta_t_hours"].append(best_lag_min / 60.0)
            data["err_ballistic"].append(err0)
            data["err_opt"].append(best_err)

            n_events += 1

        except Exception:
            continue

    if n_events == 0:
        print("❌ No valid events for optimal time-offset plot.")
        return None

    for k in data:
        data[k] = np.array(data[k], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=PLOT_FIGSIZE_TRIPLE)

    distance_types = [
        ("perp_dist", "Distance from Sun-Earth Line [AU]", "Perpendicular Distance"),
        ("downstream_dist", "Distance Downstream from Earth [AU]", "Downstream Distance"),
        ("abs_dist", "Absolute Distance from Earth [AU]", "Absolute Distance"),
    ]

    color = PLOT_COLORS["dt"]
    for ax, (dist_key, xlabel, suffix) in zip(axes, distance_types):
        _scatter_with_linear_fit(
            ax=ax,
            x=data[dist_key],
            y=data["delta_t_hours"],
            xlabel=xlabel,
            ylabel="Optimal - Ballistic time offset [hours]",
            title=f"vs {suffix}",
            color=color,
            plot_fit=False,
        )

    fig.suptitle(f"Ballistic: Optimal Time-Offset Correction vs Distance (metric={error_metric})", fontsize=FONT_SUPTITLE)
    fig.tight_layout()

    med_dt = float(np.nanmedian(data["delta_t_hours"]))
    med_improve = float(np.nanmedian(data["err_ballistic"] - data["err_opt"]))
    print(f"✓ Optimal time-offset: processed {n_events} events")
    print(f"  median offset: {med_dt:.3f} hours")
    print(f"  median improvement (ballistic - optimal): {med_improve:.3f}")

    return fig


def plot_min_dst_difference_vs_distance_obrien_ballistic(
    csv_folder: str = CSV_FOLDER,
    min_points: int = 60,
    resample_rule: str = "1min",
    exclude_l1_zeros: bool = True,
    enforce_l1_available_flag: bool = True,
    max_abs_delta_min_dst: float = 1000.0,
):
    csv_files = _list_event_csvs(csv_folder)
    if not csv_files:
        return None

    b = 7.26
    c = 11.0

    def _pdyn_npa(n_cm3: np.ndarray, v_kms: np.ndarray) -> np.ndarray:
        return 1.6726e-6 * n_cm3 * (v_kms**2)

    def _simulate_obrien_dst(t_index: pd.DatetimeIndex, bz_nT: np.ndarray, v_kms: np.ndarray, n_cm3: np.ndarray):
        bz = np.asarray(bz_nT, dtype=float)
        v = np.asarray(v_kms, dtype=float)
        n = np.asarray(n_cm3, dtype=float)

        bs = np.maximum(0.0, -bz)
        vbs = 1e-3 * v * bs
        q = np.where(vbs > 0.50, -4.4 * (vbs - 0.50), 0.0)
        tau = 2.4 * np.exp(9.74 / (4.69 + vbs))

        pdyn = _pdyn_npa(n, v)
        sqrt_p = np.sqrt(np.clip(pdyn, 0.0, np.inf))

        dt_hr = t_index.to_series().diff().dt.total_seconds().to_numpy(dtype=float) / 3600.0
        dt_hr[0] = np.nan

        dst_star = np.full(len(t_index), np.nan, dtype=float)
        dst = np.full(len(t_index), np.nan, dtype=float)

        if not np.isfinite(sqrt_p[0]):
            return None

        dst0 = 0.0
        dst_star[0] = dst0 - b * sqrt_p[0] + c
        dst[0] = dst0

        for i in range(1, len(t_index)):
            if not (np.isfinite(dt_hr[i]) and dt_hr[i] > 0):
                continue
            if not (np.isfinite(q[i]) and np.isfinite(tau[i]) and tau[i] > 0 and np.isfinite(dst_star[i - 1])):
                continue
            dst_star[i] = dst_star[i - 1] + dt_hr[i] * (q[i] - dst_star[i - 1] / tau[i])
            if np.isfinite(dst_star[i]) and np.isfinite(sqrt_p[i]):
                dst[i] = dst_star[i] + b * sqrt_p[i] - c

        return dst

    data = {"perp_dist": [], "downstream_dist": [], "abs_dist": [], "abs_delta_min_dst": []}
    n_events = 0

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            if enforce_l1_available_flag and "l1_ballistic_available" in df.columns:
                avail = pd.to_numeric(df["l1_ballistic_available"], errors="coerce").to_numpy(dtype=float)
                df = df[np.isfinite(avail) & (avail == 1.0)]
                if df.empty:
                    continue

            if "timestamp" in df.columns:
                t = pd.to_datetime(df["timestamp"], errors="coerce")
            elif "unix_timestamp" in df.columns:
                t = pd.to_datetime(pd.to_numeric(df["unix_timestamp"], errors="coerce"), unit="s", errors="coerce")
            else:
                continue

            if "B_n" not in df.columns or "swa_n" not in df.columns:
                continue

            if "swa_V_mag" in df.columns:
                v_sc = pd.to_numeric(df["swa_V_mag"], errors="coerce")
            else:
                needed_sc = ["swa_V_r", "swa_V_t", "swa_V_n"]
                if not all(c in df.columns for c in needed_sc):
                    continue
                vr = pd.to_numeric(df["swa_V_r"], errors="coerce")
                vt = pd.to_numeric(df["swa_V_t"], errors="coerce")
                vn = pd.to_numeric(df["swa_V_n"], errors="coerce")
                v_sc = np.sqrt(vr**2 + vt**2 + vn**2)

            if "l1_B_z_gse_ballistic" not in df.columns or "l1_n_ballistic" not in df.columns:
                continue

            if "l1_V_mag_ballistic" in df.columns:
                v_l1 = pd.to_numeric(df["l1_V_mag_ballistic"], errors="coerce")
            else:
                needed_l1 = ["l1_V_x_ballistic", "l1_V_y_ballistic", "l1_V_z_ballistic"]
                if not all(c in df.columns for c in needed_l1):
                    continue
                vx = pd.to_numeric(df["l1_V_x_ballistic"], errors="coerce")
                vy = pd.to_numeric(df["l1_V_y_ballistic"], errors="coerce")
                vz = pd.to_numeric(df["l1_V_z_ballistic"], errors="coerce")
                v_l1 = np.sqrt(vx**2 + vy**2 + vz**2)

            sub_sc = (
                pd.DataFrame(
                    {
                        "t": t,
                        "bz_sc": pd.to_numeric(df["B_n"], errors="coerce"),
                        "v_sc": pd.to_numeric(v_sc, errors="coerce"),
                        "n_sc": pd.to_numeric(df["swa_n"], errors="coerce"),
                    }
                )
                .dropna(subset=["t"])
                .set_index("t")
                .sort_index()
            )

            sub_l1 = (
                pd.DataFrame(
                    {
                        "t": t,
                        "bz_l1": pd.to_numeric(df["l1_B_z_gse_ballistic"], errors="coerce"),
                        "v_l1": pd.to_numeric(v_l1, errors="coerce"),
                        "n_l1": pd.to_numeric(df["l1_n_ballistic"], errors="coerce"),
                    }
                )
                .dropna(subset=["t"])
                .set_index("t")
                .sort_index()
            )

            if sub_sc.empty or sub_l1.empty:
                continue

            if exclude_l1_zeros:
                sub_l1.loc[sub_l1["v_l1"] == 0.0, "v_l1"] = np.nan
                sub_l1.loc[sub_l1["n_l1"] == 0.0, "n_l1"] = np.nan
                sub_l1.loc[sub_l1["bz_l1"] == 0.0, "bz_l1"] = np.nan

            if resample_rule is not None:
                sub_sc = sub_sc.resample(resample_rule).mean()
                sub_l1 = sub_l1.resample(resample_rule).mean()

            merged = sub_sc.join(sub_l1, how="inner").dropna(
                subset=["bz_sc", "v_sc", "n_sc", "bz_l1", "v_l1", "n_l1"]
            )

            if len(merged) < min_points:
                continue

            dst_sc = _simulate_obrien_dst(
                merged.index, merged["bz_sc"].to_numpy(), merged["v_sc"].to_numpy(), merged["n_sc"].to_numpy()
            )
            dst_l1 = _simulate_obrien_dst(
                merged.index, merged["bz_l1"].to_numpy(), merged["v_l1"].to_numpy(), merged["n_l1"].to_numpy()
            )

            if dst_sc is None or dst_l1 is None:
                continue

            min_sc = float(np.nanmin(dst_sc))
            min_l1 = float(np.nanmin(dst_l1))
            if not (np.isfinite(min_sc) and np.isfinite(min_l1)):
                continue

            abs_delta = abs(min_l1 - min_sc)
            if (not np.isfinite(abs_delta)) or (abs_delta > max_abs_delta_min_dst):
                continue

            dist = _compute_distance_metrics(df)
            if dist is None:
                continue
            mean_perp, mean_downstream, mean_abs = dist

            data["perp_dist"].append(mean_perp)
            data["downstream_dist"].append(mean_downstream)
            data["abs_dist"].append(mean_abs)
            data["abs_delta_min_dst"].append(abs_delta)

            n_events += 1

        except Exception:
            continue

    if n_events == 0:
        print("❌ No valid events for O'Brien |Δ(min Dst)| plot.")
        return None

    for k in data:
        data[k] = np.array(data[k], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=PLOT_FIGSIZE_TRIPLE)

    distance_types = [
        ("perp_dist", "Distance from Sun-Earth Line [AU]", "Perpendicular Distance"),
        ("downstream_dist", "Distance Downstream from Earth [AU]", "Downstream Distance"),
        ("abs_dist", "Absolute Distance from Earth [AU]", "Absolute Distance"),
    ]

    color = PLOT_COLORS["dst"]
    for ax, (dist_key, xlabel, suffix) in zip(axes, distance_types):
        _scatter_with_linear_fit(
            ax=ax,
            x=data[dist_key],
            y=data["abs_delta_min_dst"],
            xlabel=xlabel,
            ylabel="|Δ(min Dst)| [nT]",
            title=f"vs {suffix}",
            color=color,
            plot_fit=False,
        )

    fig.suptitle("Ballistic: O'Brien Model |Δ(min Dst)| vs Distance", fontsize=FONT_SUPTITLE)
    fig.tight_layout()
    print(f"✓ O'Brien |Δ(min Dst)|: processed {n_events} events")
    return fig


def plot_min_bz_difference_vs_distance_ballistic(
    csv_folder: str = CSV_FOLDER,
    min_points: int = 30,
    exclude_l1_zeros: bool = True,
    enforce_l1_available_flag: bool = True,
    use_absolute_difference: bool = True,
):
    csv_files = _list_event_csvs(csv_folder)
    if not csv_files:
        return None

    data = {
        "perp_dist": [],
        "downstream_dist": [],
        "abs_dist": [],
        "abs_delta_min_bz": [],
    }

    z_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    n_events = 0

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            if enforce_l1_available_flag and "l1_ballistic_available" in df.columns:
                avail = pd.to_numeric(df["l1_ballistic_available"], errors="coerce").to_numpy(dtype=float)
                df = df[np.isfinite(avail) & (avail == 1.0)]
                if df.empty:
                    continue

            needed = ["B_r", "B_t", "B_n", "sc_x_hci", "sc_y_hci", "sc_z_hci", "l1_B_z_gse_ballistic"]
            if not all(c in df.columns for c in needed):
                continue

            br = pd.to_numeric(df["B_r"], errors="coerce").to_numpy(dtype=float)
            bt = pd.to_numeric(df["B_t"], errors="coerce").to_numpy(dtype=float)
            bn = pd.to_numeric(df["B_n"], errors="coerce").to_numpy(dtype=float)

            scx = pd.to_numeric(df["sc_x_hci"], errors="coerce").to_numpy(dtype=float)
            scy = pd.to_numeric(df["sc_y_hci"], errors="coerce").to_numpy(dtype=float)
            scz = pd.to_numeric(df["sc_z_hci"], errors="coerce").to_numpy(dtype=float)

            bz_l1 = pd.to_numeric(df["l1_B_z_gse_ballistic"], errors="coerce").to_numpy(dtype=float)
            if exclude_l1_zeros:
                bz_l1 = np.where(bz_l1 == 0.0, np.nan, bz_l1)

            r = np.stack([scx, scy, scz], axis=1)  # (N,3)
            r_norm = np.linalg.norm(r, axis=1)
            valid_r = np.isfinite(r_norm) & (r_norm > 0)

            R_hat = np.full_like(r, np.nan, dtype=float)
            R_hat[valid_r] = r[valid_r] / r_norm[valid_r][:, None]

            T_raw = np.cross(z_hat[None, :], R_hat)
            T_norm = np.linalg.norm(T_raw, axis=1)
            valid_t = np.isfinite(T_norm) & (T_norm > 0)

            T_hat = np.full_like(r, np.nan, dtype=float)
            T_hat[valid_t] = T_raw[valid_t] / T_norm[valid_t][:, None]

            N_hat = np.cross(R_hat, T_hat)

            # B in HCI Cartesian from RTN components
            B_hci = (br[:, None] * R_hat) + (bt[:, None] * T_hat) + (bn[:, None] * N_hat)

            # GSE z ~ ecliptic north, same direction as HCI z_hat if your HCI z is ecliptic north
            bz_sc = B_hci[:, 2]

            mask = np.isfinite(bz_sc) & np.isfinite(bz_l1)
            if mask.sum() < min_points:
                continue

            min_bz_sc = float(np.nanmin(bz_sc[mask]))
            min_bz_l1 = float(np.nanmin(bz_l1[mask]))
            if not (np.isfinite(min_bz_sc) and np.isfinite(min_bz_l1)):
                continue

            delta = (min_bz_l1 - min_bz_sc)
            if use_absolute_difference:
                delta = abs(delta)

            dist = _compute_distance_metrics(df)
            if dist is None:
                continue
            mean_perp, mean_downstream, mean_abs = dist

            data["perp_dist"].append(mean_perp)
            data["downstream_dist"].append(mean_downstream)
            data["abs_dist"].append(mean_abs)
            data["abs_delta_min_bz"].append(delta)

            n_events += 1

        except Exception:
            continue

    if n_events == 0:
        print("❌ No valid events for |Δ(min Bz)| plot.")
        return None

    for k in data:
        data[k] = np.array(data[k], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=PLOT_FIGSIZE_TRIPLE)

    distance_types = [
        ("perp_dist", "Distance from Sun-Earth Line [AU]", "Perpendicular Distance"),
        ("downstream_dist", "Distance Downstream from Earth [AU]", "Downstream Distance"),
        ("abs_dist", "Absolute Distance from Earth [AU]", "Absolute Distance"),
    ]

    color = "#8c564b"  # distinct, not used yet
    for ax, (dist_key, xlabel, suffix) in zip(axes, distance_types):
        _scatter_with_linear_fit(
            ax=ax,
            x=data[dist_key],
            y=data["abs_delta_min_bz"],
            xlabel=xlabel,
            ylabel="|Δ(min Bz)| [nT]" if use_absolute_difference else "Δ(min Bz) [nT]",
            title=f"vs {suffix}",
            color=color,
            plot_fit=False,
        )

    fig.suptitle("Ballistic: |Δ(min Bz)| vs Distance", fontsize=FONT_SUPTITLE)
    fig.tight_layout()
    print(f"✓ |Δ(min Bz)|: processed {n_events} events")
    return fig


def plot_temperature_rmse_vs_distance_ballistic(
    csv_folder: str = CSV_FOLDER,
    min_points: int = 10,
    exclude_l1_zeros: bool = True,
    enforce_l1_available_flag: bool = True,
    plot_fit: bool = True,
    color: str = "#17becf",
):
    """
    Triple plot: per-event RMSE(T_sc vs T_L1_ballistic) vs distance metrics.

    Uses:
      - Spacecraft: swa_T
      - L1 ballistic: l1_T_ballistic
    """
    csv_files = _list_event_csvs(csv_folder)
    if not csv_files:
        return None

    data = {"perp_dist": [], "downstream_dist": [], "abs_dist": [], "rmse_T": []}
    n_events = 0

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            if "swa_T" not in df.columns or "l1_T_ballistic" not in df.columns:
                continue

            if enforce_l1_available_flag and "l1_ballistic_available" in df.columns:
                avail = pd.to_numeric(df["l1_ballistic_available"], errors="coerce").to_numpy(dtype=float)
                df = df[np.isfinite(avail) & (avail == 1.0)]
                if df.empty:
                    continue

            t_sc = pd.to_numeric(df["swa_T"], errors="coerce").to_numpy(dtype=float)
            t_l1 = pd.to_numeric(df["l1_T_ballistic"], errors="coerce").to_numpy(dtype=float)

            valid = np.isfinite(t_sc) & np.isfinite(t_l1)
            if exclude_l1_zeros:
                valid = valid & (t_l1 != 0.0)

            if valid.sum() < min_points:
                continue

            rmse_t = float(np.sqrt(np.mean((t_sc[valid] - t_l1[valid]) ** 2)))

            dist = _compute_distance_metrics(df)
            if dist is None:
                continue

            mean_perp, mean_downstream, mean_abs = dist

            data["perp_dist"].append(mean_perp)
            data["downstream_dist"].append(mean_downstream)
            data["abs_dist"].append(mean_abs)
            data["rmse_T"].append(rmse_t)

            n_events += 1

        except Exception:
            continue

    if n_events == 0:
        print("❌ No valid events for temperature RMSE plot.")
        return None

    for k in data:
        data[k] = np.array(data[k], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=PLOT_FIGSIZE_TRIPLE)

    distance_types = [
        ("perp_dist", "Distance from Sun-Earth Line [AU]", "Perpendicular Distance"),
        ("downstream_dist", "Distance Downstream from Earth [AU]", "Downstream Distance"),
        ("abs_dist", "Absolute Distance from Earth [AU]", "Absolute Distance"),
    ]

    for ax, (dist_key, xlabel, suffix) in zip(axes, distance_types):
        _scatter_with_linear_fit(
            ax=ax,
            x=data[dist_key],
            y=data["rmse_T"],
            xlabel=xlabel,
            ylabel="RMSE(T) [K]",
            title=f"vs {suffix}",
            color=color,
            plot_fit=plot_fit,
        )

    fig.suptitle("Ballistic: Temperature RMSE vs Distance Metrics", fontsize=FONT_SUPTITLE)
    fig.tight_layout()
    print(f"✓ RMSE(T): processed {n_events} events")
    return fig

# ----------------------------------- Main ----------------------------------- #

def main():
    print(f"\n{'='*80}")
    print("BALLISTIC PROPAGATION ANALYSIS")
    print(f"{'='*80}\n")
    print(f"CSV folder: {CSV_FOLDER}\n")

    fig_rmse = plot_distance_vs_rmse_ballistic(
        csv_folder=CSV_FOLDER,
        min_points=10,
        exclude_l1_zeros=True,
        enforce_l1_available_flag=True,
    )

    fig_peak_sym = plot_normalized_bmag_peak_sym_error_vs_distance_ballistic(
        csv_folder=CSV_FOLDER,
        min_points=10,
        exclude_l1_zeros=True,
        background_quantile=20.0,
        peak_quantile=95.0,
        enforce_l1_available_flag=True,
    )

    fig_res = plot_avg_rmse_vs_resolution_ballistic(
        csv_folder=CSV_FOLDER,
        resolutions_minutes=[5, 10, 15, 20, 30, 45, 60, 90, 120, 180],
        min_points_per_event=10,
        exclude_l1_zeros=True,
        enforce_l1_available_flag=True,
    )

    fig_dv = plot_velocity_jump_sym_error_vs_distance_ballistic(
        csv_folder=CSV_FOLDER,
        smooth_window="30min",
        baseline_window="60min",
        min_points_per_window=6,
        exclude_l1_zeros=True,
        enforce_l1_available_flag=True,
    )

    fig_dt = plot_optimal_time_offset_vs_distance_ballistic(
        csv_folder=CSV_FOLDER,
        min_overlap_points=30,
        exclude_l1_zeros=True,
        enforce_l1_available_flag=True,
        resample_rule="1min",
        lag_search_hours=6.0,
        lag_step_minutes=5,
        error_metric="mae",
    )

    fig_dst = plot_min_dst_difference_vs_distance_obrien_ballistic(
        csv_folder=CSV_FOLDER,
        min_points=60,
        resample_rule="1min",
        exclude_l1_zeros=True,
        enforce_l1_available_flag=True,
    )

    fig_bz = plot_min_bz_difference_vs_distance_ballistic(
        csv_folder=CSV_FOLDER,
        min_points=30,
        exclude_l1_zeros=True,
        enforce_l1_available_flag=True,
        use_absolute_difference=True,
    )

    fig_t = plot_temperature_rmse_vs_distance_ballistic(
        csv_folder=CSV_FOLDER,
        min_points=10,
        exclude_l1_zeros=True,
        enforce_l1_available_flag=True,
        plot_fit=True,
        color="#17becf",
    )
    
    figs = [fig_rmse, fig_peak_sym, fig_res, fig_dv, fig_dt, fig_dst, fig_bz, fig_t]

    if any(f is not None for f in figs):
        plt.show()
        print("\nAnalysis complete.")
    else:
        print("\nNo figures produced (no valid data processed).")


if __name__ == "__main__":
    main()
