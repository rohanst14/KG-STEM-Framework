
from __future__ import annotations

import argparse
import json
import os
import textwrap
from dataclasses import fields
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/kg_stgm_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import torch
import torch.nn as nn

from train import (
    KGSTGMForecaster,
    NODE_NAMES,
    RAW_X22_NAMES,
    TrainConfig,
    FeatureScaler,
    load_adjacency,
    load_tep_arrays,
    load_state_dict_file,
    resolve_device,
    set_seed,
)
from testing_normal import load_train_config, resolve_checkpoint, load_saved_scalers, _X22_DISPLAY_NAMES
from local_risk import FAULT_TABLE_SOURCE


MEASUREMENT_GROUPS = {
    "flow": {"nodes": {"x1", "x2", "x3", "x4", "x5", "x6", "x10", "x14", "x17", "x19"}, "mpl_marker": "o", "label": "Flow", "size_scale": 1.00},
    "pressure": {"nodes": {"x7", "x13", "x16"}, "mpl_marker": "D", "label": "Pressure", "size_scale": 1.00},
    "level": {"nodes": {"x8", "x12", "x15"}, "mpl_marker": "h", "label": "Level", "size_scale": 1.00},
    "temperature": {"nodes": {"x9", "x11", "x18", "x21", "x22"}, "mpl_marker": "^", "label": "Temperature", "size_scale": 1.30},
    "work_energy": {"nodes": {"x20"}, "mpl_marker": "p", "label": "Work/Energy", "size_scale": 1.00},
}

# Literature reports TE testing faults after the first 160 normal samples
# (i.e. sample 161 in 1-based sample numbering). Plots use a 0-based time axis.
FAULT_INTRO_TIME = 160
REACTOR_TEMP_X_INDEX = 8
REACTOR_TEMP_NORMAL_MU = 120.4
REACTOR_TEMP_NORMAL_LOW = 120.1
REACTOR_TEMP_NORMAL_HIGH = 120.7


def parse_int_ranges(raw: str, label: str) -> List[int]:
    """Parse a comma-separated list of integers and ranges (e.g. '1,3-5,7')."""
    if raw is None:
        raise ValueError(f"No {label} were provided.")
    parts = raw.split(",")
    values: List[int] = []
    for part in parts:
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start, end = int(start_s), int(end_s)
            step = 1 if start <= end else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(item))
    seen = set()
    unique: List[int] = []
    for value in values:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    if not unique:
        raise ValueError(f"No {label} were provided.")
    return unique


def parse_fault_ids(raw: str) -> List[int]:
    return parse_int_ranges(raw, "fault IDs")


def load_faulty_metadata(data: np.lib.npyio.NpzFile) -> Tuple[List[int], List[str], List[str], Dict[int, str]]:
    fault_ids = [int(v) for v in data["fault_ids"].tolist()] if "fault_ids" in data.files else list(range(1, 21))
    x_names = [str(v) for v in data["x_names"].tolist()] if "x_names" in data.files else RAW_X22_NAMES
    u_names = [str(v) for v in data["u_names"].tolist()] if "u_names" in data.files else []
    fault_name_by_id: Dict[int, str] = {}
    if "fault_names" in data.files:
        fault_names = [str(v) for v in data["fault_names"].tolist()]
        fault_name_by_id = {
            fault_id: fault_names[idx]
            for idx, fault_id in enumerate(fault_ids)
            if idx < len(fault_names)
        }
    return fault_ids, x_names, u_names, fault_name_by_id


def load_faulty_run(
    data: np.lib.npyio.NpzFile,
    fault_id: int,
    sim_id: int,
    n_x: int,
    n_u: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_key = f"fault_id_{fault_id}_sim_{sim_id}_X"
    u_key = f"fault_id_{fault_id}_sim_{sim_id}_U"
    if x_key not in data.files:
        raise KeyError(f"Key not found in faulty NPZ: {x_key}")
    if u_key not in data.files:
        raise KeyError(f"Key not found in faulty NPZ: {u_key}")

    x_full = data[x_key].astype(np.float32)
    u = data[u_key].astype(np.float32)
    if x_full.ndim != 2 or u.ndim != 2:
        raise ValueError(f"Fault {fault_id}, sim {sim_id} arrays must be rank-2.")
    if x_full.shape[0] != u.shape[0]:
        raise ValueError(f"Fault {fault_id}, sim {sim_id} X/U time dimensions do not match.")
    if x_full.shape[1] < n_x:
        raise ValueError(f"Fault {fault_id}, sim {sim_id} has {x_full.shape[1]} X vars; expected at least {n_x}.")
    if u.shape[1] != n_u:
        raise ValueError(f"Fault {fault_id}, sim {sim_id} has {u.shape[1]} U vars; expected {n_u}.")

    if not np.isfinite(x_full).all() or not np.isfinite(u).all():
        raise ValueError(f"Fault {fault_id}, sim {sim_id} contains NaN/Inf.")
    return x_full, u


def build_model(cfg: TrainConfig, adjacency: np.ndarray, n_x: int, n_u: int, device: torch.device) -> nn.Module:
    model = KGSTGMForecaster(
        adjacency=torch.from_numpy(adjacency),
        hidden_dim=cfg.hidden_dim,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        d_state=cfg.d_state,
        expand=cfg.expand,
        conv_kernel=cfg.conv_kernel,
        st_layers=cfg.st_layers,
        history=cfg.history,
        horizon=cfg.horizon,
        dropout=cfg.dropout,
        n_x=n_x,
        n_u=n_u,
    ).to(device)
    return model


def inverse_x(arr: np.ndarray, scaler: FeatureScaler) -> np.ndarray:
    return arr * scaler.std.reshape(1, -1) + scaler.mean.reshape(1, -1)


def inverse_u(arr: np.ndarray, scaler: FeatureScaler) -> np.ndarray:
    return arr * scaler.std.reshape(1, -1) + scaler.mean.reshape(1, -1)


def inverse_x_std(arr: np.ndarray, scaler: FeatureScaler) -> np.ndarray:
    return arr * scaler.std.reshape(1, -1)


@torch.no_grad()
def run_single_window(
    model: nn.Module,
    x: np.ndarray,
    u: np.ndarray,
    run_id: int,
    end_time: int,
    history: int,
    horizon: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, np.ndarray]:
    h0, h1 = end_time - history + 1, end_time + 1
    f0, f1 = end_time + 1, end_time + 1 + horizon
    if h0 < 0 or f1 > x.shape[1]:
        raise ValueError(
            f"Invalid plot window: run={run_id}, end_time={end_time}, "
            f"history={history}, horizon={horizon}, time_len={x.shape[1]}."
        )
    x_hist = torch.from_numpy(x[run_id, h0:h1]).unsqueeze(0).to(device)
    u_hist = torch.from_numpy(u[run_id, h0:h1]).unsqueeze(0).to(device)
    model.eval()
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        outputs = model(x_hist, u_hist, horizon)
    sample = {
        "x_hist": x[run_id, h0:h1].copy(),
        "u_hist": u[run_id, h0:h1].copy(),
        "x_true": x[run_id, f0:f1].copy(),
        "x_pred": outputs["x_pred"].squeeze(0).float().cpu().numpy(),
        "hist_times": np.arange(h0, h1),
        "future_times": np.arange(f0, f1),
    }
    if "x_std" in outputs:
        sample["x_std"] = outputs["x_std"].squeeze(0).float().cpu().numpy()
    return sample


@torch.no_grad()
def block_forecast_run(
    model: nn.Module,
    x_run: np.ndarray,
    u_run: np.ndarray,
    history: int,
    horizon: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    origins = np.arange(history - 1, x_run.shape[0] - horizon, stride, dtype=np.int32)
    if origins.size == 0:
        raise ValueError("No block forecast origins; check history/horizon against test length.")

    model.eval()
    pred_blocks: List[np.ndarray] = []
    std_blocks: List[np.ndarray] = []
    for start in range(0, origins.size, batch_size):
        batch_origins = origins[start:start + batch_size]
        x_hist = np.stack(
            [x_run[k - history + 1:k + 1] for k in batch_origins.tolist()],
            axis=0,
        ).astype(np.float32)
        u_hist = np.stack(
            [u_run[k - history + 1:k + 1] for k in batch_origins.tolist()],
            axis=0,
        ).astype(np.float32)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(
                torch.from_numpy(x_hist).to(device),
                torch.from_numpy(u_hist).to(device),
                horizon,
            )
        pred_blocks.append(outputs["x_pred"].float().cpu().numpy())
        if "x_std" in outputs:
            std_blocks.append(outputs["x_std"].float().cpu().numpy())
    std_np = np.concatenate(std_blocks, axis=0) if std_blocks else None
    return origins, np.concatenate(pred_blocks, axis=0), std_np


def stitch_block_forecast(pred_blocks: np.ndarray, origins: np.ndarray, series_len: int) -> Tuple[np.ndarray, np.ndarray]:
    accum = np.zeros((series_len, pred_blocks.shape[-1]), dtype=np.float32)
    counts = np.zeros((series_len, 1), dtype=np.float32)
    horizon = pred_blocks.shape[1]
    for block_idx, origin in enumerate(origins.tolist()):
        start = origin + 1
        end = start + horizon
        accum[start:end] += pred_blocks[block_idx]
        counts[start:end] += 1.0
    forecast_mask = counts[:, 0] > 0
    stitched = accum / np.maximum(counts, 1.0)
    return stitched, forecast_mask


def apply_normal_range_overrides(stats: Dict[str, np.ndarray], n_x: int) -> None:
    if REACTOR_TEMP_X_INDEX >= n_x:
        return
    stats["x_target"][REACTOR_TEMP_X_INDEX] = np.float32(REACTOR_TEMP_NORMAL_MU)
    stats["x_scale"][REACTOR_TEMP_X_INDEX] = np.float32(
        (REACTOR_TEMP_NORMAL_HIGH - REACTOR_TEMP_NORMAL_LOW) / 6.0
    )
    stats["x_low"][REACTOR_TEMP_X_INDEX] = np.float32(REACTOR_TEMP_NORMAL_LOW)
    stats["x_high"][REACTOR_TEMP_X_INDEX] = np.float32(REACTOR_TEMP_NORMAL_HIGH)


def save_normal_operating_stats(
    path: str,
    stats: Dict[str, np.ndarray],
    x_names: List[str],
    u_names: List[str],
    normal_data_npz: str,
    normal_stats_stride: int,
) -> None:
    """Persist the exact normal-operation targets and ranges used by testing/risk code."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        x_target=stats["x_target"].astype(np.float32),
        x_scale=stats["x_scale"].astype(np.float32),
        x_low=stats["x_low"].astype(np.float32),
        x_high=stats["x_high"].astype(np.float32),
        u_target=stats["u_target"].astype(np.float32),
        u_scale=stats["u_scale"].astype(np.float32),
        u_low=stats["u_low"].astype(np.float32),
        u_high=stats["u_high"].astype(np.float32),
        residual_mean=stats["residual_mean"].astype(np.float32),
        residual_std=stats["residual_std"].astype(np.float32),
        x_reference=stats["x_reference"].astype(np.float32),
        x_names=np.asarray(x_names, dtype=object),
        u_names=np.asarray(u_names, dtype=object),
        source_normal_data_npz=np.asarray(normal_data_npz, dtype=object),
        normal_stats_stride=np.asarray(int(normal_stats_stride), dtype=np.int32),
    )


def abnormality_score_from_3sigma_distance(distance: np.ndarray) -> np.ndarray:
    """Convert z-distance to an unbounded abnormality score."""
    distance = np.maximum(distance.astype(np.float32), 0.0)
    return (distance / 3.0).astype(np.float32)





def plot_full_run_forecast(
    true_x: np.ndarray,
    stitched_pred: np.ndarray,
    forecast_mask: np.ndarray,
    x_names: List[str],
    out_path: str,
    title: str,
    stitched_std: Optional[np.ndarray] = None,
    fault_time: Optional[int] = None,
    true_u: Optional[np.ndarray] = None,
    u_names: Optional[List[str]] = None,
    u_indices: Optional[List[int]] = None,
    highlight_x_indices: Optional[List[int]] = None,
    source_x_full: Optional[np.ndarray] = None,
    source_x_names: Optional[List[str]] = None,
    source_x_indices: Optional[List[int]] = None,
    normal_x_target: Optional[np.ndarray] = None,
    normal_x_std: Optional[np.ndarray] = None,
    normal_x_low: Optional[np.ndarray] = None,
    normal_x_high: Optional[np.ndarray] = None,
    normal_u_target: Optional[np.ndarray] = None,
    normal_u_std: Optional[np.ndarray] = None,
) -> None:
    # Parameters true_u, u_names, u_indices, highlight_x_indices, source_x_* are
    # retained for call-site compatibility but are intentionally unused here.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MAX_SAMPLES = 500
    true_x = true_x[:MAX_SAMPLES]
    stitched_pred = stitched_pred[:MAX_SAMPLES]
    forecast_mask = forecast_mask[:MAX_SAMPLES]
    if stitched_std is not None:
        stitched_std = stitched_std[:MAX_SAMPLES]

    time_idx = np.arange(true_x.shape[0])
    pred_idx = time_idx[forecast_mask]
    n_x = true_x.shape[1]
    n_cols = 4
    n_rows = int(np.ceil(n_x / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 5.0 * n_rows))
    axes = np.asarray(axes).ravel()

    for i in range(n_x):
        ax = axes[i]
        label = x_names[i] if i < len(x_names) else f"x{i + 1}"

        if normal_x_target is not None and normal_x_low is not None and normal_x_high is not None and i < len(normal_x_target):
            ax.axhspan(
                float(normal_x_low[i]), float(normal_x_high[i]),
                color="#10b981", alpha=0.10, zorder=0,
                label="Normal range" if i == 0 else None,
            )
            ax.axhline(
                float(normal_x_target[i]),
                color="#047857", lw=1.8, alpha=0.95, linestyle=":", zorder=1,
                label="Normal target" if i == 0 else None,
            )

        ax.plot(time_idx, true_x[:, i], color="black", lw=1.0, alpha=0.82, zorder=2,
            label="True" if i == 0 else None)
        ax.plot(pred_idx, stitched_pred[forecast_mask, i], color="#7e22ce", lw=1.8,
            alpha=0.98, zorder=4, label="Forecast" if i == 0 else None)

        if stitched_std is not None:
            mean_v = stitched_pred[forecast_mask, i]
            std_v = stitched_std[forecast_mask, i]
            ax.fill_between(pred_idx, mean_v - 1.96 * std_v, mean_v + 1.96 * std_v,
                            color="#a855f7", alpha=0.14, linewidth=0, zorder=3,
                            label="95% CI" if i == 0 else None)

        if fault_time is not None:
            ax.axvline(fault_time, color="#d62728", linestyle=":", lw=1.8, alpha=0.95,
                       label="Fault introduced" if i == 0 else None, zorder=5)

        # Truncate title to max 50 characters to avoid multi-line wrapping
        title_text = label if len(label) <= 50 else label[:47] + "..."
        ax.set_title(title_text, fontsize=14, fontweight="bold")
        ax.set_xlabel("Time Sample", fontsize=14)
        ax.set_xlim(0, MAX_SAMPLES)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)

    # Legend in the first empty cell of the last row
    empty_start = n_x
    n_total = n_rows * n_cols
    handles_by_label: Dict[str, object] = {}
    for ax in axes[:n_x]:
        hs, ls = ax.get_legend_handles_labels()
        for h, lbl in zip(hs, ls):
            handles_by_label.setdefault(lbl, h)

    if empty_start < n_total:
        legend_ax = axes[empty_start]
        legend_ax.axis("off")
        leg = legend_ax.legend(
            list(handles_by_label.values()),
            list(handles_by_label.keys()),
            loc="center",
            fontsize=22,
            frameon=True,
            title="Legend",
            title_fontsize=22,
            bbox_to_anchor=(0.5, 0.5),
            borderpad=1.0,
            labelspacing=0.8,
            handlelength=2.8,
            handletextpad=0.9,
        )
        if leg is not None:
            try:
                leg.get_title().set_fontweight("bold")
            except Exception:
                pass
        for ax in axes[empty_start + 1:]:
            ax.axis("off")
    else:
        for ax in axes[empty_start:]:
            ax.axis("off")
        if n_x > 0:
            axes[0].legend(list(handles_by_label.values()), list(handles_by_label.keys()),
                           loc="best", fontsize=22, borderpad=1.0, labelspacing=0.8,
                           handlelength=2.8, handletextpad=0.9)
    fig.tight_layout(pad=1.5, h_pad=1.5, w_pad=1.5)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_control_variables(
    true_u: np.ndarray,
    u_names: List[str],
    out_path: str,
    title: str,
    fault_time: Optional[int] = None,
    source_u_indices: Optional[List[int]] = None,
    normal_u_target: Optional[np.ndarray] = None,
    normal_u_low: Optional[np.ndarray] = None,
    normal_u_high: Optional[np.ndarray] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MAX_SAMPLES = 500
    true_u = true_u[:MAX_SAMPLES]
    time_idx = np.arange(true_u.shape[0])
    n_u = true_u.shape[1]
    # Disable highlighting of control variables in plots (keep parameter for API compatibility)
    highlight_u = set()
    n_cols = 4
    n_rows = int(np.ceil(n_u / n_cols))
    # Compute a more rectangular (landscape) figure size so plots are not square.
    per_col_w = 7.5
    max_h = 18.5
    fig_w = max(22.0, n_cols * per_col_w)
    fig_h = min(5.0 * n_rows, max_h)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 5.0 * n_rows))
    axes = np.asarray(axes).ravel()

    for i in range(n_u):
        ax = axes[i]
        label = u_names[i] if i < len(u_names) else f"u{i + 1}"
        is_source = i in highlight_u
        ax.plot(time_idx, true_u[:, i], color="#8c2d04", lw=1.5, alpha=0.95, zorder=3,
                label="Control input" if i == 0 else None)
        if (
            normal_u_target is not None
            and normal_u_low is not None
            and normal_u_high is not None
            and i < len(normal_u_target)
        ):
            ax.axhline(float(normal_u_target[i]), color="#047857", lw=1.8, alpha=0.95,
                       linestyle=":", zorder=1, label="Normal target" if i == 0 else None)
            ax.axhspan(float(normal_u_low[i]), float(normal_u_high[i]),
                       color="#10b981", alpha=0.08, zorder=0,
                       label="Normal range" if i == 0 else None)
        if fault_time is not None:
            ax.axvline(fault_time, color="#d62728", linestyle=":", lw=1.8, alpha=0.95, zorder=5,
                       label="Fault introduced" if i == 0 else None)
        # Truncate U variable title to max 50 characters to avoid multi-line wrapping
        # Title-case the control variable label so each word starts with a capital.
        label_title = label.title() if isinstance(label, str) else label
        raw_title = f"u{i + 1}: {label_title}  [fault source]" if is_source else f"u{i + 1}: {label_title}"
        u_title = raw_title if len(raw_title) <= 50 else raw_title[:47] + "..."
        ax.set_title(u_title, fontsize=14, fontweight="bold", color="#b91c1c" if is_source else "black")
        if is_source:
            ax.set_facecolor("#fff7ed")
            for spine in ax.spines.values():
                spine.set_color("#b91c1c")
                spine.set_linewidth(2.2)
        ax.set_xlabel("Time Sample", fontsize=14)
        ax.set_xlim(0, MAX_SAMPLES)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)

    # Legend in the first empty cell of the last row
    empty_start = n_u
    n_total = n_rows * n_cols
    handles_by_label: Dict[str, object] = {}
    for ax in axes[:n_u]:
        hs, ls = ax.get_legend_handles_labels()
        for h, lbl in zip(hs, ls):
            handles_by_label.setdefault(lbl, h)

    if empty_start < n_total:
        legend_ax = axes[empty_start]
        legend_ax.axis("off")
        leg = legend_ax.legend(
            list(handles_by_label.values()),
            list(handles_by_label.keys()),
            loc="center",
            fontsize=22,
            frameon=True,
            title="Legend",
            title_fontsize=22,
            bbox_to_anchor=(0.5, 0.5),
            borderpad=1.0,
            labelspacing=0.8,
            handlelength=2.8,
            handletextpad=0.9,
        )
        if leg is not None:
            try:
                leg.get_title().set_fontweight("bold")
            except Exception:
                pass
        for ax in axes[empty_start + 1:]:
            ax.axis("off")
    else:
        for ax in axes[empty_start:]:
            ax.axis("off")
        if n_u > 0:
            axes[0].legend(list(handles_by_label.values()), list(handles_by_label.keys()),
                           loc="best", fontsize=22, borderpad=1.0, labelspacing=0.8,
                           handlelength=2.8, handletextpad=0.9)

    fig.tight_layout(pad=1.5, h_pad=1.5, w_pad=1.5)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_presentation_layout() -> Dict[str, Tuple[float, float]]:
    pos: Dict[str, Tuple[float, float]] = {}

    for i, n in enumerate(["x1", "x2", "x3", "x4"]):
        pos[n] = (0.0, 5.4 - i * 2.2)
    for i, n in enumerate(["u1", "u2", "u3", "u4"]):
        pos[n] = (-2.4, 5.4 - i * 2.2)

    pos["x6"] = (4.0, 5.8)
    pos["x7"] = (3.4, 3.5)
    pos["x8"] = (4.6, 2.1)
    pos["x9"] = (4.0, 0.0)
    pos["x21"] = (4.0, -2.5)
    pos["u10"] = (1.9, -2.5)

    pos["u11"] = (7.5, 6.5)
    pos["x22"] = (7.5, 5.0)

    pos["u5"] = (7.5, 3.5)
    pos["x5"] = (7.5, 1.7)
    pos["x20"] = (7.5, 0.0)

    pos["x11"] = (11.5, 5.0)
    pos["x12"] = (10.9, 3.5)
    pos["x13"] = (11.6, 1.7)
    pos["x14"] = (10.9, -0.5)
    pos["x10"] = (11.6, -2.5)
    pos["u7"] = (9.7, -0.5)
    pos["u6"] = (9.7, -2.5)

    pos["x15"] = (15.2, 1.5)
    pos["x16"] = (15.2, -1.2)
    pos["x17"] = (15.2, 3.9)
    pos["x18"] = (16.5, -2.5)
    pos["x19"] = (15.2, -2.5)
    pos["u8"] = (13.8, 3.9)
    pos["u9"] = (13.8, -2.5)
    return pos


def measurement_marker(node_name: str) -> str:
    if node_name.startswith("u"):
        return "s"
    for group in MEASUREMENT_GROUPS.values():
        if node_name in group["nodes"]:
            return str(group["mpl_marker"])
    return "o"


def marker_size_scale(node_name: str) -> float:
    if node_name.startswith("u"):
        return 1.0
    for group in MEASUREMENT_GROUPS.values():
        if node_name in group["nodes"]:
            return float(group["size_scale"])
    return 1.0


def node_label_position(node_name: str, x_pos: float, y_pos: float) -> Tuple[float, float]:
    if measurement_marker(node_name) == "^":
        return x_pos, y_pos - 0.12
    return x_pos, y_pos

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained KG-STGM model on faulty TE test data.")
    parser.add_argument("--out_dir", type=str, default="kg_stgm_outputs")
    parser.add_argument("--data_npz", type=str, default="dataverse_files/TEP_Faulty_Testing.npz")
    parser.add_argument("--normal_data_npz", type=str, default="dataverse_files/TEP_FaultFree_Testing.npz")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sim_id", type=int, default=1, help="1-based faulty simulation run to test for every fault type.")
    parser.add_argument("--fault_ids", type=str, default="1-15", help="Comma/range list of fault IDs to test, e.g. '1-15' or '1,3,5'.")
    parser.add_argument("--plot_end_time", type=int, default=FAULT_INTRO_TIME)
    parser.add_argument("--fault_intro_time", type=int, default=FAULT_INTRO_TIME)
    # prognosis graph options removed; prognosis graphs are not produced in this script
    parser.add_argument("--test_stride", type=int, default=10, help="Forecast stride for full-run plot. Default 10 gives non-overlapping blocks for horizon=10.")
    parser.add_argument("--normal_stats_stride", type=int, default=0, help="Normal residual stats stride. 0 means stride=horizon.")
    parser.add_argument("--normal_stats_max_windows", type=int, default=4000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config or os.path.join(args.out_dir, "config.json")
    output_dir = args.output_dir or os.path.join(args.out_dir, "testing_faulty")
    os.makedirs(output_dir, exist_ok=True)

    cfg = load_train_config(config_path)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    cfg.device = args.device
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    use_amp = bool(cfg.amp) and device.type != "cpu"
    if device.type == "mps":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float16 if cfg.amp_dtype == "float16" else torch.bfloat16

    scaler_path = os.path.join(args.out_dir, "scaler_stats.npz")
    adjacency_path = os.path.join(args.out_dir, "adjacency.npy")
    checkpoint_path = resolve_checkpoint(args.out_dir, args.checkpoint)

    x_scaler, u_scaler, _ = load_saved_scalers(scaler_path)
    n_x = int(x_scaler.mean.shape[0])
    n_u = int(u_scaler.mean.shape[0])
    adjacency = load_adjacency(adjacency_path, n_nodes=n_x + n_u)

    model = build_model(cfg, adjacency, n_x, n_u, device)
    model.load_state_dict(load_state_dict_file(checkpoint_path, device))

    data = np.load(args.data_npz, allow_pickle=True)
    available_fault_ids, x_names_raw, u_names_raw, fault_name_by_id = load_faulty_metadata(data)
    requested_fault_ids = parse_fault_ids(args.fault_ids)
    # Prognosis graph origins removed; prognosis plotting disabled in this script.
    missing_fault_ids = [fault_id for fault_id in requested_fault_ids if fault_id not in available_fault_ids]
    if missing_fault_ids:
        raise ValueError(f"Requested fault IDs not present in {args.data_npz}: {missing_fault_ids}")

    test_stride = cfg.horizon if int(args.test_stride) <= 0 else max(1, int(args.test_stride))
    forecast_mode = "non-overlapping" if test_stride == cfg.horizon else "sliding-window"
    normal_stats_stride = cfg.horizon if int(args.normal_stats_stride) <= 0 else max(1, int(args.normal_stats_stride))
    # Try to load previously computed normal operating stats, if available.
    prognosis_stats = None
    normal_stats_path = os.path.join(args.out_dir, "normal_operating_stats.npz")
    if os.path.exists(normal_stats_path):
        try:
            npz = np.load(normal_stats_path, allow_pickle=True)
            prognosis_stats = {
                "x_target": npz["x_target"].astype(np.float32),
                "x_scale": npz["x_scale"].astype(np.float32),
                "x_low": npz["x_low"].astype(np.float32),
                "x_high": npz["x_high"].astype(np.float32),
                "u_target": npz["u_target"].astype(np.float32),
                "u_scale": npz["u_scale"].astype(np.float32),
                "u_low": npz["u_low"].astype(np.float32),
                "u_high": npz["u_high"].astype(np.float32),
                "residual_mean": npz["residual_mean"].astype(np.float32),
                "residual_std": npz["residual_std"].astype(np.float32),
            }
            apply_normal_range_overrides(prognosis_stats, n_x)
            print(f"Loaded normal operating stats: {normal_stats_path}")
        except Exception:
            prognosis_stats = None
            print(f"Failed to load normal operating stats from {normal_stats_path}; continuing without ranges.")
    else:
        print(f"No normal operating stats found at {normal_stats_path}; continuing without ranges.")
    plot_names = [
        f"x{i + 1}: {_X22_DISPLAY_NAMES[i]}" if i < len(_X22_DISPLAY_NAMES)
        else f"x{i + 1}: {x_names_raw[i].replace('_', ' ')}"
        for i in range(n_x)
    ]
    # Use raw U names; ensure underscores replaced for readability.
    u_display_names = [u_names_raw[i].replace("_", " ") for i in range(len(u_names_raw))]

    print(
        f"Testing KG-STGM on faulty data: {len(requested_fault_ids)} fault types, "
        f"sim_id={args.sim_id}, {forecast_mode} stride={test_stride}"
    )
    print(f"Normal stats path: {normal_stats_path}")
    print(f"Outputs will be saved under: {output_dir}/fault<N>/")

    for fault_id in requested_fault_ids:
        x_full_raw, u_raw = load_faulty_run(data, fault_id, args.sim_id, n_x=n_x, n_u=n_u)
        x_raw = x_full_raw[:, :n_x]
        time_len = x_raw.shape[0]
        min_end = cfg.history - 1
        max_end = time_len - cfg.horizon - 1
        plot_end_time = int(np.clip(args.plot_end_time, min_end, max_end))

        x = x_scaler.transform(x_raw[None, :, :])
        u = u_scaler.transform(u_raw[None, :, :])

        fault_dir = os.path.join(output_dir, f"fault{fault_id}")
        os.makedirs(fault_dir, exist_ok=True)
        tag = f"fault{fault_id}_sim{args.sim_id}_t{plot_end_time}"
        fault_label = fault_name_by_id.get(fault_id, f"Fault {fault_id}")
        source = FAULT_TABLE_SOURCE.get(fault_id, {"x": [], "u": [], "label": fault_label})
        source_x_indices = [int(idx) for idx in source["x"]]
        highlight_x_indices = [idx for idx in source_x_indices if idx < n_x]
        extra_source_x_indices = [idx for idx in source_x_indices if idx >= n_x]
        highlight_u_indices = [int(idx) for idx in source["u"]]
        highlight_graph_nodes = [f"x{idx + 1}" for idx in highlight_x_indices]
        graph_source_label = ", ".join(highlight_graph_nodes) if highlight_graph_nodes else "forecasted X nodes only"

        origins, pred_blocks, std_blocks = block_forecast_run(
            model=model,
            x_run=x[0],
            u_run=u[0],
            history=cfg.history,
            horizon=cfg.horizon,
            stride=test_stride,
            batch_size=cfg.batch_size,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        stitched_pred, forecast_mask = stitch_block_forecast(pred_blocks, origins, time_len)
        stitched_pred_raw = inverse_x(stitched_pred, x_scaler)
        stitched_std_raw = None
        if std_blocks is not None:
            stitched_std, _ = stitch_block_forecast(std_blocks, origins, time_len)
            stitched_std_raw = inverse_x_std(stitched_std, x_scaler)
        stride_label = f"{forecast_mode} stride={test_stride}"
        forecast_path = os.path.join(fault_dir, f"x_forecast_{tag}.png")
        plot_full_run_forecast(
            true_x=x_raw,
            stitched_pred=stitched_pred_raw,
            forecast_mask=forecast_mask,
            x_names=plot_names,
            out_path=forecast_path,
            title=(
                f"KG-STGM X {forecast_mode} forecast | fault {fault_id}: {fault_label} | "
                f"source: {source['label']} | sim {args.sim_id} | {stride_label}"
            ),
            stitched_std=stitched_std_raw,
            fault_time=args.fault_intro_time,
            true_u=u_raw,
            u_names=u_names_raw,
            u_indices=highlight_u_indices,
            highlight_x_indices=highlight_x_indices,
            source_x_full=x_full_raw,
            source_x_names=x_names_raw,
            source_x_indices=extra_source_x_indices,
            normal_x_target=(prognosis_stats["x_target"] if prognosis_stats is not None else None),
            normal_x_std=(prognosis_stats["x_scale"] if prognosis_stats is not None else None),
            normal_x_low=(prognosis_stats["x_low"] if prognosis_stats is not None else None),
            normal_x_high=(prognosis_stats["x_high"] if prognosis_stats is not None else None),
            normal_u_target=(prognosis_stats["u_target"] if prognosis_stats is not None else None),
            normal_u_std=(prognosis_stats["u_scale"] if prognosis_stats is not None else None),
        )
        controls_path = os.path.join(fault_dir, f"u_controls_{tag}.png")
        plot_control_variables(
            true_u=u_raw,
            u_names=u_display_names,
            out_path=controls_path,
            title=(
                f"TE manipulated variables | fault {fault_id}: {fault_label} | "
                f"source: {source['label']} | sim {args.sim_id}"
            ),
            fault_time=args.fault_intro_time,
            source_u_indices=highlight_u_indices,
            normal_u_target=(prognosis_stats["u_target"] if prognosis_stats is not None else None),
            normal_u_low=(prognosis_stats["u_low"] if prognosis_stats is not None else None),
            normal_u_high=(prognosis_stats["u_high"] if prognosis_stats is not None else None),
        )

        print(f"  [fault {fault_id:2d}] saved: {forecast_path}")
        print(f"  [fault {fault_id:2d}] saved: {controls_path}")


if __name__ == "__main__":
    main()
