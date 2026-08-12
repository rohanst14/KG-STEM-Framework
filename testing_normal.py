
"""
KG-STGM testing script.

This script is intentionally tied to KG-STGM train.py:
it reuses the same TE variable mapping, saved train-set scalers, KG adjacency,
KGSTGMForecaster class, checkpoint format, and forecast target convention.

"""

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
    RAW_X22_NAMES,
    TrainConfig,
    FeatureScaler,
    load_adjacency,
    load_state_dict_file,
    load_tep_arrays,
    resolve_device,
    set_seed,
)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_train_config(path: str) -> TrainConfig:
    raw = load_json(path)
    valid = {f.name for f in fields(TrainConfig)}
    return TrainConfig(**{k: v for k, v in raw.items() if k in valid})


def resolve_checkpoint(out_dir: str, checkpoint: str) -> str:
    if os.path.isabs(checkpoint):
        return checkpoint
    return os.path.join(out_dir, checkpoint)


def load_saved_scalers(path: str) -> Tuple[FeatureScaler, FeatureScaler, List[str]]:
    stats = np.load(path, allow_pickle=True)
    required = ["x_mean", "x_std", "u_mean", "u_std"]
    missing = [key for key in required if key not in stats.files]
    if missing:
        raise ValueError(f"{path} is missing scaler arrays: {missing}")
    x_scaler = FeatureScaler(stats["x_mean"], stats["x_std"])
    u_scaler = FeatureScaler(stats["u_mean"], stats["u_std"])
    x_names = [str(v) for v in stats["x_names"].tolist()] if "x_names" in stats.files else RAW_X22_NAMES
    return x_scaler, u_scaler, x_names


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


def inverse_x_std(arr: np.ndarray, scaler: FeatureScaler) -> np.ndarray:
    return arr * scaler.std.reshape(1, -1)


_X22_DISPLAY_NAMES: List[str] = [
    "A Feed Stream",
    "D Feed Stream",
    "E Feed Stream",
    "Total Fresh Feed to Stripper",
    "Recycle Flow into Reactor",
    "Reactor Feed Rate",
    "Reactor Pressure",
    "Reactor Level",
    "Reactor Temperature",
    "Purge Rate",
    "Separator Temperature",
    "Separator Level",
    "Separator Pressure",
    "Separator Underflow",
    "Stripper Level",
    "Stripper Pressure",
    "Stripper Underflow",
    "Stripper Temperature",
    "Stripper Steam Flow",
    "Compressor Work",
    "Reactor Cooling Water Outlet Temperature",
    "Condenser Cooling Water Outlet Temperature",
]

EVAL_SAMPLES = 1000


def plot_full_run_forecast(
    true_x: np.ndarray,
    stitched_pred: np.ndarray,
    forecast_mask: np.ndarray,
    x_names: List[str],
    out_path: str,
    stitched_std: Optional[np.ndarray] = None,
    normal_x_target: Optional[np.ndarray] = None,
    normal_x_low: Optional[np.ndarray] = None,
    normal_x_high: Optional[np.ndarray] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MAX_SAMPLES = EVAL_SAMPLES
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

        if (
            normal_x_target is not None
            and normal_x_low is not None
            and normal_x_high is not None
            and i < len(normal_x_target)
        ):
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

        title_text = label if len(label) <= 50 else textwrap.fill(label, width=50)
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
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout(pad=1.5, h_pad=1.5, w_pad=1.5)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def compute_forecast_metrics(
    true_x: np.ndarray,
    pred_x: np.ndarray,
    var_names: List[str],
    normal_x_low: np.ndarray,
    normal_x_high: np.ndarray,
) -> Dict:
    eps = 1e-8
    n_x = true_x.shape[1]
    per_var: Dict[str, Dict] = {}
    for i in range(n_x):
        y_true = true_x[:, i].astype(np.float64)
        y_pred = pred_x[:, i].astype(np.float64)
        residuals = y_pred - y_true

        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        mae = float(np.mean(np.abs(residuals)))
        mape = float(np.mean(np.abs(residuals) / np.maximum(np.abs(y_true), 1.0)) * 100.0)
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = float(1.0 - ss_res / max(ss_tot, eps))

        low = float(normal_x_low[i])
        high = float(normal_x_high[i])
        true_normal = ((y_true >= low) & (y_true <= high)).astype(int)
        pred_normal = ((y_pred >= low) & (y_pred <= high)).astype(int)
        tp = int(np.sum((pred_normal == 1) & (true_normal == 1)))
        tn = int(np.sum((pred_normal == 0) & (true_normal == 0)))
        fp = int(np.sum((pred_normal == 1) & (true_normal == 0)))
        fn = int(np.sum((pred_normal == 0) & (true_normal == 1)))
        total = max(tp + tn + fp + fn, 1)

        accuracy = float((tp + tn) / total)
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = float(2.0 * precision * recall / max(precision + recall, eps))

        vname = var_names[i] if i < len(var_names) else f"x{i + 1}"
        per_var[vname] = {
            "RMSE": round(rmse, 6),
            "MAE": round(mae, 6),
            "MAPE_%": round(mape, 4),
            "R2": round(r2, 6),
            "Accuracy": round(accuracy, 6),
            "Precision": round(precision, 6),
            "Recall": round(recall, 6),
            "F1": round(f1, 6),
        }

    metric_keys = ["RMSE", "MAE", "MAPE_%", "R2", "Accuracy", "Precision", "Recall", "F1"]
    aggregate = {
        m: round(float(np.mean([per_var[v][m] for v in per_var])), 6)
        for m in metric_keys
    }
    return {"per_variable": per_var, "aggregate": aggregate}


def _write_metrics_table(
    all_metrics: Dict,
    path: str,
    num_runs: int,
    var_names: List[str],
    start_run: int = 0,
) -> None:
    metric_keys = ["RMSE", "MAE", "MAPE_%", "R2", "Accuracy", "Precision", "Recall", "F1"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write(f"FORECAST PERFORMANCE METRICS — FAULT-FREE TEST RUNS (first {EVAL_SAMPLES} samples)\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"{'Run':<10}")
        for m in metric_keys:
            f.write(f"{m:>10}")
        f.write("\n" + "-" * (10 + 10 * len(metric_keys)) + "\n")
        for offset in range(num_runs):
            r = start_run + offset
            agg = all_metrics.get(f"run_{r}", {}).get("aggregate", {})
            f.write(f"{('Run ' + str(r)):<10}")
            for m in metric_keys:
                f.write(f"{agg.get(m, float('nan')):>10.4f}")
            f.write("\n")
        if "overall_aggregate" in all_metrics:
            f.write("-" * (10 + 10 * len(metric_keys)) + "\n")
            f.write(f"{'Overall':<10}")
            for m in metric_keys:
                f.write(f"{all_metrics['overall_aggregate'].get(m, float('nan')):>10.4f}")
            f.write("\n")

        for offset in range(num_runs):
            r = start_run + offset
            per_var = all_metrics.get(f"run_{r}", {}).get("per_variable", {})
            f.write(f"\n\n--- Run {r}: Per-Variable Metrics ---\n")
            f.write(f"{'Variable':<50}")
            for m in metric_keys:
                f.write(f"{m:>10}")
            f.write("\n" + "-" * (50 + 10 * len(metric_keys)) + "\n")
            for vname in sorted(per_var.keys()):
                vm = per_var[vname]
                f.write(f"{vname:<50}")
                for m in metric_keys:
                    f.write(f"{vm.get(m, float('nan')):>10.4f}")
                f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained KG-STGM model on TE fault-free test data.")
    parser.add_argument("--out_dir", type=str, default="kg_stgm_outputs")
    parser.add_argument("--data_npz", type=str, default="dataverse_files/TEP_FaultFree_Testing.npz")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_plot_runs", type=int, default=5,
                        help="Number of fault-free simulation runs to plot and evaluate (default: 5).")
    parser.add_argument("--start_run", type=int, default=0,
                        help="Index of the first run to plot (default: 0).")
    parser.add_argument("--test_stride", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config or os.path.join(args.out_dir, "config.json")
    output_dir = args.output_dir or os.path.join(args.out_dir, "testing_faultfree")
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
    x_raw, u_raw, variable_mapping = load_tep_arrays(args.data_npz)
    n_runs, time_len, n_x = x_raw.shape
    n_u = u_raw.shape[-1]
    adjacency = load_adjacency(adjacency_path, n_nodes=n_x + n_u)

    x = x_scaler.transform(x_raw)
    u = u_scaler.transform(u_raw)

    model = build_model(cfg, adjacency, n_x, n_u, device)
    model.load_state_dict(load_state_dict_file(checkpoint_path, device))

    test_stride = max(1, int(args.test_stride))
    forecast_mode = "non-overlapping" if test_stride >= cfg.horizon else "sliding-window"

    # Full display names — no underscores, correct complete variable names
    raw_col_names = variable_mapping["raw_x_columns_used"]
    plot_names = [
        f"x{i + 1}: {_X22_DISPLAY_NAMES[i]}" if i < len(_X22_DISPLAY_NAMES)
        else f"x{i + 1}: {raw_col_names[i].replace('_', ' ')}"
        for i in range(n_x)
    ]

    # Compute normal operating band from the fault-free test data
    flat_x = x_raw.reshape(-1, n_x).astype(np.float64)
    normal_x_target = flat_x.mean(axis=0).astype(np.float32)
    normal_x_scale = flat_x.std(axis=0).astype(np.float32)
    normal_x_low = (normal_x_target - 3.0 * normal_x_scale).astype(np.float32)
    normal_x_high = (normal_x_target + 3.0 * normal_x_scale).astype(np.float32)
    # Physical override for reactor temperature
    _REACTOR_IDX = 8
    if n_x > _REACTOR_IDX:
        normal_x_target[_REACTOR_IDX] = np.float32(120.4)
        normal_x_low[_REACTOR_IDX] = np.float32(120.1)
        normal_x_high[_REACTOR_IDX] = np.float32(120.7)

    start_run = max(0, min(int(args.start_run), max(0, n_runs - 1)))
    actual_num_runs = min(int(args.num_plot_runs), max(0, n_runs - start_run))
    all_metrics: Dict[str, Dict] = {}

    print(
        f"Evaluating {actual_num_runs} fault-free runs starting at run {start_run} ({forecast_mode}, stride={test_stride})"
    )
    for run_idx in range(start_run, start_run + actual_num_runs):
        origins, pred_blocks, std_blocks = block_forecast_run(
            model=model,
            x_run=x[run_idx],
            u_run=u[run_idx],
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

        out_path = os.path.join(output_dir, f"x_forecast_run{run_idx}.png")
        plot_full_run_forecast(
            true_x=x_raw[run_idx],
            stitched_pred=stitched_pred_raw,
            forecast_mask=forecast_mask,
            x_names=plot_names,
            out_path=out_path,
            stitched_std=stitched_std_raw,
            normal_x_target=normal_x_target,
            normal_x_low=normal_x_low,
            normal_x_high=normal_x_high,
        )
        print(f"  [run {run_idx}] saved: {out_path}")

        # Metrics over forecast-masked region, capped at the plotted sample count.
        mask_eval = forecast_mask[:EVAL_SAMPLES]
        true_masked = x_raw[run_idx, :EVAL_SAMPLES][mask_eval]
        pred_masked = stitched_pred_raw[:EVAL_SAMPLES][mask_eval]
        if true_masked.shape[0] > 0:
            run_metrics = compute_forecast_metrics(
                true_x=true_masked,
                pred_x=pred_masked,
                var_names=[f"x{i + 1}" for i in range(n_x)],
                normal_x_low=normal_x_low,
                normal_x_high=normal_x_high,
            )
        else:
            run_metrics = {"per_variable": {}, "aggregate": {}}
        all_metrics[f"run_{run_idx}"] = run_metrics

    # Overall aggregate across all runs
    metric_keys = ["RMSE", "MAE", "MAPE_%", "R2", "Accuracy", "Precision", "Recall", "F1"]
    overall: Dict[str, float] = {}
    for m in metric_keys:
        vals = [
            all_metrics[f"run_{r}"]["aggregate"][m]
            for r in range(start_run, start_run + actual_num_runs)
            if all_metrics.get(f"run_{r}", {}).get("aggregate", {}).get(m) is not None
        ]
        if vals:
            overall[m] = round(float(np.mean(vals)), 6)
    all_metrics["overall_aggregate"] = overall

    metrics_json_path = os.path.join(output_dir, "forecast_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics JSON : {metrics_json_path}")

    metrics_txt_path = os.path.join(output_dir, "forecast_metrics.txt")
    _write_metrics_table(all_metrics, metrics_txt_path, actual_num_runs, plot_names, start_run=start_run)
    print(f"Saved metrics table: {metrics_txt_path}")


if __name__ == "__main__":
    main()
