"""
KG-STGM: Knowledge-Guided Spatiotemporal Graph Mamba for the Tennessee
Eastman Process (TEP).

This module trains a multistep probabilistic forecaster over a 33-node causal
knowledge graph composed of 22 XMEAS process variables and 11 XMV manipulated
variables. The architecture is adapted from MGCN (Mamba-Integrated
Spatiotemporal Graph Convolutional Network for Long-Term Traffic Forecasting).

Reference implementation:
    https://github.com/Vincent665/MGCN-for-Traffic-Prediction

Training objective
------------------
Inputs:
    X_hist: [batch, history, 22]
    U_hist: [batch, history, 11]

Outputs:
    X_pred:    [batch, horizon, 22]
    X_log_var: [batch, horizon, 22]
    X_std:     [batch, horizon, 22]

The model treats faulty and fault-free trajectories as unlabeled forecasting
sequences. Fault identifiers are used only to locate arrays in the faulty-data
NPZ file; they are not passed to the model or used in the training loss.

The graph convention is A[dst, src] = 1 for a directed edge src -> dst.
Control-variable nodes (u1..u11) are historical-context source nodes and are
not forecast targets.

"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# TEP Variable Mapping Constants
# =============================================================================
LOGGER = logging.getLogger("kg_stgm_train")

X_NODE_NAMES = [f"x{i}" for i in range(1, 23)]
U_NODE_NAMES = [f"u{i}" for i in range(1, 12)]
NODE_NAMES = X_NODE_NAMES + U_NODE_NAMES

RAW_X22_NAMES = [
    "A_feed_stream", "D_feed_stream", "E_feed_stream",
    "Total_fresh_feed_stripper", "Recycle_flow_into_rxtr",
    "Reactor_feed_rate", "Reactor_pressure", "Reactor_level",
    "Reactor_temp", "Purge_rate", "Separator_temp", "Separator_level",
    "Separator_pressure", "Separator_underflow", "Stripper_level",
    "Stripper_pressure", "Stripper_underflow", "Stripper_temperature",
    "Stripper_steam_flow", "Compressor_work",
    "Reactor_cooling_water_outlet_temp",
    "Condenser_cooling_water_outlet_temp",
]
RAW_U11_NAMES = [
    "D_feed_flow_valve", "E_feed_flow_valve", "A_feed_flow_valve",
    "Total_feed_flow_stripper_valve", "Compressor_recycle_valve",
    "Purge_valve", "Separator_pot_liquid_flow_valve",
    "Stripper_liquid_product_flow_valve", "Stripper_steam_valve",
    "Reactor_cooling_water_flow_valve", "Condenser_cooling_water_flow_valve",
]
RAW_X22_COLS = [f"xmeas_{i}" for i in range(1, 23)]
RAW_U11_COLS = [f"xmv_{i}" for i in range(1, 12)]

# =============================================================================
# Configuration
# =============================================================================
@dataclass
class TrainConfig:
    data_npz: str = "dataverse_files/TEP_FaultFree_Training.npz"
    faulty_data_npz: str = "dataverse_files/TEP_Faulty_Training.npz"
    include_faulty_training: bool = True
    adjacency_npy: str = "final_causal_graph_outputs/tep_causal_adjacency.npy"
    out_dir: str = "kg_stgm_outputs"

    # TEP: 3-min sampling interval. history=20 (60 min), horizon=10 (30 min).
    # Non-overlapping forecast blocks: stride matches the default horizon.
    history: int = 20
    horizon: int = 10
    stride: int = 10

    # Model architecture (MGCN-style)
    d_model: int = 256          # Keep token width moderate for generalization
    d_ff: int = 1024            # Standard 4*d_model FFN; capacity is added in SSM instead
    hidden_dim: int = 128       # KGGraphLayer internal hidden dim
    # Mamba SSM hyper-params
    d_state: int = 32           # Balanced SSM memory for faster training
    expand: int = 2             # Standard Mamba expansion; faster than expand=3
    conv_kernel: int = 5        # Slightly wider local temporal mixing
    st_layers: int = 3          # Number of EncoderLayer blocks (MGCN uses 1)
    dropout: float = 0.0

    # Training
    batch_size: int = 128
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4  # light L2 regularization for generalization
    grad_clip: float = 1.0
    patience: int = 12
    val_ratio: float = 0.15
    # MGCN step-decay schedule
    lr_decay_rate: float = 0.5
    lr_decay_step: int = 20
    input_jitter_std: float = 0.01

    seed: int = 42
    num_workers: int = -1
    amp: bool = True
    amp_dtype: str = "bfloat16"
    compile_model: bool = False
    device: str = "auto"


def validate_config(cfg: TrainConfig) -> None:
    """Validate configuration values and required input files before training."""
    positive_int_fields = {
        "history": cfg.history,
        "horizon": cfg.horizon,
        "stride": cfg.stride,
        "d_model": cfg.d_model,
        "d_ff": cfg.d_ff,
        "hidden_dim": cfg.hidden_dim,
        "d_state": cfg.d_state,
        "expand": cfg.expand,
        "conv_kernel": cfg.conv_kernel,
        "st_layers": cfg.st_layers,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "patience": cfg.patience,
        "lr_decay_step": cfg.lr_decay_step,
    }
    for name, value in positive_int_fields.items():
        if value < 1:
            raise ValueError(f"{name} must be >= 1; got {value}.")

    if cfg.history < 2:
        raise ValueError(f"history must be >= 2; got {cfg.history}.")
    if not 0.0 < cfg.val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1); got {cfg.val_ratio}.")
    if not 0.0 <= cfg.dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1); got {cfg.dropout}.")
    if cfg.lr <= 0.0:
        raise ValueError(f"lr must be > 0; got {cfg.lr}.")
    if cfg.weight_decay < 0.0:
        raise ValueError(f"weight_decay must be >= 0; got {cfg.weight_decay}.")
    if cfg.grad_clip < 0.0:
        raise ValueError(f"grad_clip must be >= 0; got {cfg.grad_clip}.")
    if not 0.0 < cfg.lr_decay_rate <= 1.0:
        raise ValueError(
            f"lr_decay_rate must be in (0, 1]; got {cfg.lr_decay_rate}."
        )
    if cfg.input_jitter_std < 0.0:
        raise ValueError(
            f"input_jitter_std must be >= 0; got {cfg.input_jitter_std}."
        )
    if cfg.num_workers < -1:
        raise ValueError(f"num_workers must be -1 or >= 0; got {cfg.num_workers}.")

    required_paths = [Path(cfg.data_npz), Path(cfg.adjacency_npy)]
    if cfg.include_faulty_training:
        required_paths.append(Path(cfg.faulty_data_npz))
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(f"Required input file(s) not found:\n  - {formatted}")


# =============================================================================
# General Utilities
# =============================================================================
def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(out_dir: str) -> None:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    LOGGER.addHandler(sh)
    fh = logging.FileHandler(output_dir / "train.log", mode="w")
    fh.setFormatter(fmt)
    LOGGER.addHandler(fh)


def save_json(obj: Dict, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def load_state_dict_file(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if state and all(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    return state


def unwrap_compiled_model(model: nn.Module) -> nn.Module:
    """Return the original module when torch.compile wraps it."""
    return getattr(model, "_orig_mod", model)


def save_model_state(model: nn.Module, path: str) -> None:
    torch.save(unwrap_compiled_model(model).state_dict(), path)


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable.")
    return device


def choose_num_workers(num_workers: int, device_type: str) -> int:
    if num_workers >= 0:
        return num_workers
    cpu_count = os.cpu_count() or 1
    return min(cpu_count, 8) if device_type == "cuda" else 0


def log_device_details(device: torch.device) -> None:
    LOGGER.info("Resolved device: %s", device)
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        LOGGER.info(
            "CUDA device %d: %s | memory=%.1f GB",
            idx, props.name, props.total_memory / (1024 ** 3),
        )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


# =============================================================================
# Feature Scaler
# =============================================================================
class FeatureScaler:
    """
    Stores per-feature mean and standard deviation for z-score normalization.

    The saved keys remain x_mean/x_std and u_mean/u_std because inference
    scripts already use that checkpoint format.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), np.finfo(np.float32).eps)

    @classmethod
    def fit(cls, arr: np.ndarray) -> "FeatureScaler":
        flat = arr.reshape(-1, arr.shape[-1])
        return cls(flat.mean(axis=0), flat.std(axis=0))

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return ((arr - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        """Undo transform and recover original-scale values."""
        return (arr * self.std + self.mean).astype(np.float32)


# Backward-compatible alias used by inference/evaluation utilities
StandardScaler = FeatureScaler

# =============================================================================
# Metric Functions
# =============================================================================
def MAE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def MSE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


def RMSE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(MSE(pred, true)))


def MAPE(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs((pred - true) / (np.abs(true) + 1e-8))))


def metric(
    pred: np.ndarray,
    true: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Return (MAE, MSE, RMSE, MAPE)."""
    return MAE(pred, true), MSE(pred, true), RMSE(pred, true), MAPE(pred, true)


def gaussian_nll_loss(
    pred_mean: torch.Tensor,
    pred_log_var: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Heteroscedastic Gaussian NLL in normalized units.

    The constant log(2*pi) term is omitted because it does not affect training.
    """
    log_var = pred_log_var.float().clamp(-8.0, 4.0)
    sq_err = (target.float() - pred_mean.float()).pow(2)
    return 0.5 * (log_var + sq_err * torch.exp(-log_var)).mean()


# =============================================================================
# Early stopping
# =============================================================================
class EarlyStopping:
    """
    MGCN EarlyStopping: saves model when val loss improves; stops when
    patience consecutive epochs show no improvement.
    """

    def __init__(self, patience: int = 7) -> None:
        self.patience = patience
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, val_loss: float, net: nn.Module, path: str) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            save_model_state(net, path)
            LOGGER.info("EarlyStopping: initial best model saved.")
        elif score < self.best_score:
            self.counter += 1
            LOGGER.info(
                "EarlyStopping counter: %d / %d",
                self.counter, self.patience,
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            save_model_state(net, path)
            LOGGER.info(
                "EarlyStopping: model saved (val_loss improved to %.6f).",
                val_loss,
            )
            self.counter = 0


# =============================================================================
# Learning-rate schedule
# =============================================================================
def adjust_learning_rate(
    optimizer: optim.Optimizer,
    epoch: int,
    base_lr: float,
    decay_rate: float = 0.5,
    decay_step: int = 10,
) -> None:
    """MGCN step decay: multiply base_lr by decay_rate every decay_step epochs."""
    new_lr = base_lr * (decay_rate ** (epoch // decay_step))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


# =============================================================================
# TEP Data Loading
# =============================================================================
class TEPWindowDataset(Dataset):
    """
    Rolling-window dataset over TEP run trajectories.
    Run-level split prevents data leakage between train and validation.
    Returns: {"x_hist": [T, n_x], "u_hist": [T, n_u], "x_fut": [horizon, n_x]}
    """

    def __init__(
        self,
        x: np.ndarray,
        u: np.ndarray,
        history: int,
        horizon: int,
        stride: int,
        run_indices: np.ndarray,
    ) -> None:
        if x.ndim != 3 or u.ndim != 3:
            raise ValueError("x and u must be [runs, time, features].")
        if x.shape[:2] != u.shape[:2]:
            raise ValueError("x and u must share run/time dimensions.")
        if history < 2 or horizon < 1 or stride < 1:
            raise ValueError("history>=2, horizon>=1, stride>=1 required.")

        self.x = torch.from_numpy(np.ascontiguousarray(x))
        self.u = torch.from_numpy(np.ascontiguousarray(u))
        self.history = int(history)
        self.horizon = int(horizon)

        _, time_len, _ = x.shape
        t_min = history - 1
        t_max = time_len - horizon
        if t_max <= t_min:
            raise ValueError(
                f"Time length {time_len} too short for "
                f"history={history}, horizon={horizon}."
            )

        run_ids_list: List[np.ndarray] = []
        end_times_list: List[np.ndarray] = []
        for run_id in run_indices.astype(np.int32).tolist():
            ts = np.arange(t_min, t_max, stride, dtype=np.int32)
            if ts.size > 0:
                run_ids_list.append(np.full(ts.shape, run_id, dtype=np.int32))
                end_times_list.append(ts)
        if not run_ids_list:
            raise ValueError("No windows created; check split/history/horizon/stride.")
        self.run_ids = np.concatenate(run_ids_list)
        self.end_times = np.concatenate(end_times_list)

    def __len__(self) -> int:
        return int(self.run_ids.size)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        run_id = int(self.run_ids[idx])
        t = int(self.end_times[idx])
        h0, h1 = t - self.history + 1, t + 1
        f0, f1 = t + 1, t + 1 + self.horizon
        return {
            "x_hist": self.x[run_id, h0:h1],   # [T, n_x]
            "u_hist": self.u[run_id, h0:h1],   # [T, n_u]
            "x_fut":  self.x[run_id, f0:f1],   # [horizon, n_x]
        }


def split_runs(
    num_runs: int,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if num_runs < 2:
        raise ValueError("Need at least two runs for train/validation split.")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1); got {val_ratio}.")
    rng = np.random.default_rng(seed)
    idx = np.arange(num_runs)
    rng.shuffle(idx)
    n_val = max(1, min(int(round(num_runs * val_ratio)), num_runs - 1))
    return idx[n_val:], idx[:n_val]   # train_idx, val_idx


def _string_list(arr: np.ndarray) -> List[str]:
    return [str(v) for v in arr.tolist()]


def load_tep_arrays(
    path: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[str]]]:
    raw = np.load(path, allow_pickle=True)
    if "X" not in raw.files or "U" not in raw.files:
        raise ValueError(f"{path} must contain arrays named 'X' and 'U'.")

    x_full = raw["X"].astype(np.float32)
    u = raw["U"].astype(np.float32)

    if x_full.ndim == 2:
        if x_full.shape[0] % 500 != 0:
            raise ValueError("Rank-2 X cannot be reshaped into 500-sample TEP runs.")
        n_runs = x_full.shape[0] // 500
        x_full = x_full.reshape(n_runs, 500, -1)
        u = u.reshape(n_runs, 500, -1)

    if x_full.ndim != 3 or u.ndim != 3:
        raise ValueError("X and U must be rank-3 after loading.")
    if x_full.shape[:2] != u.shape[:2]:
        raise ValueError("X and U must share run/time dimensions.")
    if x_full.shape[-1] != 41:
        raise ValueError(f"Expected 41 XMEAS variables, got {x_full.shape[-1]}.")
    if u.shape[-1] != 11:
        raise ValueError(f"Expected 11 XMV variables, got {u.shape[-1]}.")

    x_names = _string_list(raw["x_names"]) if "x_names" in raw.files else []
    u_names = _string_list(raw["u_names"]) if "u_names" in raw.files else []
    x_cols  = _string_list(raw["x_cols"])  if "x_cols"  in raw.files else []
    u_cols  = _string_list(raw["u_cols"])  if "u_cols"  in raw.files else []

    if x_cols and x_cols[:22] != RAW_X22_COLS:
        raise ValueError(
            f"First 22 X columns mismatch.\nExpected: {RAW_X22_COLS}\nFound: {x_cols[:22]}"
        )
    if u_cols and u_cols[:11] != RAW_U11_COLS:
        raise ValueError(
            f"U columns mismatch.\nExpected: {RAW_U11_COLS}\nFound: {u_cols[:11]}"
        )
    if x_names and x_names[:22] != RAW_X22_NAMES:
        raise ValueError(
            f"First 22 X names mismatch.\nExpected: {RAW_X22_NAMES}\nFound: {x_names[:22]}"
        )
    if u_names and u_names[:11] != RAW_U11_NAMES:
        raise ValueError(
            f"U names mismatch.\nExpected: {RAW_U11_NAMES}\nFound: {u_names[:11]}"
        )

    x = x_full[:, :, :22]

    x_ok = np.isfinite(x).all(axis=(1, 2))
    u_ok = np.isfinite(u).all(axis=(1, 2))
    bad = np.where(~(x_ok & u_ok))[0]
    if bad.size > 0:
        raise ValueError(f"{bad.size} run(s) contain NaN/Inf: {bad.tolist()}")

    mapping: Dict[str, List[str]] = {
        "kg_x_nodes": X_NODE_NAMES,
        "raw_x_column_ids_used": x_cols[:22] if x_cols else RAW_X22_COLS,
        "raw_x_columns_used": x_names[:22] if x_names else RAW_X22_NAMES,
        "raw_x_column_ids_excluded": x_cols[22:] if x_cols else [],
        "raw_x_columns_excluded": x_names[22:] if x_names else [],
        "kg_u_nodes": U_NODE_NAMES,
        "raw_u_column_ids_used": u_cols[:11] if u_cols else RAW_U11_COLS,
        "raw_u_columns_used": u_names[:11] if u_names else RAW_U11_NAMES,
    }
    return x, u, mapping


def load_faulty_tep_arrays(
    path: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Load faulty TEP training trajectories as unlabeled forecasting runs.

    The faulty NPZ stores arrays under fault/simulation key names. Those IDs
    are used only to find the arrays; the returned tensors contain no labels.
    """
    raw = np.load(path, allow_pickle=True)
    required = ["fault_ids", "simulation_runs", "x_cols", "u_cols", "x_names", "u_names"]
    missing = [key for key in required if key not in raw.files]
    if missing:
        raise ValueError(f"{path} is missing required faulty-training metadata: {missing}")

    x_cols = _string_list(raw["x_cols"])
    u_cols = _string_list(raw["u_cols"])
    x_names = _string_list(raw["x_names"])
    u_names = _string_list(raw["u_names"])

    if x_cols[:22] != RAW_X22_COLS:
        raise ValueError(
            f"First 22 X columns mismatch.\nExpected: {RAW_X22_COLS}\nFound: {x_cols[:22]}"
        )
    if u_cols[:11] != RAW_U11_COLS:
        raise ValueError(
            f"U columns mismatch.\nExpected: {RAW_U11_COLS}\nFound: {u_cols[:11]}"
        )
    if x_names[:22] != RAW_X22_NAMES:
        raise ValueError(
            f"First 22 X names mismatch.\nExpected: {RAW_X22_NAMES}\nFound: {x_names[:22]}"
        )
    if u_names[:11] != RAW_U11_NAMES:
        raise ValueError(
            f"U names mismatch.\nExpected: {RAW_U11_NAMES}\nFound: {u_names[:11]}"
        )

    x_runs: List[np.ndarray] = []
    u_runs: List[np.ndarray] = []
    for fault_id in [int(v) for v in raw["fault_ids"].tolist()]:
        for sim_id in [int(v) for v in raw["simulation_runs"].tolist()]:
            x_key = f"fault_id_{fault_id}_sim_{sim_id}_X"
            u_key = f"fault_id_{fault_id}_sim_{sim_id}_U"
            if x_key not in raw.files or u_key not in raw.files:
                raise ValueError(f"{path} is missing arrays {x_key!r}/{u_key!r}.")

            x_full = raw[x_key].astype(np.float32)
            u = raw[u_key].astype(np.float32)
            if x_full.ndim != 2 or u.ndim != 2:
                raise ValueError(f"{x_key}/{u_key} must be rank-2 arrays.")
            if x_full.shape[0] != u.shape[0]:
                raise ValueError(f"{x_key}/{u_key} time dimensions do not match.")
            if x_full.shape[1] != 41:
                raise ValueError(f"Expected 41 XMEAS variables in {x_key}, got {x_full.shape[1]}.")
            if u.shape[1] != 11:
                raise ValueError(f"Expected 11 XMV variables in {u_key}, got {u.shape[1]}.")

            x = x_full[:, :22]
            if not np.isfinite(x).all() or not np.isfinite(u).all():
                raise ValueError(f"{x_key}/{u_key} contains NaN/Inf.")
            x_runs.append(x)
            u_runs.append(u)

    if not x_runs:
        raise ValueError(f"No faulty training runs found in {path}.")

    x_arr = np.stack(x_runs, axis=0).astype(np.float32)
    u_arr = np.stack(u_runs, axis=0).astype(np.float32)
    mapping: Dict[str, object] = {
        "kg_x_nodes": X_NODE_NAMES,
        "raw_x_column_ids_used": x_cols[:22],
        "raw_x_columns_used": x_names[:22],
        "raw_x_column_ids_excluded": x_cols[22:],
        "raw_x_columns_excluded": x_names[22:],
        "kg_u_nodes": U_NODE_NAMES,
        "raw_u_column_ids_used": u_cols[:11],
        "raw_u_columns_used": u_names[:11],
        "num_faulty_unlabeled_runs": int(x_arr.shape[0]),
    }
    return x_arr, u_arr, mapping


def load_training_arrays(cfg: TrainConfig) -> Tuple[np.ndarray, np.ndarray, Dict]:
    x_free, u_free, mapping = load_tep_arrays(cfg.data_npz)
    source_counts = {"fault_free_runs": int(x_free.shape[0]), "faulty_unlabeled_runs": 0}

    if not cfg.include_faulty_training:
        mapping["training_source_counts"] = source_counts
        return x_free, u_free, mapping

    x_faulty, u_faulty, faulty_mapping = load_faulty_tep_arrays(cfg.faulty_data_npz)
    if x_faulty.shape[1:] != x_free.shape[1:]:
        raise ValueError(
            "Fault-free and faulty X arrays must share time/features dimensions: "
            f"{x_free.shape[1:]} vs {x_faulty.shape[1:]}."
        )
    if u_faulty.shape[1:] != u_free.shape[1:]:
        raise ValueError(
            "Fault-free and faulty U arrays must share time/features dimensions: "
            f"{u_free.shape[1:]} vs {u_faulty.shape[1:]}."
        )

    x = np.concatenate([x_free, x_faulty], axis=0)
    u = np.concatenate([u_free, u_faulty], axis=0)
    source_counts["faulty_unlabeled_runs"] = int(x_faulty.shape[0])
    mapping["training_source_counts"] = source_counts
    mapping["faulty_training_column_check"] = {
        k: v for k, v in faulty_mapping.items()
        if k.startswith("raw_") or k.startswith("kg_")
    }
    return x, u, mapping


def load_adjacency(path: str, n_nodes: int) -> np.ndarray:
    """Load and validate the TEP causal KG adjacency matrix.
    Convention: A[dst, src] = 1 means directed edge src -> dst.
    """
    adj = (np.load(path).astype(np.float32) > 0).astype(np.float32)
    if adj.shape != (n_nodes, n_nodes):
        raise ValueError(f"Expected adjacency {(n_nodes, n_nodes)}, got {adj.shape}.")
    if adj.sum() == 0:
        raise ValueError("KG adjacency has no active edges.")
    n_x = n_nodes - 11   # u-nodes are the last 11
    if adj[n_x:, :].sum() != 0:
        raise ValueError("KG violates u-node source-only rule: u-node target rows are nonzero.")
    return adj


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device_type: str,
) -> DataLoader:
    kwargs: Dict = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device_type == "cuda",
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


# =============================================================================
# Model Components
#
# Architecture follows MGCN adapted for the TEP causal knowledge graph.
#
# MGCN block pipeline (two-branch fusion):
#   Branch A (skip):  residual_proj(x)                    -> d_model
#   Branch B (main):  KGGraphLayer -> DataEmbedding_inverted
#                     -> Encoder -> projector2             -> d_model
#   Fusion:           F.relu(A + B) -> projector1 -> LN   -> T
#
# KGGraphLayer replaces ChebConv STGCNLayer with edge-gated KG message passing.
# SelectiveSSMBlock replaces mamba_ssm.Mamba (pure Python, device-agnostic).
# =============================================================================

def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse of softplus: used for dt_proj init."""
    x_safe = x.clamp_min(torch.finfo(x.dtype).tiny)
    return x_safe + torch.log(-torch.expm1(-x_safe))


class SelectiveSSMBlock(nn.Module):
    """
    Pure-Python Mamba-style selective SSM block.

    Replaces mamba_ssm.Mamba -- works on CPU, CUDA, and MPS without any
    external C++/CUDA extensions.  Input/output shape: [B, L, D].

    Selective state-space recurrence:
        h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * x_t
        y_t = C_t * h_t + D * x_t
    """

    def __init__(
        self,
        d_model: int,
        dropout: float,
        d_state: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
    ) -> None:
        super().__init__()
        self.d_inner  = d_model * expand
        self.d_state  = d_state
        self.dt_rank  = max(1, math.ceil(d_model / 16))
        self.dt_min   = 0.001
        self.dt_max   = 0.1

        self.in_proj  = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d   = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=conv_kernel, groups=self.d_inner,
            padding=conv_kernel - 1,
        )
        self.x_proj   = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj  = nn.Linear(self.dt_rank, self.d_inner)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).view(1, d_state)
        self.A_log    = nn.Parameter(torch.log(A.repeat(self.d_inner, 1)))
        self.D        = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.ssm_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.reset_state_parameters()

    def reset_state_parameters(self) -> None:
        A = torch.arange(
            1,
            self.d_state + 1,
            dtype=self.A_log.dtype,
            device=self.A_log.device,
        ).view(1, self.d_state)
        with torch.no_grad():
            self.A_log.copy_(torch.log(A.repeat(self.d_inner, 1)))
            self.D.fill_(1.0)
        self._init_dt()

    def _init_dt(self) -> None:
        std = float(self.dt_rank) ** -0.5
        dt = torch.exp(
            torch.empty(self.d_inner).uniform_(
                math.log(self.dt_min), math.log(self.dt_max)
            )
        )
        with torch.no_grad():
            self.dt_proj.weight.uniform_(-std, std)
            self.dt_proj.bias.copy_(_inverse_softplus(dt))

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, L, D]
        B, L, _ = x.shape
        xz = self.in_proj(x)                                    # [B, L, 2*d_inner]
        x_branch, z_branch = xz.chunk(2, dim=-1)               # each [B, L, d_inner]

        # Causal depthwise conv (truncate padding on the right)
        conv_out = self.conv1d(x_branch.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_branch = F.silu(conv_out)                             # [B, L, d_inner]

        x_fp32 = x_branch.float()
        raw    = self.x_proj(x_branch)
        dt_raw, B_seq, C_seq = raw.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta  = F.softplus(self.dt_proj(dt_raw).float())       # [B, L, d_inner]
        B_seq  = B_seq.float()
        C_seq  = C_seq.float()

        A     = -torch.exp(self.A_log.float()).to(device=x.device)  # [d_inner, d_state]
        D     = self.D.float().to(device=x.device)
        decay = torch.exp(delta.unsqueeze(-1) * A)              # [B, L, d_inner, d_state]
        bu    = delta.unsqueeze(-1) * B_seq.unsqueeze(2) * x_fp32.unsqueeze(-1)

        h: torch.Tensor = bu.new_zeros(B, self.d_inner, self.d_state)
        ys: List[torch.Tensor] = []
        for t in range(L):
            h  = decay[:, t] * h + bu[:, t]
            yt = (h * C_seq[:, t].unsqueeze(1)).sum(-1) + D * x_fp32[:, t]
            ys.append(yt)
        y = torch.stack(ys, dim=1)           # [B, L, d_inner]
        y = F.silu(z_branch.float()) * y
        y = y.to(dtype=x.dtype)
        return self.ssm_dropout(self.out_proj(y))  # [B, L, d_model]


class DataEmbedding_inverted(nn.Module):
    """
    MGCN DataEmbedding_inverted: projects each token's c_in-dimensional
    features (time series of length T) to d_model.

    Input:  [B, n_tokens, c_in]
    Output: [B, n_tokens, d_model]
    """

    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, n_tokens, c_in]
        return self.dropout(self.value_embedding(x))       # [B, n_tokens, d_model]


class EncoderLayer(nn.Module):
    """
    MGCN EncoderLayer: bidirectional Mamba + Conv1d FFN + LayerNorm.

    Bidirectional pattern (exactly as in MGCN):
        new_x = mamba_fwd(x) + mamba_rev(x.flip(dims=[1])).flip(dims=[1])
    """

    def __init__(
        self,
        ssm_fwd: nn.Module,
        ssm_rev: nn.Module,
        d_model: int,
        d_ff: int = 0,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.ssm_fwd = ssm_fwd
        self.ssm_rev = ssm_rev
        self.conv1   = nn.Conv1d(d_model, d_ff,    kernel_size=1)
        self.conv2   = nn.Conv1d(d_ff,    d_model, kernel_size=1)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.enc_dropout = nn.Dropout(dropout)
        self.act     = F.gelu if activation == "gelu" else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, L, d_model]
        # Bidirectional Mamba (MGCN pattern)
        new_x = self.ssm_fwd(x) + self.ssm_rev(x.flip(dims=[1])).flip(dims=[1])
        x = x + new_x
        # Conv1d FFN (MGCN pattern)
        y = x = self.norm1(x)
        y = self.enc_dropout(self.act(self.conv1(y.transpose(-1, 1))))
        y = self.enc_dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Encoder(nn.Module):
    """MGCN Encoder: stack of EncoderLayers with optional final norm."""

    def __init__(
        self,
        attn_layers: List[nn.Module],
        norm_layer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, L, d_model]
        for layer in self.attn_layers:
            x = layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class KGGraphLayer(nn.Module):
    """
    TEP adaptation of MGCN's STGCNLayer (replaces ChebConv on DGL graph).

    Applies edge-gated message passing on the TEP causal knowledge graph.
    No external graph library required.

    Convention: A[dst, src] = 1 means edge src -> dst.
    Input / output: [B, n_nodes, n_features].
    """

    def __init__(
        self,
        n_features: int,
        adjacency: torch.Tensor,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        dst_idx, src_idx = torch.where(adjacency > 0)
        if src_idx.numel() == 0:
            raise ValueError("KG adjacency has no active edges.")
        self.register_buffer("src_idx", src_idx.long())
        self.register_buffer("dst_idx", dst_idx.long())
        self.n_nodes = int(adjacency.shape[0])
        self.n_edges = int(src_idx.numel())

        # Edge gate: how much each source node contributes to each target
        self.edge_gate = nn.Sequential(
            nn.Linear(2 * n_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.msg_proj = nn.Linear(n_features, n_features)
        self.out_norm = nn.LayerNorm(n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, n_nodes, T]
        B, N, T = x.shape
        src_f = x[:, self.src_idx]                             # [B, n_edges, T]
        dst_f = x[:, self.dst_idx]                             # [B, n_edges, T]
        gate  = torch.sigmoid(self.edge_gate(
            torch.cat([dst_f, src_f], dim=-1)                  # [B, n_edges, 2T]
        ))                                                      # [B, n_edges, 1]
        msg   = self.msg_proj(src_f) * gate                    # [B, n_edges, T]

        agg = x.new_zeros(B, N, T)
        dst_exp = self.dst_idx.view(1, -1, 1).expand(B, self.n_edges, T)
        agg.scatter_add_(1, dst_exp, msg.to(dtype=agg.dtype))
        return torch.relu(self.out_norm(x + agg))              # [B, n_nodes, T]


class KGSTGMBlock(nn.Module):
    """
    TEP adaptation of MGCN_block with faithful two-branch skip connection.

    MGCN has two branches that are fused before the final projection:
      Branch A (skip):  residual_proj(x)                         [B, N, d_model]
      Branch B (main):  KGGraphLayer -> Embed -> Encoder -> projector2  [B, N, d_model]
      Fusion:           F.relu(A + B) -> projector1 -> LayerNorm [B, N, T]

    The direct skip (branch A) provides gradient flow that bypasses GCN+Mamba,
    matching the role of MGCN's Conv2d residual_conv.

    Input / output: [B, n_nodes, T]
    """

    def __init__(
        self,
        history: int,
        d_model: int,
        d_ff: int,
        adjacency: torch.Tensor,
        hidden_dim: int,
        st_layers: int,
        d_state: int,
        expand: int,
        conv_kernel: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # Branch A: direct skip from input T-space to d_model (MGCN residual_conv)
        self.residual_proj = nn.Linear(history, d_model, bias=False)
        # Branch B: main path
        self.kg_graph_layer = KGGraphLayer(history, adjacency, hidden_dim)
        self.enc_embedding  = DataEmbedding_inverted(history, d_model, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    SelectiveSSMBlock(d_model, dropout, d_state, expand, conv_kernel),
                    SelectiveSSMBlock(d_model, dropout, d_state, expand, conv_kernel),
                    d_model, d_ff, dropout, activation="gelu",
                )
                for _ in range(st_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.projector2 = nn.Linear(d_model, d_model, bias=True)   # MGCN projector2
        # Fusion -> output (MGCN projector1)
        self.projector1 = nn.Linear(d_model, history, bias=True)
        self.out_norm   = nn.LayerNorm(history)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, n_nodes, T]
        # Branch A: direct skip
        skip = self.residual_proj(x)                       # [B, n_nodes, d_model]
        # Branch B: GCN -> embed -> bidirectional Mamba -> projector2
        gcn_out  = self.kg_graph_layer(x)                  # [B, n_nodes, T]
        enc_in   = self.enc_embedding(gcn_out)             # [B, n_nodes, d_model]
        enc_out  = self.encoder(enc_in)                    # [B, n_nodes, d_model]
        main     = self.projector2(enc_out)                # [B, n_nodes, d_model]
        # Fuse: relu(skip + main) -> project back to T (MGCN projector1)
        fused    = F.relu(skip + main)                     # [B, n_nodes, d_model]
        out      = self.projector1(fused)                  # [B, n_nodes, T]
        return self.out_norm(out)                          # [B, n_nodes, T]


class KGSTGMForecaster(nn.Module):
    """
    TEP adaptation of MGCN_submodule.

    Input  : x_hist [B, T, n_x],  u_hist [B, T, n_u]
    Output : {
        "x_pred":    [B, horizon, n_x],  # predictive mean
        "x_log_var": [B, horizon, n_x],  # predictive log variance, normalized units
        "x_std":     [B, horizon, n_x],  # predictive std, normalized units
    }
    """

    def __init__(
        self,
        adjacency: torch.Tensor,
        hidden_dim: int = 64,
        d_model: int = 128,
        d_ff: int = 256,
        d_state: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
        st_layers: int = 2,
        history: int = 20,
        horizon: int = 10,
        dropout: float = 0.1,
        n_x: int = 22,
        n_u: int = 11,
    ) -> None:
        super().__init__()
        self.n_x     = n_x
        self.n_u     = n_u
        self.n_nodes = n_x + n_u
        self.horizon = horizon

        if tuple(adjacency.shape) != (self.n_nodes, self.n_nodes):
            raise ValueError(
                f"adjacency must be [{self.n_nodes}, {self.n_nodes}], "
                f"got {tuple(adjacency.shape)}."
            )

        # Main MGCN block
        self.block = KGSTGMBlock(
            history=history, d_model=d_model, d_ff=d_ff,
            adjacency=adjacency, hidden_dim=hidden_dim, st_layers=st_layers,
            d_state=d_state, expand=expand, conv_kernel=conv_kernel, dropout=dropout,
        )

        # Final x-node projectors: history -> horizon.
        self.mean_projector = nn.Linear(history, horizon, bias=True)
        self.log_var_projector = nn.Linear(history, horizon, bias=True)

    def forward(
        self,
        x_hist: torch.Tensor,
        u_hist: torch.Tensor,
        horizon: int,
    ) -> Dict[str, torch.Tensor]:
        if horizon != self.horizon:
            raise ValueError(f"Model was built for horizon={self.horizon}, got {horizon}.")

        # Concatenate inputs and invert to token representation  [B, n_nodes, T]
        z = torch.cat([x_hist, u_hist], dim=-1).transpose(1, 2)

        # MGCN block
        h = self.block(z)                          # [B, n_nodes, T]

        # Forecast x-nodes only (like MGCN projector3: Linear(T -> horizon))
        x_seq   = h[:, : self.n_x, :]             # [B, n_x, T]
        x_delta = self.mean_projector(x_seq)       # [B, n_x, horizon]
        x_delta = x_delta.permute(0, 2, 1)        # [B, horizon, n_x]

        # Persistence residual (add last observed value as baseline)
        x_last = x_hist[:, -1, :]                 # [B, n_x]
        x_pred = x_last.unsqueeze(1) + x_delta    # [B, horizon, n_x]

        x_log_var = self.log_var_projector(x_seq).permute(0, 2, 1)
        x_log_var = x_log_var.clamp(-8.0, 4.0)
        x_std = torch.exp(0.5 * x_log_var)

        return {
            "x_pred": x_pred,
            "x_log_var": x_log_var,
            "x_std": x_std,
        }


def make_model(
    device: torch.device,
    adjacency: torch.Tensor,
    cfg: TrainConfig,
    n_x: int,
    n_u: int,
) -> KGSTGMForecaster:
    """
    MGCN make_model factory: instantiate KGSTGMForecaster and apply
    Xavier init to Linear/Conv layers while preserving SSM and LayerNorm
    state conventions.
    """
    model = KGSTGMForecaster(
        adjacency=adjacency,
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

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # Restore Mamba-style state parameters after the generic module init.
    for m in model.modules():
        if isinstance(m, SelectiveSSMBlock):
            m.reset_state_parameters()

    return model


# =============================================================================
# Validation
# =============================================================================
def compute_val_loss(
    net: nn.Module,
    val_loader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    """Switch to eval mode and return sample-weighted average NLL loss."""
    net.train(False)
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {
                k: v.to(device, non_blocking=(device.type == "cuda"))
                for k, v in batch.items()
            }
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = net(batch["x_hist"], batch["u_hist"], batch["x_fut"].size(1))
                loss = criterion(
                    outputs["x_pred"],
                    outputs["x_log_var"],
                    batch["x_fut"],
                )
            batch_size = int(batch["x_fut"].size(0))
            total_loss += loss.item() * batch_size
            total_count += batch_size
    return total_loss / max(total_count, 1)


# =============================================================================
# Prediction and final evaluation
# =============================================================================
def predict_and_save_results(
    net: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    best_epoch: int,
    out_dir: str,
    x_scaler: Optional[FeatureScaler] = None,
) -> Dict[str, float]:
    """
    MGCN predict_and_save_results_mgcn: collect all predictions, compute
    MAE / MSE / RMSE / MAPE, optionally convert to physical units,
    save final_metrics.json.
    """
    net.train(False)
    preds: List[np.ndarray] = []
    pred_stds: List[np.ndarray] = []
    trues: List[np.ndarray] = []

    with torch.no_grad():
        for batch in data_loader:
            batch = {
                k: v.to(device, non_blocking=(device.type == "cuda"))
                for k, v in batch.items()
            }
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = net(batch["x_hist"], batch["u_hist"], batch["x_fut"].size(1))
            preds.append(outputs["x_pred"].float().detach().cpu().numpy())
            pred_stds.append(outputs["x_std"].float().detach().cpu().numpy())
            trues.append(batch["x_fut"].float().detach().cpu().numpy())

    preds_np = np.concatenate(preds, axis=0)   # [N, horizon, n_x]
    stds_np = np.concatenate(pred_stds, axis=0)
    trues_np = np.concatenate(trues, axis=0)   # [N, horizon, n_x]

    # Normalized-space metrics
    mae, mse, rmse, mape = metric(preds_np, trues_np)
    LOGGER.info(
        "Normalized | rmse:%.6f  mse:%.6f  mae:%.6f",
        rmse, mse, mae,
    )

    results: Dict = {
        "best_epoch": int(best_epoch),
        "val_mae":    mae,
        "val_mse":    mse,
        "val_rmse":   rmse,
        "val_mape":   mape,
        "val_pred_std_mean": float(stds_np.mean()),
    }

    # Physical-unit metrics
    if x_scaler is not None:
        N, H, nx = preds_np.shape
        pred_orig = x_scaler.inverse_transform(preds_np.reshape(-1, nx)).reshape(N, H, nx)
        true_orig = x_scaler.inverse_transform(trues_np.reshape(-1, nx)).reshape(N, H, nx)
        std_orig = (stds_np * x_scaler.std.reshape(1, 1, nx)).astype(np.float32)
        o_mae, o_mse, o_rmse, o_mape = metric(pred_orig, true_orig)
        LOGGER.info(
            "Physical   | rmse:%.4f  mse:%.4f  mae:%.4f  mape:%.4f%%",
            o_rmse, o_mse, o_mae, o_mape * 100,
        )
        results.update({
            "val_orig_mae":  o_mae,
            "val_orig_mse":  o_mse,
            "val_orig_rmse": o_rmse,
            "val_orig_mape": o_mape,
            "val_orig_pred_std_mean": float(std_orig.mean()),
        })
        abs_err = np.abs(pred_orig - true_orig)
        sq_err  = (pred_orig - true_orig) ** 2
        for step in range(H):
            results[f"val_orig_mae_h{step + 1}"]  = float(abs_err[:, step, :].mean())
            results[f"val_orig_rmse_h{step + 1}"] = float(np.sqrt(sq_err[:, step, :].mean()))

    save_json(results, str(Path(out_dir) / "final_metrics.json"))
    return {k: v for k, v in results.items() if isinstance(v, (int, float))}


# =============================================================================
# Evaluation helper
# =============================================================================
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    use_amp: bool,
    amp_dtype: torch.dtype,
    x_scaler: Optional[FeatureScaler] = None,
) -> Dict[str, float]:
    """Compute validation metrics (normalized and optionally physical units)."""
    model.eval()
    mae_metric = nn.L1Loss()
    all_pred: List[np.ndarray] = []
    all_true: List[np.ndarray] = []
    total_mae = total_mse = total_count = 0.0

    for batch in loader:
        batch = {
            k: v.to(device, non_blocking=(device.type == "cuda"))
            for k, v in batch.items()
        }
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(batch["x_hist"], batch["u_hist"], cfg.horizon)
        pred = outputs["x_pred"].float()
        true = batch["x_fut"].float()
        n = float(pred.size(0))
        total_mae   += mae_metric(pred, true).item() * n
        total_mse   += F.mse_loss(pred, true).item() * n
        total_count += n
        if x_scaler is not None:
            all_pred.append(pred.cpu().numpy())
            all_true.append(true.cpu().numpy())

    denom = max(total_count, 1.0)
    out: Dict[str, float] = {
        "loss": total_mae / denom,
        "mae":  total_mae / denom,
        "mse":  total_mse / denom,
        "rmse": math.sqrt(total_mse / denom),
    }

    if x_scaler is not None and all_pred:
        pred_cat = np.concatenate(all_pred, axis=0)
        true_cat = np.concatenate(all_true, axis=0)
        N, H, nx = pred_cat.shape
        pred_orig = x_scaler.inverse_transform(pred_cat.reshape(-1, nx)).reshape(N, H, nx)
        true_orig = x_scaler.inverse_transform(true_cat.reshape(-1, nx)).reshape(N, H, nx)
        abs_err = np.abs(pred_orig - true_orig)
        sq_err  = (pred_orig - true_orig) ** 2
        out["orig_mae"]  = float(abs_err.mean())
        out["orig_rmse"] = float(np.sqrt(sq_err.mean()))
        nz = np.abs(true_orig) > 1e-6
        out["orig_mape"] = float(
            (abs_err[nz] / np.abs(true_orig[nz])).mean() * 100.0
        ) if nz.any() else float("nan")
        for step in range(H):
            out[f"orig_mae_h{step + 1}"]  = float(abs_err[:, step, :].mean())
            out[f"orig_rmse_h{step + 1}"] = float(np.sqrt(sq_err[:, step, :].mean()))

    return out


# =============================================================================
# Training History Plot
# =============================================================================
def plot_training_history(history: Dict[str, List[float]], out_path: str) -> None:
    if not history.get("train_loss"):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.ravel()

    panels = [
        (0, "Gaussian NLL (normalized)", "train_loss", "val_loss"),
        (1, "RMSE (normalized)",     "train_rmse", None),
        (2, "Gradient Norm",         "grad_norm",  None),
        (3, "Learning Rate",         "lr",         None),
    ]
    for ax_idx, title, train_key, val_key in panels:
        ax = axes[ax_idx]
        if train_key in history:
            ax.plot(epochs, history[train_key], label=train_key.replace("_", " "))
        if val_key and val_key in history:
            v = np.asarray(history[val_key], dtype=np.float32)
            mask = np.isfinite(v)
            if mask.any():
                ax.plot(epochs[mask], v[mask], label=val_key.replace("_", " "))
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Argument Parsing
# =============================================================================
def parse_args() -> TrainConfig:
    defaults = TrainConfig()
    p = argparse.ArgumentParser(description="Train KG-STGM (MGCN-style) on TEP data.")
    p.add_argument("--data-npz",         default=defaults.data_npz)
    p.add_argument("--faulty-data-npz",  default=defaults.faulty_data_npz)
    p.add_argument("--include-faulty-training", dest="include_faulty_training", action="store_true")
    p.add_argument("--no-faulty-training",      dest="include_faulty_training", action="store_false")
    p.set_defaults(include_faulty_training=defaults.include_faulty_training)
    p.add_argument("--adjacency-npy",    default=defaults.adjacency_npy)
    p.add_argument("--out-dir",          default=defaults.out_dir)
    p.add_argument("--history",          type=int,   default=defaults.history)
    p.add_argument("--horizon",          type=int,   default=defaults.horizon)
    p.add_argument("--stride",           type=int,   default=defaults.stride)
    p.add_argument("--d-model",          type=int,   default=defaults.d_model)
    p.add_argument("--d-ff",             type=int,   default=defaults.d_ff)
    p.add_argument("--hidden-dim",       type=int,   default=defaults.hidden_dim)
    p.add_argument("--d-state",          type=int,   default=defaults.d_state)
    p.add_argument("--expand",           type=int,   default=defaults.expand)
    p.add_argument("--conv-kernel",      type=int,   default=defaults.conv_kernel)
    p.add_argument("--st-layers",        type=int,   default=defaults.st_layers)
    p.add_argument("--dropout",          type=float, default=defaults.dropout)
    p.add_argument("--batch-size",       type=int,   default=defaults.batch_size)
    p.add_argument("--epochs",           type=int,   default=defaults.epochs)
    p.add_argument("--lr",               type=float, default=defaults.lr)
    p.add_argument("--weight-decay",     type=float, default=defaults.weight_decay)
    p.add_argument("--grad-clip",        type=float, default=defaults.grad_clip)
    p.add_argument("--patience",         type=int,   default=defaults.patience)
    p.add_argument("--val-ratio",        type=float, default=defaults.val_ratio)
    p.add_argument("--lr-decay-rate",    type=float, default=defaults.lr_decay_rate)
    p.add_argument("--lr-decay-step",    type=int,   default=defaults.lr_decay_step)
    p.add_argument("--input-jitter-std", type=float, default=defaults.input_jitter_std)
    p.add_argument("--seed",             type=int,   default=defaults.seed)
    p.add_argument("--num-workers",      type=int,   default=defaults.num_workers)
    p.add_argument("--device",           default=defaults.device)
    p.add_argument("--compile-model",    action="store_true")
    p.add_argument("--amp",    dest="amp", action="store_true")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=defaults.amp)
    p.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default=defaults.amp_dtype)
    return TrainConfig(**vars(p.parse_args()))


# =============================================================================
# Main Training Loop
# Order per epoch: 1) adjust LR  2) train batches  3) validate
# =============================================================================
def main() -> None:
    cfg = parse_args()
    validate_config(cfg)
    setup_logging(cfg.out_dir)
    set_seed(cfg.seed)

    device    = resolve_device(cfg.device)
    cfg.device = str(device)
    use_amp   = bool(cfg.amp) and device.type != "cpu"
    # MPS only supports float16 for autocast; bfloat16 is CUDA/CPU only
    if device.type == "mps":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float16 if cfg.amp_dtype == "float16" else torch.bfloat16
    # GradScaler only useful for float16 on CUDA
    use_scaler = bool(use_amp and amp_dtype == torch.float16 and device.type == "cuda")

    LOGGER.info("Configuration:\n%s", json.dumps(asdict(cfg), indent=2))
    log_device_details(device)

    # ── Data ──────────────────────────────────────────────────────────────────
    x_raw, u_raw, variable_mapping = load_training_arrays(cfg)
    n_runs, time_len, n_x = x_raw.shape
    n_u = u_raw.shape[-1]
    adjacency_np = load_adjacency(cfg.adjacency_npy, n_nodes=n_x + n_u)

    train_idx, val_idx = split_runs(n_runs, cfg.val_ratio, cfg.seed)
    x_scaler = FeatureScaler.fit(x_raw[train_idx])
    u_scaler = FeatureScaler.fit(u_raw[train_idx])
    x = x_scaler.transform(x_raw)
    u = u_scaler.transform(u_raw)

    LOGGER.info(
        "Data: runs=%d, time=%d, model nodes=%d (22 X + 11 U) | KG edges=%d",
        n_runs, time_len, n_x + n_u, int(adjacency_np.sum()),
    )
    LOGGER.info("Training source counts: %s", variable_mapping["training_source_counts"])
    LOGGER.info(
        "Excluded XMEAS (not in TEP KG): %s",
        variable_mapping["raw_x_columns_excluded"],
    )

    # Save artifacts required by inference/evaluation scripts.
    output_dir = Path(cfg.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "adjacency.npy", adjacency_np)
    np.savez_compressed(
        output_dir / "scaler_stats.npz",
        x_mean=x_scaler.mean, x_std=x_scaler.std,
        u_mean=u_scaler.mean, u_std=u_scaler.std,
        x_names=np.asarray(X_NODE_NAMES),
        u_names=np.asarray(U_NODE_NAMES),
        node_names=np.asarray(NODE_NAMES),
        normalization=np.asarray("zscore"),
        variable_mapping=json.dumps(variable_mapping),
    )

    # ── Data Loaders ──────────────────────────────────────────────────────────
    train_ds = TEPWindowDataset(x, u, cfg.history, cfg.horizon, cfg.stride, train_idx)
    val_ds   = TEPWindowDataset(x, u, cfg.history, cfg.horizon, cfg.stride, val_idx)
    nw = choose_num_workers(cfg.num_workers, device.type)
    train_loader = make_dataloader(train_ds, cfg.batch_size, True,  nw, device.type)
    val_loader   = make_dataloader(val_ds,   cfg.batch_size, False, nw, device.type)
    LOGGER.info("Train windows=%d | Val windows=%d", len(train_ds), len(val_ds))

    # ── Model  (MGCN make_model factory with Xavier init) ─────────────────────
    adjacency_t = torch.from_numpy(adjacency_np)
    net = make_model(device, adjacency_t, cfg, n_x, n_u)
    if cfg.compile_model and device.type == "cuda" and hasattr(torch, "compile"):
        LOGGER.info("Applying torch.compile(mode='reduce-overhead')")
        net = torch.compile(net, mode="reduce-overhead")  # type: ignore[assignment]

    params = count_parameters(net)
    LOGGER.info(
        "Model parameters: total=%d | trainable=%d",
        params["total"], params["trainable"],
    )

    # ── Criterion, Optimizer, GradScaler ──────────────────────────────────────
    criterion = gaussian_nll_loss
    optimizer   = optim.Adam(
        net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    grad_scaler = torch.amp.GradScaler(
        "cuda" if device.type == "cuda" else "cpu", enabled=use_scaler
    )

    # ── EarlyStopping  (MGCN) ─────────────────────────────────────────────────
    best_model_path = str(output_dir / "best_model.pt")
    early_stopping  = EarlyStopping(patience=cfg.patience)

    # ── Training History ──────────────────────────────────────────────────────
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_rmse": [],
        "val_loss":   [],
        "grad_norm":  [],
        "lr":         [],
    }
    best_val_loss = math.inf
    best_epoch    = 1   # 1-indexed; updated when val loss improves
    start_time    = time.time()

    # ── Epoch Loop ────────────────────────────────────────────────────────────
    for epoch in range(cfg.epochs):

        # 1. Adjust LR
        adjust_learning_rate(
            optimizer, epoch, cfg.lr, cfg.lr_decay_rate, cfg.lr_decay_step
        )

        # 2. Train one epoch
        net.train()
        train_nll_sum = 0.0
        train_mae_sum = 0.0
        train_mse_sum = 0.0
        train_count = 0
        grad_norm_sum   = 0.0
        grad_norm_count = 0

        for batch in train_loader:
            batch = {
                k: v.to(device, non_blocking=(device.type == "cuda"))
                for k, v in batch.items()
            }

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                x_hist = batch["x_hist"]
                u_hist = batch["u_hist"]
                if cfg.input_jitter_std > 0.0:
                    x_hist = x_hist + cfg.input_jitter_std * torch.randn_like(x_hist)
                    u_hist = u_hist + cfg.input_jitter_std * torch.randn_like(u_hist)
                outputs = net(x_hist, u_hist, cfg.horizon)
                loss = criterion(
                    outputs["x_pred"],
                    outputs["x_log_var"],
                    batch["x_fut"],
                )

            if use_scaler:
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                if cfg.grad_clip > 0.0:
                    gn = nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                else:
                    gn = torch.tensor(0.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip > 0.0:
                    gn = nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                else:
                    gn = torch.tensor(0.0)
                optimizer.step()

            if torch.isfinite(gn):
                grad_norm_sum   += float(gn)
                grad_norm_count += 1

            batch_size = int(batch["x_fut"].size(0))
            train_nll_sum += loss.item() * batch_size
            train_count += batch_size
            with torch.no_grad():
                train_mae_sum += F.l1_loss(
                    outputs["x_pred"].float().detach(),
                    batch["x_fut"].float(),
                ).item() * batch_size
                train_mse_sum += F.mse_loss(
                    outputs["x_pred"].float().detach(),
                    batch["x_fut"].float(),
                ).item() * batch_size

        # 3. Validate after training
        val_loss = compute_val_loss(
            net, val_loader, criterion, device, use_amp, amp_dtype
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch + 1

        # EarlyStopping saves best model and increments counter
        early_stopping(val_loss, net, best_model_path)

        # Epoch summary
        avg_loss = train_nll_sum / max(train_count, 1)
        avg_mae  = train_mae_sum / max(train_count, 1)
        avg_rmse = math.sqrt(train_mse_sum / max(train_count, 1))
        avg_gn   = grad_norm_sum / max(grad_norm_count, 1)
        cur_lr   = optimizer.param_groups[0]["lr"]
        is_best  = "*" if val_loss <= best_val_loss else ""
        LOGGER.info(
            "Epoch %03d | nll=%.6f | mae=%.6f | val_nll=%.6f | best=%.6f%s | gn=%.3e | lr=%.1e | %.1f s",
            epoch + 1, avg_loss, avg_mae, val_loss, best_val_loss, is_best,
            avg_gn, cur_lr, time.time() - start_time,
        )

        history["train_loss"].append(avg_loss)
        history["train_rmse"].append(avg_rmse)
        history["val_loss"].append(val_loss)
        history["grad_norm"].append(avg_gn)
        history["lr"].append(cur_lr)

        if early_stopping.early_stop:
            LOGGER.info("Early stopping triggered at epoch %d.", epoch + 1)
            break

    # ── Post-training ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    LOGGER.info(
        "Training complete in %.1f s. Best epoch=%d, best val_loss=%.6f",
        elapsed, int(best_epoch), best_val_loss,
    )

    save_model_state(net, str(output_dir / "last_model.pt"))
    save_json(asdict(cfg), str(output_dir / "config.json"))
    save_json(history, str(output_dir / "train_history.json"))
    plot_training_history(history, str(output_dir / "training_curves.png"))

    # Load best model and run final evaluation  (MGCN predict_main pattern)
    if Path(best_model_path).is_file():
        unwrap_compiled_model(net).load_state_dict(load_state_dict_file(best_model_path, device))
        LOGGER.info("Loaded best model (epoch %d) for final evaluation.", int(best_epoch))

    final_results = predict_and_save_results(
        net, val_loader, device, use_amp, amp_dtype,
        best_epoch, cfg.out_dir, x_scaler,
    )
    LOGGER.info(
        "Final | MAE=%.6f | RMSE=%.6f | MAPE=%.4f%%",
        final_results.get("val_mae",  float("nan")),
        final_results.get("val_rmse", float("nan")),
        final_results.get("val_mape", float("nan")) * 100,
    )


if __name__ == "__main__":
    main()
