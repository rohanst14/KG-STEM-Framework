"""
Main principles
---------------
1. KG-STGM provides forecast mean and forecast standard deviation for all
   measured process variables over H horizons.

2. KG-STGM forecast scenarios are sampled directly from the calibrated forecast
   distribution:

      X^(m)_{t+1:t+H} = mu_{t+1:t+H}
                       + sigma_cal_{t+1:t+H} * Z^(m)_{1:H}

   where sigma_cal matches local_risk.py's uncertainty calibration and Z is a
   standard normal innovation tensor. No copula, full-horizon union probability,
   or empirical residual-block resampling is used.

3. Process abnormality is a KG-calibrated graph monitoring event. Each
   forecast scenario is converted to the centered normalized deviation
   e=(X-mu^N)/s^N, propagated through the normalized knowledge graph, and
   summarized as a process score J_K:

       A_g^(m)(t,h) = 1[ J_K^(m)(t,h) > J_alpha ]

       P_g(t,h) = mean_m A_g^(m)(t,h)

   J_alpha is calibrated from fault-free KG-STGM forecast scenarios generated
   with the same scenario sampler used at test time, not from raw normal states.

4. Severity is evaluated on the same joint sampled process state. Variable
   severity is zero inside the normal region and increases outside the normal
   limits. If external engineering safe limits are supplied, severity reaches
   one at the safe boundary. Without safe limits, the output is an operational
   risk index, not a true safety or monetary risk.

5. Unit-level consequence uses the maximum variable severity within each
   equipment unit to avoid double-counting several correlated measurements from
   the same unit:

       C_g^(m)(t,h) = sum_k EML_k * max_{i in unit k} phi_i(X_i^(m)(t,h))

6. Global risk uses the same graph abnormality event for probability and
   conditional consequence:

       R_g(t,h) = P_g(t,h) E[C_g(t,h) | A_g(t,h)=1]

7. Forecast-horizon risk uses the maximum expected risk across the forecast
   horizons:

      R_max(t) = max_h R_g(t,h)

   The selected horizon is reported as a diagnostic. This avoids using a
   full-horizon union event or trajectory-level max consequence.

This formulation uses the same normal-limit definitions and forecast
uncertainty calibration as local_risk.py. The process probability itself is
defined only by the KG-calibrated monitoring statistic.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/kg_stgm_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
from scipy import special, stats
import torch
import torch.nn as nn

from train import (
    FeatureScaler,
    KGSTGMForecaster,
    TrainConfig,
    load_adjacency,
    load_tep_arrays,
    load_state_dict_file,
    resolve_device,
    set_seed,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_project_path(path: Optional[str]) -> Optional[str]:
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path)


TEP_UNIT_NAMES = ("reactor", "condenser", "separator", "compressor", "stripper")

# Assumed TEP unit EML allocation for dimensioned consequence, totaling $100k/h.
# Override with --unit_eml or --unit_eml_values if you need different weights.
DEFAULT_UNIT_EML = {
    "reactor": 40_000.0,
    "condenser": 5_000.0,
    "separator": 25_000.0,
    "compressor": 10_000.0,
    "stripper": 20_000.0,
}

# Maps each x1..x22 variable to one of the five TEP units above.
X_UNIT_BY_INDEX = np.array(
    [
        0,  # x1  A feed stream
        0,  # x2  D feed stream
        0,  # x3  E feed stream
        4,  # x4  A/C feed to stripper
        0,  # x5  recycle flow into reactor
        0,  # x6  reactor feed rate
        0,  # x7  reactor pressure
        0,  # x8  reactor level
        0,  # x9  reactor temperature
        2,  # x10 purge rate
        2,  # x11 separator temperature
        2,  # x12 separator level
        2,  # x13 separator pressure
        2,  # x14 separator underflow
        4,  # x15 stripper level
        4,  # x16 stripper pressure
        4,  # x17 stripper underflow
        4,  # x18 stripper temperature
        4,  # x19 stripper steam flow
        3,  # x20 compressor work
        0,  # x21 reactor cooling-water outlet temperature
        1,  # x22 condenser cooling-water outlet temperature
    ],
    dtype=np.int64,
)


@dataclass
class VariableDistributionModel:
    dist_name: str
    params: Tuple[float, ...]
    target: float
    low: float
    high: float


@dataclass
class SeverityModel:
    variable_models: List[VariableDistributionModel]
    x_unit_index: np.ndarray
    unit_eml: np.ndarray


@dataclass
class KGProbabilityModel:
    normalized_adjacency: np.ndarray
    normal_target: np.ndarray
    normal_scale: np.ndarray
    score_weights: np.ndarray
    propagation_depth: int
    rho: float
    control_limit: float


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_train_config(path: str) -> TrainConfig:
    raw = load_json(path)
    valid = {field.name for field in fields(TrainConfig)}
    return TrainConfig(**{key: value for key, value in raw.items() if key in valid})


def resolve_checkpoint(out_dir: str, checkpoint: str) -> str:
    if os.path.isabs(checkpoint):
        return checkpoint
    return os.path.join(out_dir, checkpoint)


def load_saved_scalers(path: str) -> Tuple[FeatureScaler, FeatureScaler]:
    stats = np.load(path, allow_pickle=False)
    required = ["x_mean", "x_std", "u_mean", "u_std"]
    missing = [key for key in required if key not in stats.files]
    if missing:
        raise ValueError(f"{path} is missing scaler arrays: {missing}")
    return (
        FeatureScaler(stats["x_mean"], stats["x_std"]),
        FeatureScaler(stats["u_mean"], stats["u_std"]),
    )


def build_model(
    cfg: TrainConfig,
    adjacency: np.ndarray,
    n_x: int,
    n_u: int,
    device: torch.device,
) -> nn.Module:
    return KGSTGMForecaster(
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


def parse_int_ranges(raw: str, label: str) -> List[int]:
    values: List[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            start, end = int(start_raw), int(end_raw)
            step = 1 if start <= end else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(token))
    unique = list(dict.fromkeys(values))
    if not unique:
        raise ValueError(f"No {label} were provided.")
    return unique


def load_faulty_metadata(data: np.lib.npyio.NpzFile) -> Tuple[List[int], Dict[int, str]]:
    fault_ids = (
        [int(value) for value in data["fault_ids"].tolist()]
        if "fault_ids" in data.files
        else list(range(1, 21))
    )
    names: Dict[int, str] = {}
    if "fault_names" in data.files:
        raw_names = [str(value) for value in data["fault_names"].tolist()]
        names = {
            fault_id: raw_names[index]
            for index, fault_id in enumerate(fault_ids)
            if index < len(raw_names)
        }
    return fault_ids, names


def load_faulty_run(
    data: np.lib.npyio.NpzFile,
    fault_id: int,
    sim_id: int,
    n_x: int,
    n_u: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_key = f"fault_id_{fault_id}_sim_{sim_id}_X"
    u_key = f"fault_id_{fault_id}_sim_{sim_id}_U"
    if x_key not in data.files or u_key not in data.files:
        raise KeyError(f"Missing faulty-test arrays {x_key!r}/{u_key!r}.")
    x_full = data[x_key].astype(np.float32)
    u = data[u_key].astype(np.float32)
    if x_full.ndim != 2 or u.ndim != 2:
        raise ValueError(f"Fault {fault_id}, sim {sim_id} arrays must be rank-2.")
    if x_full.shape[0] != u.shape[0]:
        raise ValueError(f"Fault {fault_id}, sim {sim_id} X/U lengths differ.")
    if x_full.shape[1] < n_x or u.shape[1] != n_u:
        raise ValueError(f"Fault {fault_id}, sim {sim_id} has incompatible features.")
    x = x_full[:, :n_x]
    if not np.isfinite(x).all() or not np.isfinite(u).all():
        raise ValueError(f"Fault {fault_id}, sim {sim_id} contains NaN/Inf.")
    return x, u


def load_normal_operating_stats(path: str, n_x: int) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} was not found. Run testing_faulty.py first."
        )
    raw = np.load(path, allow_pickle=False)
    required = ["x_target", "x_low", "x_high", "x_scale"]
    missing = [key for key in required if key not in raw.files]
    if missing:
        raise ValueError(f"{path} is missing arrays: {missing}")

    result = {
        "x_target": raw["x_target"].astype(np.float64)[:n_x],
        "x_low": raw["x_low"].astype(np.float64)[:n_x],
        "x_high": raw["x_high"].astype(np.float64)[:n_x],
        "x_scale": raw["x_scale"].astype(np.float64)[:n_x],
    }
    if "residual_std" in raw.files:
        result["residual_std"] = raw["residual_std"].astype(np.float64)[:, :n_x]

    if result["x_low"].shape != (n_x,) or result["x_high"].shape != (n_x,):
        raise ValueError(f"{path} must provide normal bounds for all {n_x} X variables.")
    if not np.all(result["x_low"] < result["x_high"]):
        raise ValueError("Every normal X region must satisfy x_low < x_high.")
    if result["x_scale"].shape != (n_x,) or not np.all(result["x_scale"] > 0.0):
        raise ValueError("Every X variable must have a positive normal scaling factor.")
    if "residual_std" in result:
        if result["residual_std"].ndim != 2 or result["residual_std"].shape[1] != n_x:
            raise ValueError("residual_std must have one column per X variable.")
        if not np.all(np.isfinite(result["residual_std"])) or not np.all(result["residual_std"] > 0.0):
            raise ValueError("Every residual_std value must be finite and positive.")
    return result


def load_distribution_models(path: str, n_x: int) -> List[VariableDistributionModel]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} was not found. Run data_fitting.py first to fit variable distributions."
        )
    payload = load_json(path)
    variables = sorted(payload.get("variables", []), key=lambda item: int(item["index"]))
    if len(variables) != n_x:
        raise ValueError(f"{path} must contain exactly {n_x} fitted X-variable models.")

    models: List[VariableDistributionModel] = []
    for expected_index, variable in enumerate(variables):
        index = int(variable["index"])
        if index != expected_index:
            raise ValueError(f"{path} has a missing or duplicate model near x{expected_index + 1}.")
        dist_name = str(variable["dist_name"])
        if not hasattr(stats, dist_name):
            raise ValueError(f"Unknown scipy distribution {dist_name!r} for x{index + 1}.")
        params = tuple(float(value) for value in variable["params"])
        if not all(np.isfinite(value) for value in params):
            raise ValueError(f"Fitted distribution for x{index + 1} contains NaN/Inf.")
        models.append(
            VariableDistributionModel(
                dist_name=dist_name,
                params=params,
                target=float(variable.get("target", np.nan)),
                low=float(variable.get("low", np.nan)),
                high=float(variable.get("high", np.nan)),
            )
        )
    return models


def default_unit_eml_array() -> np.ndarray:
    return np.asarray([DEFAULT_UNIT_EML[name] for name in TEP_UNIT_NAMES], dtype=np.float64)


def parse_unit_eml_values(raw: Optional[str]) -> Optional[np.ndarray]:
    if raw is None:
        return None
    values = np.asarray([float(tok.strip()) for tok in raw.split(",") if tok.strip()], dtype=np.float64)
    if values.shape != (len(TEP_UNIT_NAMES),):
        raise ValueError(
            f"--unit_eml_values must provide {len(TEP_UNIT_NAMES)} values in order: "
            + ",".join(TEP_UNIT_NAMES)
        )
    if not np.isfinite(values).all() or np.any(values < 0.0) or values.sum() <= 0.0:
        raise ValueError("--unit_eml_values must be finite, non-negative, and positive in total.")
    return values


def build_severity_model(
    variable_models: List[VariableDistributionModel],
    unit_eml: np.ndarray,
) -> SeverityModel:
    if unit_eml.shape != (len(TEP_UNIT_NAMES),):
        raise ValueError("unit_eml has incorrect shape.")
    if np.any(unit_eml < 0.0) or unit_eml.sum() <= 0.0:
        raise ValueError("unit_eml must be non-negative with positive total.")
    return SeverityModel(
        variable_models=variable_models,
        x_unit_index=X_UNIT_BY_INDEX.copy(),
        unit_eml=unit_eml.astype(np.float64),
    )


def load_safe_operating_limits(
    path: Optional[str],
    normal_low: np.ndarray,
    normal_high: np.ndarray,
) -> Optional[Dict[str, np.ndarray]]:
    if path is None:
        return None
    payload = load_json(path)
    if "variables" in payload:
        variables = sorted(payload["variables"], key=lambda item: int(item["index"]))
        if len(variables) != normal_low.size:
            raise ValueError(f"{path} must contain {normal_low.size} safe-limit entries.")
        indices = [int(item["index"]) for item in variables]
        if indices != list(range(normal_low.size)):
            raise ValueError(f"{path} safe-limit indices must be contiguous and zero-based.")
        safe_low = np.asarray([float(item["low"]) for item in variables], dtype=np.float64)
        safe_high = np.asarray([float(item["high"]) for item in variables], dtype=np.float64)
    else:
        low_key = "x_low" if "x_low" in payload else "safe_low"
        high_key = "x_high" if "x_high" in payload else "safe_high"
        if low_key not in payload or high_key not in payload:
            raise ValueError(
                f"{path} must provide variables[index,low,high] or x_low/x_high arrays."
            )
        safe_low = np.asarray(payload[low_key], dtype=np.float64)
        safe_high = np.asarray(payload[high_key], dtype=np.float64)

    if safe_low.shape != normal_low.shape or safe_high.shape != normal_high.shape:
        raise ValueError("Safe-limit arrays must have one value per X variable.")
    if not np.isfinite(safe_low).all() or not np.isfinite(safe_high).all():
        raise ValueError("Safe limits contain NaN or Inf.")
    if not np.all(safe_low < normal_low):
        raise ValueError("Every lower safe limit must be below its lower normal limit.")
    if not np.all(safe_high > normal_high):
        raise ValueError("Every upper safe limit must be above its upper normal limit.")
    return {"x_low": safe_low, "x_high": safe_high}


def fitted_log_tails(
    values: np.ndarray,
    models: List[VariableDistributionModel],
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float64)
    if x.shape[-1] != len(models):
        raise ValueError("Sample variable dimension does not match fitted models.")
    log_cdf = np.empty_like(x, dtype=np.float64)
    log_survival = np.empty_like(x, dtype=np.float64)
    for index, model in enumerate(models):
        distribution = getattr(stats, model.dist_name)
        if model.dist_name == "skewnorm":
            shape, loc, scale = model.params
            z = (x[..., index] - loc) / max(float(scale), np.finfo(np.float64).tiny)
            cdf = special.ndtr(z) - 2.0 * special.owens_t(z, float(shape))
            cdf = np.clip(cdf, np.finfo(np.float64).tiny, 1.0 - np.finfo(np.float64).eps)
            sf = np.clip(1.0 - cdf, np.finfo(np.float64).tiny, 1.0)
            log_cdf[..., index] = np.log(cdf)
            log_survival[..., index] = np.log(sf)
        else:
            log_cdf[..., index] = distribution.logcdf(x[..., index], *model.params)
            log_survival[..., index] = distribution.logsf(x[..., index], *model.params)

    invalid = (
        np.isnan(log_cdf)
        | np.isposinf(log_cdf)
        | np.isnan(log_survival)
        | np.isposinf(log_survival)
    )
    if np.any(invalid):
        raise ValueError("Fitted marginal tail evaluation produced an invalid value.")
    return log_cdf, log_survival


def variable_severity_index(
    samples: np.ndarray,
    models: List[VariableDistributionModel],
    normal_low: np.ndarray,
    normal_high: np.ndarray,
    safe_limits: Optional[Dict[str, np.ndarray]],
) -> np.ndarray:
    """
    Dimensionless variable severity in [0, 1].

    Severity is zero inside [normal_low, normal_high]. If safe limits exist,
    severity reaches one at [safe_low, safe_high]. If safe limits do not exist,
    severity rises toward one in the fitted distribution tail.
    """
    values = np.asarray(samples, dtype=np.float64)

    log_cdf, log_survival = fitted_log_tails(values, models)
    low_log_cdf, _ = fitted_log_tails(normal_low.reshape(1, -1), models)
    _, high_log_survival = fitted_log_tails(normal_high.reshape(1, -1), models)

    low_log_cdf = low_log_cdf[0]
    high_log_survival = high_log_survival[0]

    lower_numerator = -np.expm1(np.minimum(log_cdf - low_log_cdf, 0.0))
    upper_numerator = -np.expm1(np.minimum(log_survival - high_log_survival, 0.0))

    if safe_limits is None:
        lower_denominator = np.ones_like(low_log_cdf)
        upper_denominator = np.ones_like(high_log_survival)
    else:
        safe_low_log_cdf, _ = fitted_log_tails(safe_limits["x_low"].reshape(1, -1), models)
        _, safe_high_log_survival = fitted_log_tails(safe_limits["x_high"].reshape(1, -1), models)
        lower_denominator = -np.expm1(np.minimum(safe_low_log_cdf[0] - low_log_cdf, 0.0))
        upper_denominator = -np.expm1(
            np.minimum(safe_high_log_survival[0] - high_log_survival, 0.0)
        )

    if np.any(lower_denominator <= 0.0) or np.any(upper_denominator <= 0.0):
        raise ValueError(
            "Normal/safe boundaries are indistinguishable under a fitted marginal CDF."
        )

    lower_severity = lower_numerator / lower_denominator
    upper_severity = upper_numerator / upper_denominator

    severity = np.where(
        values < normal_low,
        lower_severity,
        np.where(values > normal_high, upper_severity, 0.0),
    )
    return np.clip(severity, 0.0, 1.0)


def conditional_consequence(expected_consequence: np.ndarray, event_probability: np.ndarray) -> np.ndarray:
    return np.divide(
        expected_consequence,
        event_probability,
        out=np.zeros_like(expected_consequence, dtype=np.float64),
        where=event_probability > 0.0,
    )


def normalize_kg_adjacency(adjacency: np.ndarray) -> np.ndarray:
    """
    Column-normalize A[dst, src] so deviations propagate downstream from each
    source node to its KG children with bounded total outgoing influence.
    """
    adj = np.maximum(np.asarray(adjacency, dtype=np.float64), 0.0)
    col_sum = adj.sum(axis=0, keepdims=True)
    return np.divide(adj, col_sum, out=np.zeros_like(adj), where=col_sum > 0.0)


def standardized_deviation(
    values: np.ndarray,
    normal_target: np.ndarray,
    normal_scale: np.ndarray,
) -> np.ndarray:
    """
    Centered normalized deviation e_i = (X_i - mu_i^N) / s_i^N.
    """
    x = np.asarray(values, dtype=np.float64)
    target = normal_target.reshape((1,) * (x.ndim - 1) + (-1,))
    scale = normal_scale.reshape((1,) * (x.ndim - 1) + (-1,))
    return (x - target) / scale


def kg_monitoring_score(samples: np.ndarray, probability_model: KGProbabilityModel) -> np.ndarray:
    """
    KG-calibrated graph monitoring score J_K for forecasted process states.

    The model uses full KG propagation on x/u nodes. Forecast deviations are
    defined only for x nodes; u-node initial deviations are zero. The returned
    score is computed over the propagated x-node deviations.
    """
    d_x = standardized_deviation(
        samples,
        probability_model.normal_target,
        probability_model.normal_scale,
    )
    n_x = d_x.shape[-1]
    n_nodes = probability_model.normalized_adjacency.shape[0]
    if n_nodes < n_x:
        raise ValueError("KG adjacency has fewer nodes than the X-variable deviation vector.")

    propagated = np.zeros(d_x.shape[:-1] + (n_nodes,), dtype=np.float64)
    propagated[..., :n_x] = d_x
    current = propagated.copy()

    for depth in range(1, probability_model.propagation_depth + 1):
        current = current @ probability_model.normalized_adjacency.T
        propagated += (probability_model.rho ** depth) * current

    r_x = propagated[..., :n_x]
    weights = probability_model.score_weights.reshape((1,) * (r_x.ndim - 1) + (-1,))
    return np.sum(weights * r_x ** 2, axis=-1)


def build_kg_probability_model(
    normal_stats: Dict[str, np.ndarray],
    adjacency: np.ndarray,
    n_x: int,
    propagation_depth: int,
    rho: float,
) -> KGProbabilityModel:
    if propagation_depth < 0:
        raise ValueError("kg_propagation_depth must be non-negative.")
    if not (0.0 <= rho < 1.0):
        raise ValueError("kg_rho must be in [0, 1).")
    return KGProbabilityModel(
        normalized_adjacency=normalize_kg_adjacency(adjacency),
        normal_target=normal_stats["x_target"],
        normal_scale=normal_stats["x_scale"],
        score_weights=np.ones(n_x, dtype=np.float64),
        propagation_depth=int(propagation_depth),
        rho=float(rho),
        control_limit=0.0,
    )


def set_kg_control_limit(
    probability_model: KGProbabilityModel,
    normal_scores: np.ndarray,
    alpha: float,
) -> KGProbabilityModel:
    if not (0.0 < alpha < 1.0):
        raise ValueError("kg_alpha must be between 0 and 1.")
    normal_scores = np.asarray(normal_scores, dtype=np.float64).reshape(-1)
    if normal_scores.ndim != 1 or normal_scores.size < 2 or not np.isfinite(normal_scores).all():
        raise ValueError("Normal KG monitoring score calibration produced invalid values.")
    probability_model.control_limit = float(np.quantile(normal_scores, 1.0 - alpha))
    if not np.isfinite(probability_model.control_limit):
        raise ValueError("KG monitoring control limit is not finite.")
    return probability_model


@torch.no_grad()
def forecast_distributions(
    model: nn.Module,
    x_norm: np.ndarray,
    u_norm: np.ndarray,
    x_scaler: FeatureScaler,
    history: int,
    horizon: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    start_origin: Optional[int] = None,
    end_origin: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_origin = history - 1
    max_origin = x_norm.shape[0] - horizon - 1
    start = min_origin if start_origin is None else int(start_origin)
    end = max_origin if end_origin is None else int(end_origin)
    if start < min_origin or end > max_origin or start > end:
        raise ValueError(
            f"Invalid origin range {start}..{end}; valid is {min_origin}..{max_origin}."
        )
    if stride < 1:
        raise ValueError("stride must be positive.")

    origins = np.arange(start, end + 1, stride, dtype=np.int64)

    mean_blocks: List[np.ndarray] = []
    std_blocks: List[np.ndarray] = []
    model.eval()

    for batch_start in range(0, origins.size, batch_size):
        batch_origins = origins[batch_start:batch_start + batch_size]
        x_hist = np.stack(
            [x_norm[k - history + 1:k + 1] for k in batch_origins.tolist()],
            axis=0,
        ).astype(np.float32)
        u_hist = np.stack(
            [u_norm[k - history + 1:k + 1] for k in batch_origins.tolist()],
            axis=0,
        ).astype(np.float32)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(
                torch.from_numpy(x_hist).to(device),
                torch.from_numpy(u_hist).to(device),
                horizon,
            )

        if "x_std" not in outputs:
            raise ValueError("The KG-STGM model must return outputs['x_std'].")

        mean_blocks.append(outputs["x_pred"].float().cpu().numpy())
        std_blocks.append(outputs["x_std"].float().cpu().numpy())

    mean_norm = np.concatenate(mean_blocks, axis=0).astype(np.float64)
    std_norm = np.concatenate(std_blocks, axis=0).astype(np.float64)

    if not np.isfinite(mean_norm).all() or not np.isfinite(std_norm).all():
        raise ValueError("Forecast mean/std contains NaN or Inf.")
    if not np.all(std_norm > 0.0):
        raise ValueError("Forecast standard deviations must be positive.")

    x_mean = x_scaler.mean.reshape(1, 1, -1).astype(np.float64)
    x_std = x_scaler.std.reshape(1, 1, -1).astype(np.float64)
    return origins, mean_norm * x_std + x_mean, std_norm * x_std


def calibrated_forecast_sigma(
    sigma_raw: np.ndarray,
    normal_stats: Dict[str, np.ndarray],
) -> np.ndarray:
    calibration_std = normal_stats.get("residual_std")
    if calibration_std is None:
        raise ValueError(
            "normal_operating_stats.npz must contain residual_std so global_risk.py "
            "uses the same uncertainty calibration as local_risk.py."
        )
    if sigma_raw.shape[-1] != calibration_std.shape[1]:
        raise ValueError("sigma_raw and residual_std variable dimensions do not match.")
    horizon = sigma_raw.shape[-2]
    if calibration_std.shape[0] < horizon:
        raise ValueError(
            f"normal residual_std has {calibration_std.shape[0]} horizons, "
            f"but forecast horizon is {horizon}."
        )
    broadcast_shape = (1,) * (sigma_raw.ndim - 2) + calibration_std[:horizon].shape
    calibration = calibration_std[:horizon].reshape(broadcast_shape)
    return np.sqrt(sigma_raw ** 2 + calibration ** 2)


def sample_forecast_scenarios(
    mean_raw: np.ndarray,
    std_raw: np.ndarray,
    mc_samples: int,
    rng: Optional[np.random.Generator] = None,
    innovations: Optional[np.ndarray] = None,
) -> np.ndarray:
    if mc_samples < 1:
        raise ValueError("mc_samples must be positive.")
    if mean_raw.shape != std_raw.shape:
        raise ValueError("mean_raw and std_raw must have the same shape.")
    if not np.all(std_raw > 0.0):
        raise ValueError("Calibrated forecast standard deviations must be positive.")
    expected_shape = (mc_samples, *mean_raw.shape)
    if innovations is None:
        if rng is None:
            raise ValueError("Provide rng or precomputed innovations.")
        innovations = rng.standard_normal(size=expected_shape)
    else:
        innovations = np.asarray(innovations, dtype=np.float64)
        if innovations.shape != expected_shape:
            raise ValueError(
                f"Innovations have shape {innovations.shape}; expected {expected_shape}."
            )
    samples = (
        mean_raw.reshape(1, *mean_raw.shape)
        + std_raw.reshape(1, *std_raw.shape) * innovations
    )
    if not np.isfinite(samples).all():
        raise ValueError("Forecast samples contain NaN or Inf.")
    return samples


@torch.no_grad()
def calibrate_kg_scores_from_normal_forecasts(
    model: nn.Module,
    normal_data_npz: str,
    x_scaler: FeatureScaler,
    u_scaler: FeatureScaler,
    normal_stats: Dict[str, np.ndarray],
    probability_model: KGProbabilityModel,
    history: int,
    horizon: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    mc_samples: int,
    mc_seed: int,
    max_origins: int,
    n_x: int,
) -> np.ndarray:
    """
    Calibrate J_alpha on the same object used at test time: KG-STGM normal
    forecast scenarios sampled as mean + calibrated sigma * standard normal
    innovations.
    """
    if mc_samples < 1:
        raise ValueError("kg calibration mc_samples must be positive.")
    x_raw_all, u_raw_all, _mapping = load_tep_arrays(normal_data_npz)
    x_raw_all = x_raw_all[:, :, :n_x].astype(np.float32)
    u_raw_all = u_raw_all[:, :, :u_scaler.mean.shape[0]].astype(np.float32)

    x_norm_all = x_scaler.transform(x_raw_all)
    u_norm_all = u_scaler.transform(u_raw_all)

    run_ids: List[int] = []
    origins: List[int] = []
    max_origin = x_raw_all.shape[1] - horizon - 1
    for run_id in range(x_raw_all.shape[0]):
        for origin in range(history - 1, max_origin + 1, stride):
            run_ids.append(run_id)
            origins.append(origin)
    if not origins:
        raise ValueError("No normal forecast windows are available for KG calibration.")

    if max_origins > 0 and len(origins) > max_origins:
        take = np.linspace(0, len(origins) - 1, max_origins, dtype=np.int64)
        run_ids = [run_ids[int(idx)] for idx in take]
        origins = [origins[int(idx)] for idx in take]

    score_blocks: List[np.ndarray] = []
    rng = np.random.default_rng(mc_seed)

    for start in range(0, len(origins), batch_size):
        batch_runs = run_ids[start:start + batch_size]
        batch_origins = origins[start:start + batch_size]
        x_hist = np.stack(
            [x_norm_all[r, k - history + 1:k + 1] for r, k in zip(batch_runs, batch_origins)],
            axis=0,
        ).astype(np.float32)
        u_hist = np.stack(
            [u_norm_all[r, k - history + 1:k + 1] for r, k in zip(batch_runs, batch_origins)],
            axis=0,
        ).astype(np.float32)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(
                torch.from_numpy(x_hist).to(device),
                torch.from_numpy(u_hist).to(device),
                horizon,
            )
        if "x_std" not in outputs:
            raise ValueError("The KG-STGM model must return outputs['x_std'].")

        mean_norm = outputs["x_pred"].float().cpu().numpy().astype(np.float64)
        std_norm = outputs["x_std"].float().cpu().numpy().astype(np.float64)
        x_mean = x_scaler.mean.reshape(1, 1, -1).astype(np.float64)
        x_std = x_scaler.std.reshape(1, 1, -1).astype(np.float64)
        mean_raw = mean_norm * x_std + x_mean
        std_raw = std_norm * x_std
        std_cal = calibrated_forecast_sigma(std_raw, normal_stats)

        for pos in range(mean_raw.shape[0]):
            samples = sample_forecast_scenarios(
                mean_raw=mean_raw[pos],
                std_raw=std_cal[pos],
                mc_samples=mc_samples,
                rng=rng,
            )
            score_blocks.append(kg_monitoring_score(samples, probability_model).reshape(-1))

    normal_scores = np.concatenate(score_blocks, axis=0)
    if normal_scores.size < 2 or not np.isfinite(normal_scores).all():
        raise ValueError("Normal forecast KG calibration produced invalid scores.")
    return normal_scores


def process_risk_components(
    samples: np.ndarray,
    normal_stats: Dict[str, np.ndarray],
    severity_model: SeverityModel,
    probability_model: KGProbabilityModel,
    safe_limits: Optional[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """
    samples: shape (M, H, N_x)
    """
    graph_score = kg_monitoring_score(samples, probability_model)
    process_abnormal = graph_score > probability_model.control_limit

    process_unsafe: Optional[np.ndarray] = None
    if safe_limits is not None:
        safe_low = safe_limits["x_low"].reshape(1, 1, -1)
        safe_high = safe_limits["x_high"].reshape(1, 1, -1)
        process_unsafe = np.any((samples < safe_low) | (samples > safe_high), axis=-1)

    variable_severity = variable_severity_index(
        samples=samples,
        models=severity_model.variable_models,
        normal_low=normal_stats["x_low"],
        normal_high=normal_stats["x_high"],
        safe_limits=safe_limits,
    )

    # Unit consequence: EML_unit * max variable severity inside the unit.
    unit_consequence = np.zeros(
        samples.shape[:-1] + (len(TEP_UNIT_NAMES),),
        dtype=np.float64,
    )
    for unit_idx in range(len(TEP_UNIT_NAMES)):
        mask = severity_model.x_unit_index == unit_idx
        if not np.any(mask):
            raise ValueError(f"No measured variables are assigned to unit {TEP_UNIT_NAMES[unit_idx]}.")
        unit_consequence[..., unit_idx] = (
            severity_model.unit_eml[unit_idx] * np.max(variable_severity[..., mask], axis=-1)
        )

    process_consequence = np.sum(unit_consequence, axis=-1)
    process_probability = process_abnormal.mean(axis=0)
    mean_graph_score = graph_score.mean(axis=0)
    mean_consequence = process_consequence.mean(axis=0)
    expected_event_consequence = (process_consequence * process_abnormal).mean(axis=0)
    conditional_cons = conditional_consequence(expected_event_consequence, process_probability)
    process_risk = process_probability * conditional_cons

    if not np.allclose(
        expected_event_consequence,
        process_risk,
        rtol=1e-12,
        atol=1e-9,
    ):
        raise RuntimeError("KG probability risk identity R=P_g*E[C_g|A_g] failed.")

    result = {
        "process_probability": process_probability,
        "mean_consequence": mean_consequence,
        "conditional_consequence": conditional_cons,
        "risk": process_risk,
        "mean_graph_score": mean_graph_score,
    }
    if process_unsafe is not None:
        result["unsafe_probability"] = process_unsafe.mean(axis=0)
    return result


def compute_global_risk_series(
    fault_id: int,
    origins: np.ndarray,
    mean_raw: np.ndarray,
    std_raw: np.ndarray,
    normal_stats: Dict[str, np.ndarray],
    severity_model: SeverityModel,
    probability_model: KGProbabilityModel,
    safe_limits: Optional[Dict[str, np.ndarray]],
    mc_samples: int,
    mc_seed: int,
) -> Dict[str, np.ndarray]:
    """
    Compute global risk for every forecast origin.

    For each origin, risk is first computed separately at every forecast horizon
    as P_g(t,h) * E[C_g(t,h) | A_g(t,h)]. The unconditional E[C_g(t,h)] is also
    retained as the stable severity diagnostic shown in the middle plot. The
    proactive/global risk is the maximum of the per-horizon risk values.
    """
    n_origins, horizon, _n_x = mean_raw.shape
    if std_raw.shape != mean_raw.shape:
        raise ValueError("mean_raw and std_raw must have identical shapes.")

    process_probability = np.zeros((n_origins, horizon), dtype=np.float64)
    mean_consequence = np.zeros((n_origins, horizon), dtype=np.float64)
    conditional_cons = np.zeros((n_origins, horizon), dtype=np.float64)
    risk = np.zeros((n_origins, horizon), dtype=np.float64)
    mean_graph_score = np.zeros((n_origins, horizon), dtype=np.float64)
    unsafe_probability = (
        np.zeros((n_origins, horizon), dtype=np.float64)
        if safe_limits is not None
        else None
    )

    max_risk = np.zeros(n_origins, dtype=np.float64)
    max_risk_horizon = np.zeros(n_origins, dtype=np.int64)
    probability_at_max_risk = np.zeros(n_origins, dtype=np.float64)
    consequence_at_max_risk = np.zeros(n_origins, dtype=np.float64)

    innovation_rng = np.random.default_rng(mc_seed + 1009 * fault_id)
    common_innovations = innovation_rng.standard_normal(
        size=(mc_samples, horizon, mean_raw.shape[-1])
    )

    for pos, origin in enumerate(origins.tolist()):
        samples = sample_forecast_scenarios(
            mean_raw=mean_raw[pos],
            std_raw=std_raw[pos],
            mc_samples=mc_samples,
            innovations=common_innovations,
        )

        components = process_risk_components(
            samples=samples,
            normal_stats=normal_stats,
            severity_model=severity_model,
            probability_model=probability_model,
            safe_limits=safe_limits,
        )

        process_probability[pos] = components["process_probability"]
        mean_consequence[pos] = components["mean_consequence"]
        conditional_cons[pos] = components["conditional_consequence"]
        risk[pos] = components["risk"]
        mean_graph_score[pos] = components["mean_graph_score"]
        if unsafe_probability is not None:
            unsafe_probability[pos] = components["unsafe_probability"]

        max_idx = int(np.argmax(risk[pos]))
        max_risk_horizon[pos] = max_idx + 1
        probability_at_max_risk[pos] = process_probability[pos, max_idx]
        consequence_at_max_risk[pos] = conditional_cons[pos, max_idx]
        max_risk[pos] = risk[pos, max_idx]

    if not np.allclose(max_risk, np.max(risk, axis=1), rtol=1e-12, atol=1e-9):
        raise RuntimeError("Max-horizon risk must be max_h risk(t,h).")
    if not np.allclose(
        max_risk,
        probability_at_max_risk * consequence_at_max_risk,
        rtol=1e-12,
        atol=1e-9,
    ):
        raise RuntimeError("Max-risk probability-consequence-risk identity failed.")

    result = {
        "origins": origins,
        "time_first": origins + 1,

        # 1-step quantities
        "process_probability_first": process_probability[:, 0],
        "mean_consequence_first": mean_consequence[:, 0],
        "conditional_consequence_first": conditional_cons[:, 0],
        "risk_first": risk[:, 0],
        "mean_graph_score_first": mean_graph_score[:, 0],
        "kg_control_limit": float(probability_model.control_limit),

        # Values at h* = argmax_h R_g(t,h), where R_g(t,h) is computed per horizon.
        "probability_at_max_risk": probability_at_max_risk,
        "conditional_consequence_at_max_risk": consequence_at_max_risk,
        "max_horizon_risk": max_risk,

        # timing diagnostic
        "max_risk_horizon": max_risk_horizon,

        # backward-compatible names for old plotting or post-processing
        "probability_first": process_probability[:, 0],
        "severity_first": mean_consequence[:, 0],
        "process_risk": risk[:, 0],
        "proactive_probability": probability_at_max_risk,
        "proactive_conditional_consequence": consequence_at_max_risk,
        "proactive_risk": max_risk,
        "proactive_horizon": max_risk_horizon,
    }

    if unsafe_probability is not None:
        result["unsafe_probability_first"] = unsafe_probability[:, 0]

    return result


def validate_optional_threshold(value: Optional[float], label: str) -> Optional[float]:
    if value is None:
        return None
    threshold = float(value)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return threshold


def first_threshold_crossing_time(
    times: np.ndarray,
    values: np.ndarray,
    threshold: Optional[float],
) -> Optional[int]:
    if threshold is None:
        return None
    crossed = np.asarray(values, dtype=np.float64) >= float(threshold)
    if not np.any(crossed):
        return None
    return int(np.asarray(times, dtype=np.int64)[int(np.argmax(crossed))])


def format_crossing_time(value: Optional[int]) -> str:
    return "not crossed" if value is None else str(int(value))


def format_lead_time(max_time: Optional[int], one_step_time: Optional[int]) -> str:
    if max_time is None or one_step_time is None:
        return "n/a"
    return str(int(one_step_time) - int(max_time))


def lead_time_value(max_time: Optional[int], one_step_time: Optional[int]) -> Optional[int]:
    if max_time is None or one_step_time is None:
        return None
    return int(one_step_time) - int(max_time)


def plot_global_risk(
    result: Dict[str, np.ndarray],
    fault_id: int,
    fault_label: str,
    out_path: str,
    fault_intro_time: Optional[int],
    alarm_threshold: Optional[float],
    esh_threshold: Optional[float],
    plot_start_time: int,
    plot_end_time: Optional[int],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = result["time_first"]

    fig, axes = plt.subplots(1, 3, figsize=(22.0, 6.3), sharex=False)

    axes[0].plot(
        t,
        result["process_probability_first"],
        color="tab:green",
        linewidth=2.0,
        label="1-step ahead",
    )
    if "unsafe_probability_first" in result:
        axes[0].plot(
            t,
            result["unsafe_probability_first"],
            color="#b91c1c",
            linewidth=1.8,
            label="Unsafe",
        )
    axes[0].set_ylabel("Probability", fontsize=14)
    axes[0].set_xlabel("Time Sample", fontsize=14)
    axes[0].set_ylim(-0.02, 1.02)

    axes[1].plot(
        t,
        result["mean_consequence_first"],
        color="tab:orange",
        linewidth=2.0,
        label="1-step ahead",
    )
    axes[1].set_ylabel("Severity ($/h)", fontsize=14)
    axes[1].set_xlabel("Time Sample", fontsize=14)

    axes[2].plot(
        t,
        result["risk_first"],
        color="tab:blue",
        linewidth=2.0,
        label="1-step Risk",
    )
    axes[2].plot(
        t,
        result["max_horizon_risk"],
        color="tab:red",
        linewidth=2.0,
        label="Max Risk",
    )
    if alarm_threshold is not None:
        axes[2].axhline(
            alarm_threshold,
            color="tab:orange",
            linestyle="--",
            linewidth=2.0,
            label="Alarm",
        )
    if esh_threshold is not None:
        axes[2].axhline(
            esh_threshold,
            color="#991b1b",
            linestyle="--",
            linewidth=2.0,
            label="ESH",
        )
    axes[2].set_ylabel("Risk ($/h)", fontsize=14)
    axes[2].set_xlabel("Time Sample", fontsize=14)

    for axis in axes:
        if fault_intro_time is not None:
            axis.axvline(
                fault_intro_time,
                color="#d62728",
                linestyle=":",
                lw=1.8,
                alpha=0.95,
                zorder=5,
                label="Fault Introduced",
            )
        axis.grid(True, alpha=0.25)
        axis.tick_params(axis="both", labelsize=12)
        x_end = int(t[-1]) if plot_end_time is None else min(int(plot_end_time), int(t[-1]))
        if int(t[0]) == int(t[-1]):
            axis.set_xlim(int(plot_start_time), max(int(plot_start_time) + 1, int(t[-1]) + 1))
        else:
            axis.set_xlim(int(plot_start_time), x_end)
        if x_end > int(plot_start_time):
            axis.set_xticks(np.linspace(int(plot_start_time), x_end, 6, dtype=int))
        axis.legend(
            loc="best",
            fontsize=14,
            framealpha=0.94,
            ncol=1,
        )

    fig.tight_layout(pad=1.5, h_pad=1.5, w_pad=1.5)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_kg_score_diagnostic(
    result: Dict[str, np.ndarray],
    out_path: str,
    fault_intro_time: Optional[int],
    plot_start_time: int,
    plot_end_time: Optional[int],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = result["time_first"]
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 5.0), sharex=False)
    ax.plot(
        t,
        result["mean_graph_score_first"],
        color="tab:purple",
        linewidth=2.0,
        label="1-step J_K",
    )
    ax.axhline(
        result["kg_control_limit"],
        color="#991b1b",
        linestyle="--",
        linewidth=2.0,
        label="J_alpha",
    )
    if fault_intro_time is not None:
        ax.axvline(
            fault_intro_time,
            color="#d62728",
            linestyle=":",
            lw=1.8,
            alpha=0.95,
            zorder=5,
            label="Fault Introduced",
        )
    ax.set_ylabel("KG Score", fontsize=14)
    ax.set_xlabel("Time Sample", fontsize=14)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=12)
    x_end = int(t[-1]) if plot_end_time is None else min(int(plot_end_time), int(t[-1]))
    if int(t[0]) == int(t[-1]):
        ax.set_xlim(int(plot_start_time), max(int(plot_start_time) + 1, int(t[-1]) + 1))
    else:
        ax.set_xlim(int(plot_start_time), x_end)
    if x_end > int(plot_start_time):
        ax.set_xticks(np.linspace(int(plot_start_time), x_end, 6, dtype=int))
    ax.legend(loc="best", fontsize=14, framealpha=0.94, ncol=1)
    fig.tight_layout(pad=1.5, h_pad=1.5, w_pad=1.5)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dependency-preserving global process risk assessment for KG-STGM TEP forecasts."
    )
    parser.add_argument("--out_dir", type=str, default="kg_stgm_outputs")
    parser.add_argument("--data_npz", type=str, default="dataverse_files/TEP_Faulty_Testing.npz")
    parser.add_argument("--normal_data_npz", type=str, default="dataverse_files/TEP_FaultFree_Testing.npz")
    parser.add_argument("--normal_stats", type=str, default=None)
    parser.add_argument("--severity_models", type=str, default=None)
    parser.add_argument("--safe_limits", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sim_id", type=int, default=1)
    parser.add_argument("--fault_ids", type=str, default="1")
    parser.add_argument("--test_stride", type=int, default=1)
    parser.add_argument("--start_origin", type=int, default=None)
    parser.add_argument("--end_origin", type=int, default=499)
    parser.add_argument("--plot_start_time", type=int, default=0)
    parser.add_argument("--plot_end_time", type=int, default=500)
    parser.add_argument("--fault_intro_time", type=int, default=160)
    parser.add_argument("--mc_samples", type=int, default=2000)
    parser.add_argument("--mc_seed", type=int, default=42)
    parser.add_argument(
        "--kg_alpha",
        type=float,
        default=0.005,
        help=(
            "Upper-tail false-alarm level for the KG graph monitoring score control limit. "
            "Default 0.005 corresponds to the validation-selected 99.5th percentile."
        ),
    )
    parser.add_argument(
        "--kg_propagation_depth",
        type=int,
        default=2,
        help=(
            "Maximum KG propagation depth K_p. Default 2 includes direct and "
            "second-hop KG neighborhoods while avoiding whole-graph diffusion."
        ),
    )
    parser.add_argument(
        "--kg_rho",
        type=float,
        default=0.5,
        help=(
            "Downstream KG propagation decay factor rho in [0, 1). Default 0.5 "
            "moderately weights direct neighbors and down-weights second-hop effects."
        ),
    )
    parser.add_argument(
        "--kg_calibration_max_origins",
        type=int,
        default=400,
        help="Maximum normal forecast origins for KG control-limit calibration; 0 uses all origins.",
    )
    parser.add_argument(
        "--kg_calibration_mc_samples",
        type=int,
        default=0,
        help="MC scenarios per normal forecast origin for KG calibration; 0 uses --mc_samples.",
    )
    parser.add_argument(
        "--kg_calibration_stride",
        type=int,
        default=0,
        help="Normal forecast-origin stride for KG calibration; 0 uses the forecast horizon.",
    )
    parser.add_argument(
        "--lambda_time",
        type=float,
        default=0.0,
        help="Deprecated; must remain 0 because risk is max_h P_h*C_h.",
    )
    parser.add_argument(
        "--unit_eml",
        type=float,
        default=None,
        help="Optional override: use the same EML for every unit.",
    )
    parser.add_argument(
        "--unit_eml_values",
        type=str,
        default=None,
        help=(
            "Optional comma-separated EML values in reactor,condenser,separator,"
            "compressor,stripper order."
        ),
    )
    parser.add_argument("--alarm_threshold", type=float, default=70000.0)
    parser.add_argument("--esh_threshold", type=float, default=80000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = resolve_project_path(args.out_dir)
    data_npz_path = resolve_project_path(args.data_npz)
    normal_data_npz_path = resolve_project_path(args.normal_data_npz)
    config_path = resolve_project_path(args.config) if args.config else os.path.join(out_dir, "config.json")
    output_dir = (
        resolve_project_path(args.output_dir)
        if args.output_dir
        else os.path.join(out_dir, "global_risk")
    )
    normal_stats_path = (
        resolve_project_path(args.normal_stats)
        if args.normal_stats
        else os.path.join(out_dir, "normal_operating_stats.npz")
    )
    severity_models_path = resolve_project_path(args.severity_models) if args.severity_models else os.path.join(
        out_dir,
        "data_fitting_parameters.json",
    )
    os.makedirs(output_dir, exist_ok=True)

    cfg = load_train_config(config_path)
    cfg.batch_size = int(args.batch_size)
    cfg.device = args.device
    set_seed(cfg.seed)

    device = resolve_device(cfg.device)
    use_amp = bool(cfg.amp) and device.type != "cpu"
    amp_dtype = (
        torch.float16
        if device.type == "mps" or cfg.amp_dtype == "float16"
        else torch.bfloat16
    )

    x_scaler, u_scaler = load_saved_scalers(os.path.join(out_dir, "scaler_stats.npz"))
    n_x = int(x_scaler.mean.shape[0])
    n_u = int(u_scaler.mean.shape[0])
    if n_x != 22:
        raise ValueError(f"Global TEP risk expects 22 X variables, got {n_x}.")

    adjacency = load_adjacency(
        os.path.join(out_dir, "adjacency.npy"),
        n_nodes=n_x + n_u,
    )
    kg_propagation_depth = int(args.kg_propagation_depth)
    kg_depth_source = "configured"
    model = build_model(cfg, adjacency, n_x, n_u, device)
    model.load_state_dict(
        load_state_dict_file(resolve_checkpoint(out_dir, args.checkpoint), device)
    )

    normal_stats = load_normal_operating_stats(normal_stats_path, n_x)
    variable_models = load_distribution_models(severity_models_path, n_x)
    kg_alpha = float(args.kg_alpha)
    kg_alpha_source = "configured"
    kg_rho = float(args.kg_rho)
    kg_rho_source = "configured"
    probability_model = build_kg_probability_model(
        normal_stats=normal_stats,
        adjacency=adjacency,
        n_x=n_x,
        propagation_depth=kg_propagation_depth,
        rho=kg_rho,
    )
    kg_calibration_stride = (
        cfg.horizon
        if int(args.kg_calibration_stride) <= 0
        else max(1, int(args.kg_calibration_stride))
    )
    kg_calibration_mc_samples = (
        int(args.mc_samples)
        if int(args.kg_calibration_mc_samples) <= 0
        else int(args.kg_calibration_mc_samples)
    )
    normal_scores = calibrate_kg_scores_from_normal_forecasts(
        model=model,
        normal_data_npz=normal_data_npz_path,
        x_scaler=x_scaler,
        u_scaler=u_scaler,
        normal_stats=normal_stats,
        probability_model=probability_model,
        history=cfg.history,
        horizon=cfg.horizon,
        stride=kg_calibration_stride,
        batch_size=cfg.batch_size,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        mc_samples=kg_calibration_mc_samples,
        mc_seed=int(args.mc_seed) + 1729,
        max_origins=int(args.kg_calibration_max_origins),
        n_x=n_x,
    )
    probability_model = set_kg_control_limit(
        probability_model=probability_model,
        normal_scores=normal_scores,
        alpha=kg_alpha,
    )

    unit_values = parse_unit_eml_values(args.unit_eml_values)
    if unit_values is not None:
        unit_eml = unit_values
    elif args.unit_eml is not None:
        unit_eml = np.full(len(TEP_UNIT_NAMES), float(args.unit_eml), dtype=np.float64)
    else:
        unit_eml = default_unit_eml_array()

    severity_model = build_severity_model(variable_models, unit_eml)

    safe_limits = load_safe_operating_limits(
        resolve_project_path(args.safe_limits),
        normal_stats["x_low"],
        normal_stats["x_high"],
    )

    alarm_threshold = validate_optional_threshold(args.alarm_threshold, "alarm_threshold")
    esh_threshold = validate_optional_threshold(args.esh_threshold, "esh_threshold")
    if alarm_threshold is not None and esh_threshold is not None and alarm_threshold >= esh_threshold:
        raise ValueError("alarm_threshold must be lower than esh_threshold.")
    if float(args.lambda_time) != 0.0:
        raise ValueError("--lambda_time is deprecated; max-horizon risk does not use time weighting.")

    data = np.load(data_npz_path, allow_pickle=True)
    available_fault_ids, fault_name_by_id = load_faulty_metadata(data)
    requested_fault_ids = parse_int_ranges(args.fault_ids, "fault IDs")
    missing = [fault_id for fault_id in requested_fault_ids if fault_id not in available_fault_ids]
    if missing:
        raise ValueError(f"Requested fault IDs not present in {data_npz_path}: {missing}")

    stride = cfg.horizon if int(args.test_stride) <= 0 else max(1, int(args.test_stride))
    forecast_mode = "non-overlapping" if stride == cfg.horizon else "sliding-window"

    print(
        f"Global process risk: {len(requested_fault_ids)} fault(s), sim_id={args.sim_id}, "
        f"{forecast_mode} stride={stride}, M={args.mc_samples}, "
        f"risk=max_h P_h*C_h"
    )
    print(f"Normal stats: {normal_stats_path}")
    print(f"Severity distribution models: {severity_models_path}")
    print(
        "KG probability score: "
        f"alpha={kg_alpha:.6g} ({kg_alpha_source}), "
        f"Kp={kg_propagation_depth} ({kg_depth_source}), "
        f"rho={kg_rho:.6g} ({kg_rho_source}), "
        f"J_alpha={probability_model.control_limit:.6g}"
    )
    print(
        "KG control-limit calibration: "
        f"normal forecast scenarios from {normal_data_npz_path}, "
        f"{normal_scores.size} scores, stride={kg_calibration_stride}, "
        f"M_cal={kg_calibration_mc_samples}"
    )
    print("Forecast scenarios: KG-STGM mean + calibrated sigma * standard normal innovations")
    if safe_limits is None:
        print("Safe limits: not supplied; output is operational-risk index.")
    else:
        print("Safe limits: supplied; unsafe probability and safe-boundary severity enabled.")
    print("Unit EML assumptions:")
    for unit_name, eml in zip(TEP_UNIT_NAMES, severity_model.unit_eml.tolist()):
        print(f"  {unit_name}: ${eml:,.0f}/h")
    loss_family_counts = Counter(model.dist_name for model in severity_model.variable_models)
    print(f"Fitted severity marginal families: {dict(sorted(loss_family_counts.items()))}")

    crossing_summaries: List[Dict[str, object]] = []

    for fault_id in requested_fault_ids:
        x_raw, u_raw = load_faulty_run(data, fault_id, args.sim_id, n_x=n_x, n_u=n_u)

        origins, mean_raw, std_raw = forecast_distributions(
            model=model,
            x_norm=x_scaler.transform(x_raw),
            u_norm=u_scaler.transform(u_raw),
            x_scaler=x_scaler,
            history=cfg.history,
            horizon=cfg.horizon,
            stride=stride,
            batch_size=cfg.batch_size,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            start_origin=args.start_origin,
            end_origin=args.end_origin,
        )
        std_cal = calibrated_forecast_sigma(std_raw, normal_stats)

        print(
            f"  [fault {fault_id:2d}] MC risk evaluation: "
            f"{origins.size} origins x {int(args.mc_samples)} samples x {cfg.horizon} horizons",
            flush=True,
        )
        result = compute_global_risk_series(
            fault_id=fault_id,
            origins=origins,
            mean_raw=mean_raw,
            std_raw=std_cal,
            normal_stats=normal_stats,
            severity_model=severity_model,
            probability_model=probability_model,
            safe_limits=safe_limits,
            mc_samples=int(args.mc_samples),
            mc_seed=int(args.mc_seed),
        )

        fault_dir = os.path.join(output_dir, f"fault{fault_id}")
        os.makedirs(fault_dir, exist_ok=True)

        fault_label = fault_name_by_id.get(fault_id, f"Fault {fault_id}")
        tag = f"fault{fault_id}_sim{args.sim_id}_stride{stride}"
        plot_path = os.path.join(fault_dir, f"global_risk_{tag}.png")
        score_plot_path = os.path.join(fault_dir, f"kg_score_{tag}.png")

        plot_global_risk(
            result=result,
            fault_id=fault_id,
            fault_label=fault_label,
            out_path=plot_path,
            fault_intro_time=args.fault_intro_time,
            alarm_threshold=alarm_threshold,
            esh_threshold=esh_threshold,
            plot_start_time=int(args.plot_start_time),
            plot_end_time=args.plot_end_time,
        )
        plot_kg_score_diagnostic(
            result=result,
            out_path=score_plot_path,
            fault_intro_time=args.fault_intro_time,
            plot_start_time=int(args.plot_start_time),
            plot_end_time=args.plot_end_time,
        )

        peak_origin = int(np.argmax(result["max_horizon_risk"]))
        print(
            f"  [fault {fault_id:2d}] max forecast-horizon expected risk="
            f"{result['max_horizon_risk'][peak_origin]:.4f} "
            f"at origin={origins[peak_origin]}, forecast horizon="
            f"{result['max_risk_horizon'][peak_origin]}"
        )
        fault_crossing_summary: Dict[str, object] = {
            "fault_id": int(fault_id),
            "fault_label": fault_label,
            "sim_id": int(args.sim_id),
            "stride": int(stride),
            "risk_definition": "max_horizon_risk = max_h P_g(t,h) * severity(t,h)",
            "time_axis": "time_first",
            "fault_intro_time": int(args.fault_intro_time),
            "peak_max_horizon_risk": float(result["max_horizon_risk"][peak_origin]),
            "peak_max_horizon_risk_time": int(origins[peak_origin]),
            "peak_max_risk_horizon": int(result["max_risk_horizon"][peak_origin]),
            "thresholds": [],
        }
        for threshold_label, threshold in (
            ("Alarm", alarm_threshold),
            ("ESH", esh_threshold),
        ):
            if threshold is None:
                continue
            max_cross_time = first_threshold_crossing_time(
                result["time_first"],
                result["max_horizon_risk"],
                threshold,
            )
            one_step_cross_time = first_threshold_crossing_time(
                result["time_first"],
                result["risk_first"],
                threshold,
            )
            lead = lead_time_value(max_cross_time, one_step_cross_time)
            threshold_summary = {
                "label": threshold_label,
                "threshold": float(threshold),
                "max_risk_cross_time": None if max_cross_time is None else int(max_cross_time),
                "one_step_risk_cross_time": None if one_step_cross_time is None else int(one_step_cross_time),
                "lead_time_samples": lead,
                "lead_time_definition": "one_step_risk_cross_time - max_risk_cross_time",
            }
            fault_crossing_summary["thresholds"].append(threshold_summary)
            print(
                f"  [fault {fault_id:2d}] {threshold_label} threshold={threshold:.0f}: "
                f"Max Risk crosses at t={format_crossing_time(max_cross_time)}, "
                f"1-step Risk crosses at t={format_crossing_time(one_step_cross_time)}, "
                f"lead(1-step - max)={format_lead_time(max_cross_time, one_step_cross_time)} samples"
            )
        threshold_json_path = os.path.join(fault_dir, f"threshold_crossings_{tag}.json")
        with open(threshold_json_path, "w", encoding="utf-8") as f:
            json.dump(fault_crossing_summary, f, indent=2)
        crossing_summaries.append(fault_crossing_summary)
        print(f"  [fault {fault_id:2d}] saved: {plot_path}")
        print(f"  [fault {fault_id:2d}] saved: {score_plot_path}")
        print(f"  [fault {fault_id:2d}] saved: {threshold_json_path}")

    combined_crossing_path = os.path.join(
        output_dir,
        f"threshold_crossings_faults_{'_'.join(str(int(v)) for v in requested_fault_ids)}_sim{args.sim_id}_stride{stride}.json",
    )
    with open(combined_crossing_path, "w", encoding="utf-8") as f:
        json.dump(crossing_summaries, f, indent=2)
    print(f"Saved combined threshold crossing summary: {combined_crossing_path}")


if __name__ == "__main__":
    main()
