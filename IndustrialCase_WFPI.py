"""
IndustrialCase_WFPI.py

Weak first-principles prediction interval construction.
The function accepts three splits (train, validation, test-like) and returns
intervals for the three corresponding splits. The third split may also be an
unlabeled pool when needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


@dataclass
class WFPIConfig:
    tmpd_idx: int = 0
    tppd_idx: int = 1
    d_idx: int = 2
    b_idx: int = 3
    f_idx: int = 4
    min_ppd_ppm: float = 5.0
    max_ppd_ppm: float = 200.0
    eps: float = 1e-6
    pc_percentiles_if_unlabeled: Tuple[float, float] = (5.0, 95.0)
    sd_percentiles_if_unlabeled: Tuple[float, float] = (5.0, 95.0)


def _last_step_feature(X: np.ndarray, idx: int) -> np.ndarray:
    idx = min(idx, X.shape[2] - 1)
    return np.clip(X[:, -1, idx], 1e-6, None)


def _clip_intervals(L: np.ndarray, U: np.ndarray, lo: float, hi: float, eps: float) -> np.ndarray:
    L = np.clip(L, lo, hi)
    U = np.clip(U, lo, hi)
    U = np.maximum(U, L + eps)
    U = np.clip(U, lo, hi)
    L = np.minimum(L, U - eps)
    L = np.clip(L, lo, hi)
    return np.column_stack([L, U]).astype(np.float32)


def _normalize_ppm(values_ppm: np.ndarray, cfg: WFPIConfig) -> np.ndarray:
    denom = max(cfg.max_ppd_ppm - cfg.min_ppd_ppm, cfg.eps)
    norm = (values_ppm - cfg.min_ppd_ppm) / denom
    return np.clip(norm, 0.0, 1.0)


def _clip_normalized_intervals(L_ppm: np.ndarray, U_ppm: np.ndarray, cfg: WFPIConfig) -> np.ndarray:
    L_ppm = np.minimum(L_ppm, U_ppm)
    U_ppm = np.maximum(L_ppm, U_ppm)
    L = _normalize_ppm(L_ppm, cfg)
    U = _normalize_ppm(U_ppm, cfg)
    return _clip_intervals(L, U, 0.0, 1.0, cfg.eps)


def _proxy_label_from_inputs(X: np.ndarray, cfg: WFPIConfig) -> np.ndarray:
    tmpd = _last_step_feature(X, cfg.tmpd_idx)
    tppd = _last_step_feature(X, cfg.tppd_idx)
    D = _last_step_feature(X, cfg.d_idx)
    F = np.maximum(_last_step_feature(X, cfg.f_idx), cfg.eps)
    proxy_norm = 0.45 * tmpd + 0.35 * (1.0 - tppd) + 0.20 * (D / F)
    proxy_norm = np.clip(proxy_norm, cfg.eps, 1.0 - cfg.eps)
    return cfg.min_ppd_ppm + proxy_norm * (cfg.max_ppd_ppm - cfg.min_ppd_ppm)


def build_wfpi(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test_like: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    config: Optional[WFPIConfig] = None,
) -> Dict[str, Any]:
    if config is None:
        config = WFPIConfig()

    X_train = _to_numpy(X_train).astype(np.float32)
    X_val = _to_numpy(X_val).astype(np.float32)
    X_test_like = _to_numpy(X_test_like).astype(np.float32)

    X_cal = np.concatenate([X_train, X_val], axis=0)
    if y_train is not None and y_val is not None:
        y_cal = np.concatenate([_to_numpy(y_train).reshape(-1), _to_numpy(y_val).reshape(-1)], axis=0).astype(np.float32)
        y_cal = np.clip(y_cal, config.min_ppd_ppm + config.eps, config.max_ppd_ppm - config.eps)
    else:
        y_cal = _proxy_label_from_inputs(X_cal, config)

    tmpd_cal = _last_step_feature(X_cal, config.tmpd_idx)
    tppd_cal = _last_step_feature(X_cal, config.tppd_idx)
    D_cal = _last_step_feature(X_cal, config.d_idx)
    B_cal = np.maximum(_last_step_feature(X_cal, config.b_idx), config.eps)
    F_cal = np.maximum(_last_step_feature(X_cal, config.f_idx), config.eps)

    sd = -np.log(tmpd_cal + config.eps) - np.log(y_cal + config.eps)
    if y_train is not None and y_val is not None:
        sd_min = float(np.min(sd))
        sd_max = float(np.max(sd))
    else:
        ql, qu = config.sd_percentiles_if_unlabeled
        sd_min, sd_max = np.percentile(sd, [ql, qu])

    pc = (tppd_cal * D_cal + y_cal * B_cal) / F_cal
    if y_train is not None and y_val is not None:
        pc_min = float(np.min(pc))
        pc_max = float(np.max(pc))
    else:
        ql, qu = config.pc_percentiles_if_unlabeled
        pc_min, pc_max = np.percentile(pc, [ql, qu])

    def build_for_split(X: np.ndarray) -> np.ndarray:
        tmpd = _last_step_feature(X, config.tmpd_idx)
        tppd = _last_step_feature(X, config.tppd_idx)
        D = _last_step_feature(X, config.d_idx)
        B = np.maximum(_last_step_feature(X, config.b_idx), config.eps)
        F = np.maximum(_last_step_feature(X, config.f_idx), config.eps)

        L_sd = np.exp(-np.log(tmpd + config.eps) - sd_max)
        U_sd = np.exp(-np.log(tmpd + config.eps) - sd_min)
        L_pc = (pc_min * F - tppd * D) / B
        U_pc = (pc_max * F - tppd * D) / B

        L = np.maximum(L_sd, L_pc)
        U = np.minimum(U_sd, U_pc)
        return _clip_normalized_intervals(L, U, config)

    return {
        "train": build_for_split(X_train),
        "val": build_for_split(X_val),
        "test": build_for_split(X_test_like),
        "metadata": {
            "sd_min": sd_min,
            "sd_max": sd_max,
            "pc_min": pc_min,
            "pc_max": pc_max,
        },
    }
