"""
IndustrialCase_SFPI.py

Strong first-principles prediction interval construction.
The third split may be a test set or an unlabeled pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.interpolate import CubicSpline
except Exception:  # pragma: no cover
    CubicSpline = None


REQUIRED_TAGS: Tuple[str, ...] = (
    "FI35141.PV",
    "FI35144.PV",
    "FI35146.PV",
    "PI35144.PV",
    "PI35141a9.PV",
    "RAMAN_T35141_MPD.PV",
    "RAMAN_T35141_OPD.PV",
    "RAMAN_T35141_PPD.PV",
    "TI35141a1.PV",
    "TI35141a2.PV",
    "TI35141a3.PV",
    "TI35141a4.PV",
    "TI35141a5.PV",
    "TI35141a6.PV",
    "TI35141a7.PV",
    "TI35141a8.PV",
    "TI35141a9.PV",
)


@dataclass
class SFPIConfig:
    tag_names: Optional[Sequence[str]] = None
    center_total_stages: int = 51
    upper_total_stages: int = 47
    lower_total_stages: int = 54
    min_ppd_ppm: float = 5.0
    max_ppd_ppm: float = 200.0
    eps: float = 1e-6

    # Fallback-mode parameters (used only when industrial tags are unavailable)
    top_idx: int = 1
    stage_scale: float = 0.04
    nonlinearity_power: float = 1.25


# -------------------------
# Common helpers
# -------------------------
def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _clip_intervals(L: np.ndarray, U: np.ndarray, eps: float) -> np.ndarray:
    L = np.clip(np.asarray(L, dtype=np.float64), 0.0, 1.0)
    U = np.clip(np.asarray(U, dtype=np.float64), 0.0, 1.0)
    low = np.minimum(L, U)
    up = np.maximum(L, U)
    up = np.maximum(up, low + eps)
    up = np.clip(up, 0.0, 1.0)
    low = np.minimum(low, up - eps)
    low = np.clip(low, 0.0, 1.0)
    return np.column_stack([low, up]).astype(np.float32)


def _normalize_ppm(values_ppm: np.ndarray, cfg: SFPIConfig) -> np.ndarray:
    values_ppm = np.asarray(values_ppm, dtype=np.float64)
    denom = max(cfg.max_ppd_ppm - cfg.min_ppd_ppm, cfg.eps)
    norm = (values_ppm - cfg.min_ppd_ppm) / denom
    return np.clip(norm, 0.0, 1.0)


def _normalize_interval_ppm(low_ppm: np.ndarray, up_ppm: np.ndarray, cfg: SFPIConfig) -> np.ndarray:
    low = _normalize_ppm(low_ppm, cfg)
    up = _normalize_ppm(up_ppm, cfg)
    return _clip_intervals(low, up, cfg.eps)


def _last_step_feature(X: np.ndarray, idx: int) -> np.ndarray:
    idx = min(idx, X.shape[2] - 1)
    return np.clip(X[:, -1, idx], 1e-6, 1.0 - 1e-6)


def _norm_to_ppm(y_norm: np.ndarray, cfg: SFPIConfig) -> np.ndarray:
    return cfg.min_ppd_ppm + np.asarray(y_norm, dtype=np.float64) * (cfg.max_ppd_ppm - cfg.min_ppd_ppm)


# -------------------------
# Industrial STS-style implementation
# -------------------------
class STSStrongInterval:
    def __init__(self, data_x: np.ndarray, tag_names: Sequence[str], cfg: SFPIConfig):
        self.data_x = np.asarray(data_x, dtype=np.float64)
        self.tag_names = list(tag_names)
        self.cfg = cfg

        self.location = np.array([1, 15, 21, 23, 25, 29, 35, 40, 51], dtype=np.float64) / 51.0
        self.xw = 200.0 * 1e-6

    def main(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._extract_values()
        center = self._compute_bottom_ppd_ppm(self.cfg.center_total_stages, bound_flag=False)
        up = self._compute_bottom_ppd_ppm(self.cfg.upper_total_stages, bound_flag=True)
        low = self._compute_bottom_ppd_ppm(self.cfg.lower_total_stages, bound_flag=True)
        return center, up, low

    def _idx(self, tag: str) -> int:
        return self.tag_names.index(tag)

    def _extract_values(self) -> None:
        self.F = np.mean(self.data_x[:, :, self._idx("FI35141.PV")], axis=1) / 108.143
        self.RL = np.mean(self.data_x[:, :, self._idx("FI35144.PV")], axis=1) / 108.143 * 1040.0
        self.D = np.mean(self.data_x[:, :, self._idx("FI35146.PV")], axis=1) / 108.143
        self.RP = np.mean(self.data_x[:, :, self._idx("PI35144.PV")], axis=1)
        bp_raw = np.mean(self.data_x[:, :, self._idx("PI35141a9.PV")], axis=1)
        self.BP = self.bottom_pressure(bp_raw)

        self.T = np.zeros((len(self.F), 9), dtype=np.float64)
        for i in range(9):
            self.T[:, i] = np.mean(self.data_x[:, :, self._idx(f"TI35141a{i + 1}.PV")], axis=1)

        self.TopMPD = np.mean(self.data_x[:, :, self._idx("RAMAN_T35141_MPD.PV")], axis=1) * 1e-6
        self.TopOPD = np.mean(self.data_x[:, :, self._idx("RAMAN_T35141_OPD.PV")], axis=1) * 1e-2
        self.TopPPD = np.mean(self.data_x[:, :, self._idx("RAMAN_T35141_PPD.PV")], axis=1) * 1e-2

    @staticmethod
    def middle_feed_stage(total_stages: int) -> int:
        # 0-based switching index used in the stage loop; keeps feed approximately in the middle.
        return total_stages // 2 + 1

    def _compute_bottom_ppd_ppm(self, total_stages: int, bound_flag: bool) -> np.ndarray:
        out = np.zeros(len(self.F), dtype=np.float64)
        feed_stage = self.middle_feed_stage(total_stages)
        sim_location = (np.arange(total_stages, dtype=np.float64) + 1.0) / float(total_stages)

        for j in range(len(self.F)):
            out[j] = self._calculation_one_sample(j, total_stages, feed_stage, sim_location, bound_flag)

        # Convert mole fraction to ppm.
        return out * 1e6

    def _calculation_one_sample(
        self,
        j: int,
        total_stages: int,
        feed_stage: int,
        sim_location: np.ndarray,
        bound_flag: bool,
    ) -> float:
        t_profile = self._cubic_spline_interpolation(self.location, self.T[j, :], sim_location)
        p_profile = np.linspace(self.RP[j], self.BP[j], total_stages)

        if bound_flag:
            xD = float(self.TopPPD[j])
        else:
            denom = np.sum(self.TopPPD[j] + self.TopOPD[j] + self.TopMPD[j])
            denom = max(float(denom), self.cfg.eps)
            xD = float((self.TopPPD[j] + self.TopOPD[j]) / denom)

        y_vals: List[float] = [xD]
        x_vals: List[float] = []

        reflux_ratio = self.RL[j] / max(self.D[j], self.cfg.eps)
        bottom_flow = self.F[j] - self.D[j]
        stripping_L = self.RL[j] + self.F[j]

        for i in range(total_stages):
            x_liq, _ = self.equilibrium_func(y_vals[-1], t_profile[i], p_profile[i])
            x_vals.append(x_liq)

            if i < total_stages - 1:
                if i < feed_stage:
                    y_next = self.distillate_operation_func(x_liq, reflux_ratio, xD)
                else:
                    y_next = self.stripping_operation_func(x_liq, stripping_L, bottom_flow, self.xw)
                y_vals.append(y_next)

        return float(np.clip(x_vals[-1], self.cfg.eps, 1.0))

    def _cubic_spline_interpolation(self, x: np.ndarray, y: np.ndarray, new_x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        new_x = np.asarray(new_x, dtype=np.float64)
        if CubicSpline is not None:
            cs = CubicSpline(x, y)
            return np.asarray(cs(new_x), dtype=np.float64)
        return np.asarray(np.interp(new_x, x, y), dtype=np.float64)

    @staticmethod
    def kij(T: float, P: float, i: int) -> float:
        C1 = np.array([43.79824472, 75.75524472], dtype=np.float64)
        C2 = np.array([-8946.0, -11236.0], dtype=np.float64)
        C3 = np.zeros(2, dtype=np.float64)
        C4 = np.zeros(2, dtype=np.float64)
        C5 = np.array([-3.6984, -8.0069], dtype=np.float64)
        C6 = np.array([6.49e-07, 1.56e-18], dtype=np.float64)
        C7 = np.array([2.0, 6.0], dtype=np.float64)
        T_k = T + 273.15
        K = np.exp(C1[i] + C2[i] / (C3[i] + T_k) + C4[i] * T_k + C5[i] * np.log(T_k) + C6[i] * T_k ** C7[i])
        return float(K / max(P, 1e-12))

    def equilibrium_func(self, y: float, t: float, p: float) -> Tuple[float, float]:
        k1 = self.kij(t, p, 0)
        k2 = self.kij(t, p, 1)
        alpha = k2 / max(k1, 1e-12)
        x = y / (alpha - (alpha - 1.0) * y)
        return float(np.clip(x, self.cfg.eps, 1.0)), float(alpha)

    @staticmethod
    def distillate_operation_func(x: float, R: float, xD: float) -> float:
        return float(R / (R + 1.0) * x + xD / (R + 1.0))

    @staticmethod
    def stripping_operation_func(x: float, L: float, W: float, xW: float) -> float:
        denom = L - W
        if abs(denom) < 1e-12:
            denom = np.sign(denom) * 1e-12 if denom != 0 else 1e-12
        return float(L / denom * x + xW / denom)

    @staticmethod
    def bottom_pressure(bp: np.ndarray) -> np.ndarray:
        return 0.7633 * np.asarray(bp, dtype=np.float64) + 6.502


# -------------------------
# Fallback surrogate implementation
# -------------------------
def _bottom_from_top_and_stage_norm(top: np.ndarray, stages: float, cfg: SFPIConfig) -> np.ndarray:
    top = np.clip(top, cfg.eps, 1.0 - cfg.eps)
    base = np.power(1.0 - top, cfg.nonlinearity_power)
    bottom = base * np.exp(-cfg.stage_scale * stages) + 0.02
    return np.clip(bottom, cfg.eps, 1.0 - cfg.eps)


def _build_fallback_split(X: np.ndarray, cfg: SFPIConfig) -> np.ndarray:
    top = _last_step_feature(X, cfg.top_idx)
    up_norm = _bottom_from_top_and_stage_norm(top, float(cfg.upper_total_stages), cfg)
    low_norm = _bottom_from_top_and_stage_norm(top, float(cfg.lower_total_stages), cfg)
    up_ppm = _norm_to_ppm(up_norm, cfg)
    low_ppm = _norm_to_ppm(low_norm, cfg)
    return _normalize_interval_ppm(low_ppm, up_ppm, cfg)


# -------------------------
# Public builder
# -------------------------
def _has_required_tags(tag_names: Optional[Sequence[str]]) -> bool:
    if tag_names is None:
        return False
    tag_set = set(tag_names)
    return all(tag in tag_set for tag in REQUIRED_TAGS)


def build_sfpi(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test_like: np.ndarray,
    y_train: Optional[np.ndarray] = None,  # kept for interface compatibility
    y_val: Optional[np.ndarray] = None,    # kept for interface compatibility
    config: Optional[SFPIConfig] = None,
) -> Dict[str, Any]:
    del y_train, y_val

    if config is None:
        config = SFPIConfig()

    X_train = _to_numpy(X_train).astype(np.float32)
    X_val = _to_numpy(X_val).astype(np.float32)
    X_test_like = _to_numpy(X_test_like).astype(np.float32)

    if _has_required_tags(config.tag_names):
        sts_train = STSStrongInterval(X_train, config.tag_names, config)
        train_center_ppm, train_up_ppm, train_low_ppm = sts_train.main()

        sts_val = STSStrongInterval(X_val, config.tag_names, config)
        val_center_ppm, val_up_ppm, val_low_ppm = sts_val.main()

        sts_test = STSStrongInterval(X_test_like, config.tag_names, config)
        test_center_ppm, test_up_ppm, test_low_ppm = sts_test.main()

        return {
            "train": _normalize_interval_ppm(train_low_ppm, train_up_ppm, config),
            "val": _normalize_interval_ppm(val_low_ppm, val_up_ppm, config),
            "test": _normalize_interval_ppm(test_low_ppm, test_up_ppm, config),
            "metadata": {
                "mode": "industrial_stage_by_stage",
                "center_total_stages": config.center_total_stages,
                "upper_total_stages": config.upper_total_stages,
                "lower_total_stages": config.lower_total_stages,
                "feed_stage_rule": "fixed middle feed stage",
                "ppm_min": config.min_ppd_ppm,
                "ppm_max": config.max_ppd_ppm,
                "normalization": "(ppm - 5) / (200 - 5)",
                "train_center_ppm": train_center_ppm.astype(np.float32),
                "val_center_ppm": val_center_ppm.astype(np.float32),
                "test_center_ppm": test_center_ppm.astype(np.float32),
            },
        }

    return {
        "train": _build_fallback_split(X_train, config),
        "val": _build_fallback_split(X_val, config),
        "test": _build_fallback_split(X_test_like, config),
        "metadata": {
            "mode": "fallback_surrogate",
            "center_total_stages": config.center_total_stages,
            "upper_total_stages": config.upper_total_stages,
            "lower_total_stages": config.lower_total_stages,
            "feed_stage_rule": "fixed middle feed stage",
            "ppm_min": config.min_ppd_ppm,
            "ppm_max": config.max_ppd_ppm,
            "normalization": "(ppm - 5) / (200 - 5)",
        },
    }


__all__ = ["SFPIConfig", "build_sfpi", "STSStrongInterval"]