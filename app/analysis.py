"""退火曲线分析：内外温差、穿越退火区间速率、有效保温时长与违规检测。

违规类型：
- thermal_shock        热冲击：|表面-中心| 超过工件允许温差
- fast_cooling         过快降温：中心穿越退火区间的速率超过该工件允许值
- center_not_equalized 中心未均温：保温段结束时中心与炉温差距过大
- program_jump         程序跳变：相邻段设定温度不连续（仅名义程序）
- kiln_max_temp        超过窑炉最高温

实测复核：缺测区间（采样间隔超过 max_gap_min）保持未知——
首个缺测点之后的曲线无法诚实评估，不插值、不判定安全。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .models import AnalysisOptions, Constraints, KilnProgram, PieceSpec
from .program import assemble
from .thermal import simulate_piece

HOLD_RATE_EPS = 0.05  # °C/min，炉温变化低于该速率视为保温
MIN_HOLD_MIN = 5.0    # 参与均温判定的最短保温时长


def piece_limits(p: PieceSpec) -> Tuple[float, float]:
    """返回 (半厚度 m, 允许升降温速率 °C/min)。

    平板双面换热准稳态下 ΔT = r·L²/(2α)，故 r_max = 2α·ΔT_allow/L²。
    """
    L = p.max_thickness_mm / 1000.0 / 2.0
    r_max = 2.0 * p.thermal_diffusivity_m2s * p.allowed_delta_t_c / L**2 * 60.0
    return L, r_max


def _runs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """布尔序列中连续 True 区间的 (起点, 终点[开]) 列表。"""
    m = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    d = np.flatnonzero(np.diff(m.astype(np.int8)))
    return d[0::2], d[1::2]


def _measured_curve(
    time_min: Sequence[float],
    temp_c: Sequence[float],
    max_gap_min: float,
    dt_s: float,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """实测炉温重采样到等间距网格；超过 max_gap_min 的缺测区间置 NaN。"""
    order = np.argsort(time_min)
    t = np.asarray(time_min, dtype=float)[order]
    T = np.asarray(temp_c, dtype=float)[order]
    keep = np.concatenate(([True], np.diff(t) > 1e-9))
    t, T = t[keep], T[keep]

    dt_min = dt_s / 60.0
    grid = np.arange(t[0], t[-1] + 1e-9, dt_min)
    kiln = np.interp(grid, t, T)

    unknown: List[dict] = []
    for a, b in zip(t[:-1], t[1:]):
        if b - a > max_gap_min:
            unknown.append({"start_min": float(a), "end_min": float(b)})
            kiln[(grid > a) & (grid < b)] = np.nan
    return grid, kiln, unknown


def analyze(
    pieces: Sequence[PieceSpec],
    program: Optional[KilnProgram] = None,
    measured: Optional[dict] = None,
    constraints: Optional[Constraints] = None,
    options: Optional[AnalysisOptions] = None,
) -> dict:
    """对一组工件执行热分析。program 与 measured 二选一。"""
    options = options or AnalysisOptions()
    constraints = constraints or Constraints()
    dt_s = options.dt_s
    dt_min = dt_s / 60.0

    spans = None
    jumps: List[dict] = []
    unknown_intervals: List[dict] = []
    if measured is not None:
        t_min, kiln, unknown_intervals = _measured_curve(
            measured["time_min"], measured["temp_c"], measured["max_gap_min"], dt_s
        )
    else:
        asm = assemble(program, dt_s, options.jump_tol_c)
        t_min, kiln, spans, jumps = asm.t_min, asm.temp_c, asm.spans, asm.jumps

    # 首个缺测点之后，玻璃内部状态无法继续诚实推演，截断评估范围
    valid = np.isfinite(kiln)
    n_valid = kiln.size if bool(valid.all()) else int(np.argmin(valid))
    t_v = t_min[:n_valid]
    kiln_v = kiln[:n_valid]

    def seg_index_at(i: int) -> Optional[int]:
        if spans is None:
            return None
        for sp in spans:
            if sp.i0 <= i <= sp.i1:
                return sp.index
        return None

    violations: List[dict] = []
    for j in jumps:
        violations.append({
            "type": "program_jump",
            "piece": None,
            "time_min": round(j["time_min"], 3),
            "segment_index": j["segment_index"],
            "value": round(j["to_c"] - j["from_c"], 3),
            "limit": options.jump_tol_c,
            "message": (
                f"段 {j['segment_index']} 起点设定从 {j['from_c']:.1f}°C "
                f"跳变到 {j['to_c']:.1f}°C"
            ),
        })

    if n_valid >= 2:
        kiln_rate = np.gradient(kiln_v, dt_min)
        over = np.flatnonzero(kiln_v > constraints.kiln_max_temp_c)
        if over.size:
            i = int(over[np.argmax(kiln_v[over])])
            violations.append({
                "type": "kiln_max_temp",
                "piece": None,
                "time_min": round(float(t_v[i]), 3),
                "segment_index": seg_index_at(i),
                "value": round(float(kiln_v[i]), 3),
                "limit": constraints.kiln_max_temp_c,
                "message": (
                    f"{t_v[i]:.1f} min 程序温度 {kiln_v[i]:.1f}°C "
                    f"超过窑炉上限 {constraints.kiln_max_temp_c:.1f}°C"
                ),
            })
        hold_mask = np.abs(kiln_rate) < HOLD_RATE_EPS
        hold_intervals = [
            (s, e) for s, e in zip(*_runs(hold_mask))
            if (e - s) * dt_min >= MIN_HOLD_MIN
        ]
    else:
        kiln_rate = np.zeros(n_valid)
        hold_intervals = []

    timeseries = {
        "t_min": np.round(t_v, 3).tolist(),
        "kiln_c": np.round(kiln_v, 3).tolist(),
        "pieces": {},
    }
    summaries: List[dict] = []

    for p in pieces:
        L, r_max = piece_limits(p)
        eq_tol = options.equalization_tol_c or p.allowed_delta_t_c / 2.0
        center, surface = simulate_piece(
            p.thermal_diffusivity_m2s, p.max_thickness_mm, dt_s, kiln_v,
            n_nodes=options.nodes, h_w_m2k=options.surface_h_w_m2k,
        )
        if center.size < 2:
            continue
        delta = surface - center
        center_rate = -np.gradient(center, dt_min)  # 降温速率取正

        # 热冲击：内外温差超限
        shock = np.abs(delta) > p.allowed_delta_t_c
        for s, e in zip(*_runs(shock)):
            i = int(s + np.argmax(np.abs(delta[s:e])))
            phase = "升温" if kiln_rate[i] > HOLD_RATE_EPS else (
                "降温" if kiln_rate[i] < -HOLD_RATE_EPS else "保温")
            violations.append({
                "type": "thermal_shock",
                "piece": p.name,
                "time_min": round(float(t_v[i]), 3),
                "segment_index": seg_index_at(i),
                "value": round(float(abs(delta[i])), 3),
                "limit": p.allowed_delta_t_c,
                "message": (
                    f"{p.name} 在 {t_v[i]:.1f} min（{phase}）内外温差 "
                    f"{abs(delta[i]):.1f}°C 超过允许 {p.allowed_delta_t_c:.1f}°C"
                ),
            })

        # 过快降温：中心位于退火区间 [应变点, 退火点] 且降温速率超限
        in_window = (center <= p.anneal_point_c) & (center >= p.strain_point_c)
        cooling = kiln_rate < -HOLD_RATE_EPS
        fast = in_window & cooling & (center_rate > r_max)
        for s, e in zip(*_runs(fast)):
            i = int(s + np.argmax(center_rate[s:e]))
            violations.append({
                "type": "fast_cooling",
                "piece": p.name,
                "time_min": round(float(t_v[i]), 3),
                "segment_index": seg_index_at(i),
                "value": round(float(center_rate[i]), 3),
                "limit": round(r_max, 3),
                "message": (
                    f"{p.name} 在 {t_v[i]:.1f} min 以 {center_rate[i]:.1f}°C/min "
                    f"穿越退火区间，超过允许 {r_max:.1f}°C/min"
                ),
            })

        # 中心未均温：保温段（温度不低于应变点）结束时中心仍滞后
        for s, e in hold_intervals:
            i = e - 1
            hold_temp = float(np.mean(kiln_v[s:e]))
            if hold_temp < p.strain_point_c:
                continue
            gap = abs(float(center[i]) - float(kiln_v[i]))
            if gap > eq_tol:
                violations.append({
                    "type": "center_not_equalized",
                    "piece": p.name,
                    "time_min": round(float(t_v[i]), 3),
                    "segment_index": seg_index_at(i),
                    "value": round(gap, 3),
                    "limit": eq_tol,
                    "message": (
                        f"{p.name} 在 {t_v[i]:.1f} min 保温结束时中心与炉温相差 "
                        f"{gap:.1f}°C，超过 {eq_tol:.1f}°C"
                    ),
                })

        eff_hold = float(np.sum(np.abs(center - p.anneal_point_c) <= options.soak_band_c) * dt_min)
        cross = center_rate[in_window & cooling]
        heat_mask = kiln_rate > HOLD_RATE_EPS
        cool_mask = kiln_rate < -HOLD_RATE_EPS
        summaries.append({
            "name": p.name,
            "layers": p.layers,
            "max_thickness_mm": p.max_thickness_mm,
            "thermal_time_constant_min": round(L**2 / p.thermal_diffusivity_m2s / 60.0, 3),
            "max_allowed_rate_c_per_min": round(r_max, 3),
            "max_abs_delta_t_c": round(float(np.max(np.abs(delta))), 3),
            "max_delta_heating_c": round(float(np.max(delta[heat_mask])) if heat_mask.any() else 0.0, 3),
            "max_delta_cooling_c": round(float(np.max(-delta[cool_mask])) if cool_mask.any() else 0.0, 3),
            "effective_hold_min": round(eff_hold, 3),
            "anneal_range_crossing_rate_c_per_min": round(float(cross.max()) if cross.size else 0.0, 3),
        })
        timeseries["pieces"][p.name] = {
            "center_c": np.round(center, 3).tolist(),
            "surface_c": np.round(surface, 3).tolist(),
            "delta_c": np.round(delta, 3).tolist(),
        }

    if violations:
        verdict = "fail"
    elif unknown_intervals:
        verdict = "unknown"
    else:
        verdict = "pass"

    return {
        "verdict": verdict,
        "unknown_intervals": unknown_intervals,
        "total_duration_min": round(float(t_min[-1]), 3) if t_min.size else 0.0,
        "assessed_until_min": round(float(t_v[-1]), 3) if t_v.size else 0.0,
        "pieces": summaries,
        "violations": sorted(violations, key=lambda v: v["time_min"]),
        "timeseries": timeseries,
    }
