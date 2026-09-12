"""退火曲线分析：内外温差、穿越退火区间速率、有效保温时长与违规检测。

违规类型：
- thermal_shock        热冲击：|表面-中心| 超过工件允许温差
- fast_cooling         过快降温：中心穿越退火区间的速率超过该工件允许值
- center_not_equalized 中心未均温：保温段结束时中心与（局部）炉温差距过大
- program_jump         程序跳变：相邻段设定温度不连续（仅名义程序），附实际受影响工件
- kiln_max_temp        超过窑炉最高温（理想设定与各区域预测炉温均检查）

三种炉温来源（互斥）：
- program    名义程序：理想设定即炉温
- measured   实测复核：缺测区间保持未知，首个缺测点之后截断评估
- prediction 窑炉热响应档案：各工件按其探头区域取局部预测炉温，
             结果同时给出理想设定、局部预测、跟随误差、超调与最不利工件。
             档案不适用（装载质量/温区超出校准范围）时由上层直接判 rejected，
             不进入本模块的安全判定。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

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


def _follow_metrics(setpoint: np.ndarray, predicted: np.ndarray,
                    base_rate: np.ndarray, dt_min: float,
                    overshoot_mask: Optional[np.ndarray] = None) -> dict:
    """理想设定 vs 局部预测的跟随误差与超调。

    - 跟随误差：升温段炉温落后设定（setpoint−predicted）的最大值；
      降温段炉温落后降温（predicted−setpoint）的最大值。
    - 超调：仅在 overshoot_mask（升温段及紧随其后的保温段）统计
      predicted−setpoint 的正值；降温后保温初期的余热属于滞后。
    """
    heat = base_rate > HOLD_RATE_EPS
    cool = base_rate < -HOLD_RATE_EPS
    lag = setpoint - predicted
    out = {
        "max_heating_lag_c": round(float(np.max(lag[heat])) if heat.any() else 0.0, 3),
        "max_cooling_lag_c": round(float(np.max(-lag[cool])) if cool.any() else 0.0, 3),
        "max_abs_follow_error_c": 0.0,
        "max_overshoot_c": 0.0,
        "overshoot_time_min": None,
    }
    ramping = heat | cool
    out["max_abs_follow_error_c"] = round(
        float(np.max(np.abs(lag[ramping]))) if ramping.any() else 0.0, 3
    )
    mask = overshoot_mask if overshoot_mask is not None else ~cool
    over = np.where(mask, predicted - setpoint, -np.inf)
    i = int(np.argmax(over))
    if over[i] > 0:
        out["max_overshoot_c"] = round(float(over[i]), 3)
        out["overshoot_time_min"] = round(float(i * dt_min), 3)
    return out


def analyze(
    pieces: Sequence[PieceSpec],
    program: Optional[KilnProgram] = None,
    measured: Optional[dict] = None,
    prediction: Optional[dict] = None,
    constraints: Optional[Constraints] = None,
    options: Optional[AnalysisOptions] = None,
) -> dict:
    """对一组工件执行热分析。program / measured / prediction 三选一。"""
    options = options or AnalysisOptions()
    constraints = constraints or Constraints()
    dt_s = options.dt_s
    dt_min = dt_s / 60.0

    spans = None
    jumps: List[dict] = []
    unknown_intervals: List[dict] = []
    is_prediction = prediction is not None

    if prediction is not None:
        t_min = np.asarray(prediction["t_min"], dtype=float)
        setpoint = np.asarray(prediction["setpoint_c"], dtype=float)
        kiln = setpoint.copy()  # 兜底；实际逐工件取区域预测
        seg_indices = np.asarray(prediction["seg_index"], dtype=int)
        jumps = list(prediction["jumps"])

        def seg_index_at(i: int) -> Optional[int]:
            idx = int(seg_indices[i])
            return None if idx < 0 else idx
    elif measured is not None:
        t_min, kiln, unknown_intervals = _measured_curve(
            measured["time_min"], measured["temp_c"], measured["max_gap_min"], dt_s
        )
        setpoint = kiln

        def seg_index_at(i: int) -> Optional[int]:
            return None
    else:
        asm = assemble(program, dt_s, options.jump_tol_c)
        t_min, kiln, spans, jumps = asm.t_min, asm.temp_c, asm.spans, asm.jumps
        setpoint = kiln

        def seg_index_at(i: int) -> Optional[int]:
            for sp in spans:
                if sp.i0 <= i <= sp.i1:
                    return sp.index
            return None

    # 首个缺测点之后，玻璃内部状态无法继续诚实推演，截断评估范围
    valid = np.isfinite(kiln)
    n_valid = kiln.size if bool(valid.all()) else int(np.argmin(valid))
    t_v = t_min[:n_valid]
    ideal_v = setpoint[:n_valid]

    violations: List[dict] = []
    # 程序跳变：设定阶跃几乎瞬间全部落在工件表面与中心之间
    for j in jumps:
        mag = abs(j["to_c"] - j["from_c"])
        affected = sorted(
            (p for p in pieces if mag > p.allowed_delta_t_c),
            key=lambda p: p.allowed_delta_t_c,
        )
        names = [p.name for p in affected]
        msg = (
            f"段 {j['segment_index']} 起点设定从 {j['from_c']:.1f}°C "
            f"跳变到 {j['to_c']:.1f}°C"
        )
        if names:
            msg += f"，受影响工件: {', '.join(names)}"
        violations.append({
            "type": "program_jump",
            "piece": names[0] if names else None,
            "pieces": names,
            "time_min": round(j["time_min"], 3),
            "segment_index": j["segment_index"],
            "value": round(j["to_c"] - j["from_c"], 3),
            "limit": options.jump_tol_c,
            "message": msg,
        })

    # 共享参考速率：名义/实测用炉温，预测模式用理想设定（保温段由程序定义）
    base_rate = np.gradient(ideal_v, dt_min) if n_valid >= 2 else np.zeros(n_valid)

    if n_valid >= 2:
        over = np.flatnonzero(ideal_v > constraints.kiln_max_temp_c)
        if over.size:
            i = int(over[np.argmax(ideal_v[over])])
            violations.append({
                "type": "kiln_max_temp",
                "piece": None,
                "zone": None,
                "time_min": round(float(t_v[i]), 3),
                "segment_index": seg_index_at(i),
                "value": round(float(ideal_v[i]), 3),
                "limit": constraints.kiln_max_temp_c,
                "message": (
                    f"{t_v[i]:.1f} min 设定温度 {ideal_v[i]:.1f}°C "
                    f"超过窑炉上限 {constraints.kiln_max_temp_c:.1f}°C"
                ),
            })
        # 预测模式：各区域局部预测炉温本身也可能超窑炉上限（超调/偏置）
        if is_prediction:
            checked_zones = set()
            for zone, zdata in prediction["zones"].items():
                if zone in checked_zones:
                    continue
                checked_zones.add(zone)
                zc = np.asarray(zdata["predicted_c"])[:n_valid]
                overz = np.flatnonzero(zc > constraints.kiln_max_temp_c)
                if overz.size:
                    i = int(overz[np.argmax(zc[overz])])
                    violations.append({
                        "type": "kiln_max_temp",
                        "piece": None,
                        "zone": zone,
                        "time_min": round(float(t_v[i]), 3),
                        "segment_index": seg_index_at(i),
                        "value": round(float(zc[i]), 3),
                        "limit": constraints.kiln_max_temp_c,
                        "message": (
                            f"{t_v[i]:.1f} min 区域 {zone} 预测炉温 {zc[i]:.1f}°C "
                            f"超过窑炉上限 {constraints.kiln_max_temp_c:.1f}°C"
                        ),
                    })
        hold_mask = np.abs(base_rate) < HOLD_RATE_EPS
        hold_intervals = [
            (s, e) for s, e in zip(*_runs(hold_mask))
            if (e - s) * dt_min >= MIN_HOLD_MIN
        ]
    else:
        hold_intervals = []

    # 预测模式下各区域的跟随指标（按区域预算，工件引用其区域结果）
    zone_metrics: Dict[str, dict] = {}
    if is_prediction:
        over_mask = np.asarray(prediction.get("overshoot_mask", np.ones_like(ideal_v, dtype=bool)))[:n_valid]
        for zone, zdata in prediction["zones"].items():
            pred_v = np.asarray(zdata["predicted_c"])[:n_valid]
            zone_metrics[zone] = {
                "channel_id": zdata["channel_id"],
                "position": zdata.get("position", ""),
                **_follow_metrics(ideal_v, pred_v, base_rate, dt_min, over_mask),
            }

    timeseries = {
        "t_min": np.round(t_v, 3).tolist(),
        "kiln_c": np.round(kiln[:n_valid], 3).tolist(),
        "pieces": {},
    }
    if is_prediction:
        timeseries["setpoint_c"] = np.round(ideal_v, 3).tolist()
        timeseries["zones"] = {
            zone: {
                "channel_id": zdata["channel_id"],
                "predicted_c": np.round(
                    np.asarray(zdata["predicted_c"])[:n_valid], 3
                ).tolist(),
            }
            for zone, zdata in prediction["zones"].items()
        }

    summaries: List[dict] = []
    piece_viol_counts: Dict[str, int] = {}

    for p in pieces:
        L, r_max = piece_limits(p)
        eq_tol = options.equalization_tol_c or p.allowed_delta_t_c / 2.0

        if is_prediction:
            zone = prediction["piece_zones"][p.name]
            piece_kiln = np.asarray(prediction["zones"][zone]["predicted_c"])[:n_valid]
        else:
            zone = None
            piece_kiln = kiln[:n_valid]

        center, surface = simulate_piece(
            p.thermal_diffusivity_m2s, p.max_thickness_mm, dt_s, piece_kiln,
            n_nodes=options.nodes, h_w_m2k=options.surface_h_w_m2k,
        )
        if center.size < 2:
            continue
        delta = surface - center
        piece_rate = np.gradient(piece_kiln, dt_min)
        center_rate = -np.gradient(center, dt_min)  # 降温速率取正

        before = len(violations)

        # 热冲击：内外温差超限
        shock = np.abs(delta) > p.allowed_delta_t_c
        for s, e in zip(*_runs(shock)):
            i = int(s + np.argmax(np.abs(delta[s:e])))
            phase = "升温" if base_rate[i] > HOLD_RATE_EPS else (
                "降温" if base_rate[i] < -HOLD_RATE_EPS else "保温")
            loc = f"（区域 {zone}）" if zone else ""
            violations.append({
                "type": "thermal_shock",
                "piece": p.name,
                "zone": zone,
                "time_min": round(float(t_v[i]), 3),
                "segment_index": seg_index_at(i),
                "value": round(float(abs(delta[i])), 3),
                "limit": p.allowed_delta_t_c,
                "message": (
                    f"{p.name}{loc} 在 {t_v[i]:.1f} min（{phase}）内外温差 "
                    f"{abs(delta[i]):.1f}°C 超过允许 {p.allowed_delta_t_c:.1f}°C"
                ),
            })

        # 过快降温：中心位于退火区间 [应变点, 退火点] 且局部炉温在降
        in_window = (center <= p.anneal_point_c) & (center >= p.strain_point_c)
        cooling = piece_rate < -HOLD_RATE_EPS
        fast = in_window & cooling & (center_rate > r_max)
        for s, e in zip(*_runs(fast)):
            i = int(s + np.argmax(center_rate[s:e]))
            loc = f"（区域 {zone}）" if zone else ""
            violations.append({
                "type": "fast_cooling",
                "piece": p.name,
                "zone": zone,
                "time_min": round(float(t_v[i]), 3),
                "segment_index": seg_index_at(i),
                "value": round(float(center_rate[i]), 3),
                "limit": round(r_max, 3),
                "message": (
                    f"{p.name}{loc} 在 {t_v[i]:.1f} min 以 {center_rate[i]:.1f}°C/min "
                    f"穿越退火区间，超过允许 {r_max:.1f}°C/min"
                ),
            })

        # 中心未均温：保温段（温度不低于应变点）结束时中心仍滞后于局部炉温
        for s, e in hold_intervals:
            i = e - 1
            hold_temp = float(np.mean(ideal_v[s:e]))
            if hold_temp < p.strain_point_c:
                continue
            gap = abs(float(center[i]) - float(piece_kiln[i]))
            if gap > eq_tol:
                loc = f"（区域 {zone}）" if zone else ""
                violations.append({
                    "type": "center_not_equalized",
                    "piece": p.name,
                    "zone": zone,
                    "time_min": round(float(t_v[i]), 3),
                    "segment_index": seg_index_at(i),
                    "value": round(gap, 3),
                    "limit": eq_tol,
                    "message": (
                        f"{p.name}{loc} 在 {t_v[i]:.1f} min 保温结束时中心与炉温相差 "
                        f"{gap:.1f}°C，超过 {eq_tol:.1f}°C"
                    ),
                })

        piece_viol_counts[p.name] = len(violations) - before

        eff_hold = float(np.sum(np.abs(center - p.anneal_point_c) <= options.soak_band_c) * dt_min)
        cross = center_rate[in_window & cooling]
        heat_mask = piece_rate > HOLD_RATE_EPS
        cool_mask = piece_rate < -HOLD_RATE_EPS
        max_abs_delta = float(np.max(np.abs(delta)))
        summary = {
            "name": p.name,
            "zone": zone,
            "layers": p.layers,
            "max_thickness_mm": p.max_thickness_mm,
            "thermal_time_constant_min": round(L**2 / p.thermal_diffusivity_m2s / 60.0, 3),
            "max_allowed_rate_c_per_min": round(r_max, 3),
            "max_abs_delta_t_c": round(max_abs_delta, 3),
            "max_delta_heating_c": round(float(np.max(delta[heat_mask])) if heat_mask.any() else 0.0, 3),
            "max_delta_cooling_c": round(float(np.max(-delta[cool_mask])) if cool_mask.any() else 0.0, 3),
            "effective_hold_min": round(eff_hold, 3),
            "anneal_range_crossing_rate_c_per_min": round(float(cross.max()) if cross.size else 0.0, 3),
        }
        if is_prediction:
            summary["following"] = zone_metrics[zone]
        summaries.append(summary)
        timeseries["pieces"][p.name] = {
            "center_c": np.round(center, 3).tolist(),
            "surface_c": np.round(surface, 3).tolist(),
            "delta_c": np.round(delta, 3).tolist(),
            "kiln_c": np.round(piece_kiln, 3).tolist(),
        }

    result_extra: dict = {}
    if is_prediction:
        # 最不利工件：违规数优先，其次温差利用率，再看穿越速率裕度
        by_name = {s["name"]: s for s in summaries}
        def _score(s: dict) -> Tuple[float, ...]:
            vcnt = piece_viol_counts.get(s["name"], 0)
            delta_use = s["max_abs_delta_t_c"] / max(
                next(p.allowed_delta_t_c for p in pieces if p.name == s["name"]), 1e-9
            )
            rate_ratio = (
                s["anneal_range_crossing_rate_c_per_min"] / s["max_allowed_rate_c_per_min"]
                if s["max_allowed_rate_c_per_min"] > 0 else 0.0
            )
            return (vcnt, delta_use, rate_ratio)
        worst = max(summaries, key=_score) if summaries else None
        result_extra["prediction"] = {
            "profile_id": prediction["profile_id"],
            "profile_version": prediction["profile_version"],
            "profile_name": prediction.get("profile_name"),
            "zones": zone_metrics,
            "worst_piece": (
                {
                    "name": worst["name"],
                    "zone": worst["zone"],
                    "violation_count": piece_viol_counts.get(worst["name"], 0),
                    "max_abs_delta_t_c": worst["max_abs_delta_t_c"],
                    "max_overshoot_c": worst["following"]["max_overshoot_c"],
                    "max_abs_follow_error_c": worst["following"]["max_abs_follow_error_c"],
                }
                if worst else None
            ),
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
        **result_extra,
    }
