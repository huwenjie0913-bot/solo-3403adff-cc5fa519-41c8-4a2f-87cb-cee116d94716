"""排程：在锁定段、窑炉上限、最小控温步长与总时长约束下，
迭代调整斜率与保温时长，直到所有工件合规；无法满足时列出冲突。

调整策略（只向更温和方向走，保证迭代收敛）：
- fast_cooling / thermal_shock → 所有速率超过 min(材料允许)×安全系数
  的 ramp/cool 段统一降速（对最慢工件本就不安全）；
- center_not_equalized        → 延长该保温段，并记录其均温下限，
  后续压缩总时长时不得再低于该下限（避免来回振荡）；
- kiln_max_temp               → 目标温度压到窑炉上限；
- program_jump                → 去掉 hold 段的显式 target_c，使其随前段温度。

无违规但超出总时长上限时，不直接判冲突，先在安全范围内压缩：
未锁定 ramp/cool 提速至允许上限，再缩短未锁定保温段
（不低于均温下限与最小控温步长）；压缩后仍超限才报告
duration_limit 冲突。锁定段无法调整时记录 locked_segment
冲突并列出受限材料。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from .analysis import analyze, piece_limits
from .models import AnalysisOptions, Constraints, KilnProgram, PieceSpec
from .program import quantize_program


def _dedup(conflicts: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for c in conflicts:
        key = (c["type"], c.get("segment_index"), tuple(c.get("pieces", [])))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _compress(
    prog: KilnProgram,
    excess_min: float,
    rate_cap: float,
    hold_floors: Dict[int, float],
    constraints: Constraints,
) -> bool:
    """在安全范围内压缩总时长，返回是否有改动。

    1) 未锁定 ramp/cool 提速至 rate_cap（按量化后的最短时长换算速率）；
    2) 未锁定 hold 缩短，但不低于 hold_floors（均温下限）与最小控温步长。

    所有改动保持最小控温步长的整数倍，与 quantize_program 幂等兼容，
    因此不会在"压缩—量化"之间来回振荡。
    """
    step = constraints.min_step_min
    remaining = excess_min
    changed = False

    # 1) 提速：量化后的最短时长 min_dur 保证换算速率不超过 rate_cap
    cur = prog.start_c
    for seg in prog.segments:
        if seg.kind in ("ramp", "cool") and not seg.locked and seg.rate_c_per_min:
            delta = abs(seg.target_c - cur)
            if delta > 0 and seg.rate_c_per_min < rate_cap - 1e-9:
                min_dur = max(step, math.ceil(delta / rate_cap / step - 1e-9) * step)
                cur_dur = delta / seg.rate_c_per_min
                if cur_dur - min_dur > 1e-6:
                    seg.rate_c_per_min = delta / min_dur
                    remaining -= cur_dur - min_dur
                    changed = True
        cur = seg.target_c if seg.target_c is not None else cur

    if remaining <= 1e-9:
        return changed

    # 2) 缩短未锁定保温段（按步长整数倍，向下不低于均温下限）
    for idx, seg in enumerate(prog.segments):
        if remaining <= 1e-9:
            break
        if seg.kind != "hold" or seg.locked:
            continue
        floor = max(step, math.ceil(hold_floors.get(idx, step) / step - 1e-9) * step)
        avail = seg.duration_min - floor
        if avail <= 1e-9:
            continue
        cut = min(avail, math.ceil(remaining / step - 1e-9) * step)
        if cut <= 1e-9:
            continue
        seg.duration_min -= cut
        remaining -= cut
        changed = True
    return changed


def optimize(job: dict, safety_factor: float = 0.9, max_iterations: int = 40) -> dict:
    pieces = [PieceSpec(**p) for p in job["pieces"]]
    constraints = Constraints(**job["constraints"])
    options = AnalysisOptions(**job["options"])
    prog = KilnProgram(**job["program"])

    # 各工件允许的升降温速率上限（°C/min）
    reqs = {p.name: piece_limits(p)[1] for p in pieces}
    slowest = min(reqs, key=reqs.get)
    need = round(min(reqs.values()) * safety_factor, 4)

    conflicts: List[dict] = []

    # 退火点高于窑炉上限：结构性冲突，直接判不可行
    over = [p.name for p in pieces if p.anneal_point_c > constraints.kiln_max_temp_c]
    if over:
        conflicts.append({
            "type": "kiln_max_temp",
            "pieces": over,
            "message": f"退火点高于窑炉上限 {constraints.kiln_max_temp_c}°C: {', '.join(over)}",
        })
        return {"status": "infeasible", "conflicts": conflicts}

    hold_floors: Dict[int, float] = {}  # 段号 -> 均温所需的最短保温时长
    last: Optional[dict] = None
    for _ in range(max_iterations):
        prog = quantize_program(prog, constraints.min_step_min)
        res = analyze(pieces, program=prog, constraints=constraints, options=options)
        last = res
        viols = res["violations"]

        if not viols:
            if (
                constraints.max_total_duration_min
                and res["total_duration_min"] > constraints.max_total_duration_min + 1e-6
            ):
                # 无违规但超时长：先尝试压缩，压不动才报告冲突
                if _compress(
                    prog,
                    res["total_duration_min"] - constraints.max_total_duration_min,
                    need,
                    hold_floors,
                    constraints,
                ):
                    continue
                conflicts.append({
                    "type": "duration_limit",
                    "pieces": [slowest],
                    "required_min": res["total_duration_min"],
                    "limit_min": constraints.max_total_duration_min,
                    "message": (
                        f"提速至安全上限并缩短未锁定保温后仍需 "
                        f"{res['total_duration_min']:.0f} min，"
                        f"超过总时长上限 {constraints.max_total_duration_min:.0f} min；"
                        f"瓶颈工件: {slowest}（允许速率 {reqs[slowest]:.1f}°C/min）"
                    ),
                })
                return {"status": "infeasible", "conflicts": _dedup(conflicts), "analysis": last}
            return {"status": "ok", "program": prog.model_dump(), "analysis": res}

        changed = False
        viol_types = {v["type"] for v in viols}

        # 过快降温 / 热冲击：任何速率超过 min(材料允许)×安全系数 的
        # ramp/cool 段对最慢工件都不安全，统一降速；锁定段记为冲突。
        if viol_types & {"fast_cooling", "thermal_shock"}:
            for idx, seg in enumerate(prog.segments):
                if seg.kind not in ("ramp", "cool"):
                    continue
                if seg.rate_c_per_min is None or seg.rate_c_per_min <= need:
                    continue
                if seg.locked:
                    bad = [n for n, r in reqs.items() if r < seg.rate_c_per_min]
                    conflicts.append({
                        "type": "locked_segment",
                        "segment_index": idx,
                        "pieces": bad,
                        "message": (
                            f"段 {idx} 已锁定，速率 {seg.rate_c_per_min}°C/min "
                            f"超过材料允许值: {', '.join(bad)}"
                        ),
                    })
                else:
                    seg.rate_c_per_min = need
                    changed = True

        # 超过窑炉上限：所有目标温度超限的段统一压到上限
        if "kiln_max_temp" in viol_types:
            for idx, seg in enumerate(prog.segments):
                if seg.target_c is not None and seg.target_c > constraints.kiln_max_temp_c:
                    if seg.locked:
                        conflicts.append({
                            "type": "locked_segment",
                            "segment_index": idx,
                            "pieces": [],
                            "message": f"段 {idx} 已锁定且目标温度超过窑炉上限",
                        })
                    else:
                        seg.target_c = constraints.kiln_max_temp_c
                        changed = True

        for v in viols:
            si = v.get("segment_index")
            if si is None:
                continue
            seg = prog.segments[si]

            if v["type"] == "center_not_equalized" and seg.kind == "hold":
                if seg.locked:
                    conflicts.append({
                        "type": "locked_segment",
                        "segment_index": si,
                        "pieces": [v["piece"]],
                        "message": f"段 {si} 已锁定，{v['piece']} 保温结束时中心未均温",
                    })
                else:
                    extra = max(
                        piece_limits(p)[0] ** 2 / p.thermal_diffusivity_m2s for p in pieces
                    ) / 60.0
                    seg.duration_min += extra
                    # 记录均温下限：压缩总时长时不得再低于该值
                    hold_floors[si] = max(hold_floors.get(si, 0.0), seg.duration_min)
                    changed = True

            elif v["type"] == "program_jump" and seg.kind == "hold" and seg.target_c is not None:
                if seg.locked:
                    conflicts.append({
                        "type": "locked_segment",
                        "segment_index": si,
                        "pieces": [],
                        "message": f"段 {si} 已锁定且存在程序跳变",
                    })
                else:
                    seg.target_c = None  # 随前段温度，消除跳变
                    changed = True

        if not changed:
            break

    if not conflicts:
        bad_pieces = sorted({v["piece"] for v in (last["violations"] if last else []) if v.get("piece")})
        conflicts.append({
            "type": "unresolved",
            "pieces": bad_pieces,
            "message": "在迭代上限内无法消除所有违规",
            "violations": (last["violations"] if last else [])[:20],
        })
    return {"status": "infeasible", "conflicts": _dedup(conflicts), "analysis": last}
