"""窑炉程序装配与量化。

assemble: 把段序列展开为等间距炉温曲线，并检测程序跳变；
quantize_program: 把每段持续时间向上取整到最小控温步长的整数倍
（ramp/cool 通过降低速率实现，不会因此产生更快的升降温）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .models import KilnProgram


@dataclass
class SegmentSpan:
    index: int
    kind: str
    i0: int  # 在时间网格上的起止下标（含）
    i1: int
    start_c: float
    end_c: float
    rate_c_per_min: float


@dataclass
class Assembled:
    t_min: np.ndarray
    temp_c: np.ndarray
    spans: List[SegmentSpan]
    jumps: List[dict]
    total_min: float


def assemble(program: KilnProgram, dt_s: float, jump_tol_c: float = 1.0) -> Assembled:
    dt_min = dt_s / 60.0
    t: List[float] = [0.0]
    temp: List[float] = [program.start_c]
    spans: List[SegmentSpan] = []
    jumps: List[dict] = []
    cur = program.start_c

    for idx, seg in enumerate(program.segments):
        i0 = len(t) - 1
        start = cur
        rate = 0.0
        if seg.kind == "hold":
            target = seg.target_c if seg.target_c is not None else cur
            if abs(target - cur) > jump_tol_c:
                jumps.append({
                    "segment_index": idx,
                    "time_min": t[-1],
                    "from_c": cur,
                    "to_c": target,
                })
            cur = target
            n = max(1, int(round(seg.duration_min / dt_min)))
            base_t = t[-1]
            t.extend(base_t + dt_min * k for k in range(1, n + 1))
            temp.extend([cur] * n)
        else:
            target = float(seg.target_c)
            rate = float(seg.rate_c_per_min)
            dur = abs(target - cur) / rate if rate > 0 else 0.0
            n = max(1, int(math.ceil(dur / dt_min - 1e-9)))
            base_t = t[-1]
            for k in range(1, n + 1):
                t.append(base_t + dt_min * k)
                temp.append(cur + (target - cur) * k / n)
            cur = target
        spans.append(SegmentSpan(idx, seg.kind, i0, len(t) - 1, start, cur, rate))

    return Assembled(np.asarray(t), np.asarray(temp), spans, jumps, t[-1])


def quantize_program(program: KilnProgram, min_step_min: float) -> KilnProgram:
    """段持续时间向上取整到 min_step_min 的整数倍。

    ramp/cool 段通过降低速率来匹配取整后的时长，因此量化只会
    让曲线更温和，不会引入更快的升降温。
    locked=True 的段不可修改，原样保留。
    """
    segs = []
    cur = program.start_c
    for seg in program.segments:
        s = seg.model_copy()
        if s.locked:
            # 锁定段不参与量化，仅推进链式温度
            cur = s.target_c if s.target_c is not None else cur
            segs.append(s)
            continue
        if s.kind == "hold":
            s.duration_min = max(
                min_step_min,
                math.ceil(s.duration_min / min_step_min - 1e-9) * min_step_min,
            )
            cur = s.target_c if s.target_c is not None else cur
        else:
            delta = abs(s.target_c - cur)
            raw = delta / s.rate_c_per_min if s.rate_c_per_min else 0.0
            dur = max(min_step_min, math.ceil(raw / min_step_min - 1e-9) * min_step_min)
            s.rate_c_per_min = delta / dur if dur > 0 else s.rate_c_per_min
            cur = s.target_c
        segs.append(s)
    return KilnProgram(start_c=program.start_c, segments=segs)
