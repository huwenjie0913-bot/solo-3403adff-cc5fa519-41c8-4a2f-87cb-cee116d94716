"""窑炉热响应辨识与曲线跟随预测。

每个探头（通道）拟合 **纯滞后 + 一阶惯性（FOPDT）** 模型：

    τ · dx/dt = K·(u(t-θ) - b0) - (x - b0)
    y(t)    = x(t)

等价写法：y 对设定 u 的稳态增益为 K、偏置 b 满足 y_ss = K·u + b，即

    v'(t) = (u(t-θ) - v(t)) / τ,   v(t0) = u(t0-θ)
    y(t)  = K·v(t) + b

对固定 (θ, τ)，v 与 (K, b) 线性无关，可用线性最小二乘求 K、b；
θ、τ 采用"网格粗搜 + Nelder-Mead 精修"两级估计。

数据质量（均指出具体通道与时间区间）：
- unsorted_samples          上传采样乱序（记录原始顺序中的下降区间），自动排序后继续
- duplicate_timestamps      同一时刻多个读数，取均值后继续
- out_of_setpoint_range     实测时刻超出设定时间轴，该段不参与拟合
- long_gap                  相邻采样间隔超过阈值（缺省 10×采样间隔），缺口不参与拟合
- insufficient_excitation   有效数据温度变化不足
- unidentifiable            参数撞边界 / 时间常数低于采样分辨率 / 拟合度过低

长缺口把数据切成多段：每段独立做零初始假设（段首 v=u_d），
缺口区间本身不插值、不参与拟合。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

# 参数不可辨识的判定阈值
GAP_FACTOR = 10.0          # 长缺口缺省阈值 = 10 × 标称采样间隔
MIN_POINTS = 5             # 单通道参与拟合的最少网格点数
TAU_RESOLUTION = 1.5       # τ 小于该倍数采样间隔即低于分辨率、不可辨识
MIN_R2 = 0.5               # 拟合优度低于该值视为不可辨识
RANGE_TOL_C = 1.0          # 适用温区/档案范围判定的数值容差 °C

K_BOUNDS = (0.01, 5.0)
BIAS_BOUNDS = (-500.0, 500.0)


# ---------------------------------------------------------------------------
# FOPDT 仿真
# ---------------------------------------------------------------------------

def _first_order_response(u_del: np.ndarray, dt_min: float, tau_min: float) -> np.ndarray:
    """等间距序列上一阶惯性响应 v'=(u_d-v)/τ，隐式/精确离散均用解析系数。

    离散：v[n] = a·v[n-1] + (1-a)·u_d[n]，a = exp(-dt/τ)；
    用 scipy lfilter 加速，初始 v[0] = u_d[0]（段首处于局部稳态）。
    """
    a = math.exp(-dt_min / tau_min)
    # y[n] = (1-a)x[n] + a·y[n-1]  →  b=[1-a], a=[1,-a]；zi=x[0] 使 y[0]=x[0]
    y, _ = lfilter(np.array([1.0 - a]), np.array([1.0, -a]), u_del, zi=[u_del[0]])
    return y


def simulate_fopdt(
    t_min: np.ndarray,
    setpoint_c: np.ndarray,
    delay_min: float,
    tau_min: float,
    gain: float,
    bias_c: float,
) -> np.ndarray:
    """在等间距时间轴上由设定温度预测区域炉温。t<0 的设定取首值。"""
    t = np.asarray(t_min, dtype=float)
    u = np.asarray(setpoint_c, dtype=float)
    dt_min = float(t[1] - t[0]) if t.size >= 2 else 1.0
    u_del = np.interp(t - delay_min, t, u, left=u[0])
    v = _first_order_response(u_del, dt_min, tau_min)
    return gain * v + bias_c


# ---------------------------------------------------------------------------
# 单通道拟合
# ---------------------------------------------------------------------------

def _issue(itype: str, message: str, channel: str, start: float, end: Optional[float] = None) -> dict:
    out = {"type": itype, "channel_id": channel,
           "start_min": round(float(start), 3), "message": message}
    if end is not None:
        out["end_min"] = round(float(end), 3)
    return out


def _resample_segments(
    t_set: np.ndarray, u_set: np.ndarray,
    t_y: np.ndarray, y: np.ndarray,
    interval_min: float, gap_min: float,
    channel: str,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]], List[dict]]:
    """按长缺口把（已裁剪到设定轴内的）实测点切段，重采样到标称采样网格。

    返回 (segments, gap_issues)；每个 segment 为 (t, u, y) 等间距数组，
    缺口区间本身不插值、不参与拟合。
    """
    issues: List[dict] = []
    if t_y.size < 2:
        return [], issues

    # 长缺口：缺口区间不参与拟合（不插值）
    breaks = [0]
    dt = np.diff(t_y)
    for i, gap in enumerate(dt):
        if gap > gap_min + 1e-9:
            issues.append(_issue(
                "long_gap",
                f"通道 {channel} 在 {t_y[i]:.1f}–{t_y[i + 1]:.1f} min 有 {gap:.1f} min "
                f"长缺口（阈值 {gap_min:.1f}），不参与拟合",
                channel, float(t_y[i]), float(t_y[i + 1]),
            ))
            breaks.append(i + 1)
    breaks.append(t_y.size)

    segments: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for s, e in zip(breaks[:-1], breaks[1:]):
        t0 = math.ceil(t_y[s] / interval_min - 1e-9) * interval_min
        t1 = math.floor(t_y[e - 1] / interval_min + 1e-9) * interval_min
        if t1 < t0:
            continue
        grid = np.arange(t0, t1 + 1e-9, interval_min)
        ug = np.interp(grid, t_set, u_set)
        yg = np.interp(grid, t_y, y)
        if grid.size >= 2:
            segments.append((grid, ug, yg))
    return segments, issues


def _linear_fit(
    segments: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    delay_min: float, tau_min: float,
) -> Tuple[float, float, np.ndarray, float, np.ndarray, np.ndarray]:
    """固定 (θ,τ) 时对全部段做线性 LS。

    模型  y = K·v + b + Σ_s d_s·a^n·1_{段s}：
    每个长缺口后新开的段带一个自由初态偏差 d_s，按 a^n 指数衰减，
    吸收"缺口后真实炉温未知、不能假定局部稳态"带来的段首瞬态；
    稳态点因此不会扭曲 K、b。返回 (K, b, d, rmse, y_true, y_pred)。
    """
    cols: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    a_values: List[float] = []
    for t, u, y in segments:
        dt_min = float(t[1] - t[0])
        a = math.exp(-dt_min / tau_min)
        a_values.append(a)
        u_del = np.interp(t - delay_min, t, u, left=u[0])
        v = _first_order_response(u_del, dt_min, tau_min)
        cols.append(v)
        ys.append(y)
    v_all = np.concatenate(cols)
    y_all = np.concatenate(ys)

    A_cols = [v_all, np.ones_like(v_all)]
    offset = 0
    for (t, _, _), a in zip(segments, a_values):
        col = np.zeros_like(v_all)
        n = t.size
        col[offset:offset + n] = a ** np.arange(n)
        A_cols.append(col)
        offset += n
    A = np.column_stack(A_cols)
    coef, *_ = np.linalg.lstsq(A, y_all, rcond=None)
    K, b = float(coef[0]), float(coef[1])
    d = coef[2:]
    pred = A @ coef
    resid = y_all - pred
    rmse = float(np.sqrt(np.mean(resid**2)))
    return K, b, d, rmse, y_all, pred


# 向后兼容别名（旧内部调用点）
def _fit_segments(
    segments: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    delay_min: float, tau_min: float,
) -> Tuple[float, float, float, np.ndarray]:
    K, b, _, rmse, _, pred = _linear_fit(segments, delay_min, tau_min)
    return K, b, rmse, pred


def fit_channel(
    setpoint_t: Sequence[float],
    setpoint_c: Sequence[float],
    channel: dict,
    excitation_min_c: float,
    long_gap_override: Optional[float] = None,
) -> dict:
    """拟合单个探头通道。channel 为 ProfileChannel 的 model_dump。"""
    cid = channel["channel_id"]
    zone = channel["zone"]
    interval_min = float(channel["sample_interval_min"])
    gap_min = float(long_gap_override if long_gap_override is not None else GAP_FACTOR * interval_min)

    t_set = np.asarray(setpoint_t, dtype=float)
    u_set = np.asarray(setpoint_c, dtype=float)

    raw = channel["samples"]
    issues: List[dict] = []

    # 乱序：按上传原始顺序检测相邻时刻下降区间
    rt = np.asarray([s["time_min"] for s in raw], dtype=float)
    ry = np.asarray([s["temp_c"] for s in raw], dtype=float)
    for i in range(1, rt.size):
        if rt[i] < rt[i - 1] - 1e-9:
            issues.append(_issue(
                "unsorted_samples",
                f"通道 {cid} 采样乱序：{rt[i - 1]:.1f} min 之后出现 {rt[i]:.1f} min",
                cid, float(rt[i - 1]), float(rt[i]),
            ))

    order = np.argsort(rt, kind="mergesort")
    t_sorted, y_sorted = rt[order], ry[order]

    # 重复时间戳：取均值并记录
    keep_t, keep_y, dup_intervals = [], [], []
    i = 0
    while i < t_sorted.size:
        j = i + 1
        while j < t_sorted.size and abs(t_sorted[j] - t_sorted[i]) <= 1e-9:
            j += 1
        keep_t.append(t_sorted[i])
        keep_y.append(float(np.mean(y_sorted[i:j])))
        if j - i > 1:
            dup_intervals.append(float(t_sorted[i]))
        i = j
    for tt in dup_intervals:
        issues.append(_issue(
            "duplicate_timestamps",
            f"通道 {cid} 在 {tt:.1f} min 存在重复采样，已取均值", cid, tt,
        ))
    t_y = np.asarray(keep_t)
    y = np.asarray(keep_y)

    base = {
        "channel_id": cid, "zone": zone, "position": channel.get("position", ""),
        "sample_interval_min": interval_min, "long_gap_threshold_min": round(gap_min, 3),
    }

    # 先按设定轴范围剔除越界点（范围问题仍照常报告），再在剩余原始点上
    # 判定激励是否充足——平坦数据不该被长缺口切碎后误报为不可辨识。
    lo, hi = float(t_set[0]), float(t_set[-1])
    inside = (t_y >= lo - 1e-9) & (t_y <= hi + 1e-9)
    for ty in t_y[~inside]:
        issues.append(_issue(
            "out_of_setpoint_range",
            f"通道 {cid} 实测时刻 {ty:.1f} min 超出设定时间轴 [{lo:.1f}, {hi:.1f}]",
            cid, float(ty),
        ))
    t_in, y_in = t_y[inside], y[inside]

    segments, gap_issues = _resample_segments(
        t_set, u_set, t_in, y_in, interval_min, gap_min, cid,
    )
    issues += gap_issues
    total_points = sum(s[0].size for s in segments)

    def _gap_list() -> list:
        return [{"start_min": i["start_min"], "end_min": i["end_min"]}
                for i in gap_issues]

    exc_y_global = float(y_in.max() - y_in.min()) if y_in.size else 0.0
    # 设定激励只在通道实测实际覆盖的窗口内统计（大轴上的平稳保温段不应
    # 因为轴上别处有斜坡而被当成激励充足）
    if t_in.size:
        win = (t_set >= float(t_in[0]) - 1e-9) & (t_set <= float(t_in[-1]) + 1e-9)
        exc_u_global = float(u_set[win].max() - u_set[win].min()) if win.any() else 0.0
    else:
        exc_u_global = 0.0
    if t_in.size < 2:
        return {
            **base, "status": "unidentifiable",
            "parameters": None, "applicable_range_c": None,
            "used_samples": int(t_in.size),
            "excluded_gap_intervals": _gap_list(),
            "issues": issues + [_issue(
                "unidentifiable",
                f"通道 {cid} 在设定时间轴内没有足够实测点，参数不可辨识",
                cid, float(t_y[0]) if t_y.size else 0.0,
                float(t_y[-1]) if t_y.size else 0.0,
            )],
        }
    if max(exc_u_global, exc_y_global) < excitation_min_c:
        return {
            **base, "status": "insufficient_excitation",
            "parameters": None,
            "applicable_range_c": (
                {"min_c": round(float(u_set.min()), 3),
                 "max_c": round(float(u_set.max()), 3)}
                if u_set.size else None
            ),
            "used_samples": total_points,
            "excluded_gap_intervals": _gap_list(),
            "issues": issues + [_issue(
                "insufficient_excitation",
                f"通道 {cid} 有效区间温度变化仅 {max(exc_u_global, exc_y_global):.1f}°C"
                f"（要求 ≥ {excitation_min_c:.1f}°C），激励不足",
                cid, float(t_in[0]) if t_in.size else 0.0,
                float(t_in[-1]) if t_in.size else 0.0,
            )],
        }

    if total_points < MIN_POINTS:
        return {
            **base, "status": "unidentifiable",
            "parameters": None,
            "applicable_range_c": None,
            "used_samples": total_points,
            "excluded_gap_intervals": _gap_list(),
            "issues": issues + [_issue(
                "unidentifiable",
                f"通道 {cid} 有效采样仅 {total_points} 点（少于 {MIN_POINTS}），参数不可辨识",
                cid, float(t_in[0]) if t_in.size else 0.0,
                float(t_in[-1]) if t_in.size else 0.0,
            )],
        }

    u_all = np.concatenate([s[1] for s in segments])
    y_all = np.concatenate([s[2] for s in segments])

    # ---- θ、τ 粗搜（θ 按采样间隔，τ 对数网格）；K、b、段初态走线性 LS ----
    longest = max((float(s[0][-1] - s[0][0]) for s in segments), default=0.0)
    theta_max = max(interval_min, 0.5 * longest)
    tau_hi = max(longest * 1.5, interval_min * 2)
    thetas = np.arange(0.0, theta_max + 1e-9, interval_min)
    taus = np.geomspace(max(interval_min * 0.5, 1e-6), tau_hi, 10)
    best = (np.inf, None)
    for th in thetas:
        for tu in taus:
            K, b, _, rmse, _, _ = _linear_fit(segments, float(th), float(tu))
            if not (K_BOUNDS[0] <= K <= K_BOUNDS[1] and BIAS_BOUNDS[0] <= b <= BIAS_BOUNDS[1]):
                continue
            if rmse < best[0]:
                best = (rmse, (float(th), float(tu)))
    if best[1] is None:  # 网格点 LS 全越界：放松边界取最优，交给可辨识性判定
        for th in thetas:
            for tu in taus:
                K, b, _, rmse, _, _ = _linear_fit(segments, float(th), float(tu))
                if rmse < best[0]:
                    best = (rmse, (float(th), float(tu)))

    # ---- Nelder-Mead 在 (θ, τ) 上精修；每步内层线性 LS ----
    def cost(x: np.ndarray) -> float:
        th, tu = x
        if th < 0.0 or tu <= 0.0:
            return 1e12
        _, _, _, rmse, _, _ = _linear_fit(segments, float(th), float(tu))
        return rmse**2

    opt = minimize(cost, np.asarray(best[1]), method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 400})
    th, tu = float(opt.x[0]), float(opt.x[1])
    K, b, d, rmse, true, pred = _linear_fit(segments, th, tu)

    # 残差与拟合优度
    t_used_arr = np.concatenate([s[0] for s in segments])
    resid = true - pred
    sse = float(np.sum(resid**2))
    sst = float(np.sum((true - true.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 1e-12 else 0.0
    max_abs = float(np.max(np.abs(resid)))

    # ---- 可辨识性：撞边界 / τ 低于采样分辨率 / 拟合度过低 ----
    boundary = (
        th >= theta_max - interval_min * 0.5
        or tu >= tau_hi - interval_min
        or tu <= interval_min * 0.5
        or K <= K_BOUNDS[0] * 1.5 or K >= K_BOUNDS[1] * 0.98
    )
    if boundary or tu < TAU_RESOLUTION * interval_min or r2 < MIN_R2:
        why = ("参数撞上搜索边界" if boundary else
               f"时间常数 {tu:.1f} min 低于采样分辨率"
               f"（< {TAU_RESOLUTION * interval_min:.1f} min）"
               if tu < TAU_RESOLUTION * interval_min else
               f"拟合优度 R²={r2:.3f} 过低")
        return {
            **base, "status": "unidentifiable",
            "parameters": None,
            "applicable_range_c": {"min_c": round(float(u_all.min()), 3),
                                   "max_c": round(float(u_all.max()), 3)},
            "used_samples": total_points,
            "excluded_gap_intervals": [
                {"start_min": i["start_min"], "end_min": i["end_min"]}
                for i in gap_issues
            ],
            "issues": issues + [_issue(
                "unidentifiable",
                f"通道 {cid} 参数不可辨识：{why}",
                cid, float(t_used_arr[0]), float(t_used_arr[-1]),
            )],
        }

    return {
        **base, "status": "identified",
        "parameters": {
            "delay_min": round(th, 3),
            "time_constant_min": round(tu, 3),
            "gain": round(K, 5),
            "bias_c": round(b, 4),
            "r_squared": round(r2, 4),
            "rmse_c": round(rmse, 4),
            "max_abs_residual_c": round(max_abs, 4),
        },
        "applicable_range_c": {"min_c": round(float(u_all.min()), 3),
                               "max_c": round(float(u_all.max()), 3)},
        "used_samples": total_points,
        "excluded_gap_intervals": [
            {"start_min": i["start_min"], "end_min": i["end_min"]}
            for i in gap_issues
        ],
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# 档案级辨识
# ---------------------------------------------------------------------------

def identify_profile(body: dict) -> dict:
    """对档案上传的全部通道执行辨识，返回档案拟合结果（不入库）。"""
    set_t = np.asarray([s["time_min"] for s in body["setpoints"]], dtype=float)
    set_u = np.asarray([s["temp_c"] for s in body["setpoints"]], dtype=float)
    excitation_min_c = float(body.get("excitation_min_c", 20.0))
    long_gap = body.get("long_gap_min")

    channels_out = []
    all_issues: List[dict] = []
    for ch in body["channels"]:
        res = fit_channel(set_t, set_u, ch, excitation_min_c, long_gap)
        channels_out.append(res)
        all_issues.extend(res["issues"])

    # 同区域多通道：只允许一个代表（取辨识成功者），重复区域在此提示
    zones: Dict[str, List[str]] = {}
    for ch in channels_out:
        zones.setdefault(ch["zone"], []).append(ch["channel_id"])
    for zone, ids in zones.items():
        if len(ids) > 1:
            all_issues.append({
                "type": "duplicate_zone",
                "channel_id": ",".join(ids),
                "start_min": None,
                "message": f"区域 {zone} 有多个通道 {ids}，预测时取拟合残差最小者",
            })

    identified = [c for c in channels_out if c["status"] == "identified"]
    status = "active" if identified else "inactive"
    return {
        "name": body["name"],
        "status": status,
        "load_mass_min_kg": body["load_mass_min_kg"],
        "load_mass_max_kg": body["load_mass_max_kg"],
        "settings": {
            "excitation_min_c": excitation_min_c,
            "long_gap_min": long_gap,
        },
        "setpoint_excitation_c": round(float(set_u.max() - set_u.min()), 3),
        "setpoint_time_min": [round(float(x), 3) for x in set_t],
        "setpoint_c": [round(float(x), 3) for x in set_u],
        "channels": channels_out,
        "issues": all_issues,
    }


def channel_for_zone(profile: dict, zone: str) -> Optional[dict]:
    """同区域存在多个辨识成功通道时，取 RMSE 最小者。"""
    cand = [c for c in profile["channels"]
            if c["zone"] == zone and c["status"] == "identified"]
    if not cand:
        return None
    return min(cand, key=lambda c: c["parameters"]["rmse_c"])


# ---------------------------------------------------------------------------
# 预测与适用性
# ---------------------------------------------------------------------------

def build_prediction(
    program, options, profile: dict, piece_zones: Dict[str, str],
) -> dict:
    """把程序设定曲线按各工件区域转换为局部预测炉温（供 analyze 使用）。"""
    from .program import assemble  # 避免模块导入环

    asm = assemble(program, options.dt_s, options.jump_tol_c)
    t = asm.t_min
    ideal = asm.temp_c

    seg_index = np.full(t.shape, -1, dtype=int)
    # 超调掩码：升温段 + 紧邻升温段的保温段（连续保温也算）。
    # 降温结束后保温初期炉温高于新设定属于降温滞后，不计超调。
    overshoot_mask = np.zeros(t.shape, dtype=bool)
    last_non_hold = ""
    for sp in asm.spans:
        sl = slice(sp.i0, sp.i1 + 1)
        seg_index[sl] = sp.index
        if sp.kind == "ramp":
            overshoot_mask[sl] = True
        elif sp.kind == "hold" and last_non_hold == "ramp":
            overshoot_mask[sl] = True
        if sp.kind != "hold":
            last_non_hold = sp.kind

    zones: Dict[str, dict] = {}
    for zone in sorted(set(piece_zones.values())):
        ch = channel_for_zone(profile, zone)
        if ch is None:
            continue
        p = ch["parameters"]
        pred = simulate_fopdt(
            t, ideal,
            delay_min=p["delay_min"], tau_min=p["time_constant_min"],
            gain=p["gain"], bias_c=p["bias_c"],
        )
        zones[zone] = {
            "channel_id": ch["channel_id"],
            "position": ch["position"],
            "predicted_c": pred,
            "parameters": p,
            "applicable_range_c": ch["applicable_range_c"],
        }

    return {
        "profile_id": profile["id"],
        "profile_version": profile.get("version", 1),
        "profile_name": profile.get("name"),
        "t_min": t,
        "setpoint_c": ideal,
        "seg_index": seg_index,
        "overshoot_mask": overshoot_mask,
        "jumps": asm.jumps,
        "zones": zones,
        "piece_zones": dict(piece_zones),
    }


def check_applicability(
    profile: dict, load_mass_kg: Optional[float], piece_zones: Dict[str, str],
    ideal_c: Optional[np.ndarray] = None,
) -> List[dict]:
    """档案适用性：装载质量与各区域温区必须落在校准范围内。

    返回拒绝原因列表；空列表表示适用。任何不满足都不得判定安全。
    """
    rejections: List[dict] = []
    lo_m, hi_m = profile["load_mass_min_kg"], profile["load_mass_max_kg"]
    if load_mass_kg is None or not (lo_m - 1e-9 <= load_mass_kg <= hi_m + 1e-9):
        rejections.append({
            "type": "load_mass_out_of_range",
            "piece": None, "zone": None,
            "value": load_mass_kg,
            "calibrated_range_kg": [lo_m, hi_m],
            "message": (
                f"装载质量 {load_mass_kg} kg 不在档案校准范围 "
                f"[{lo_m}, {hi_m}] kg 内，拒绝判定安全"
            ),
        })

    if ideal_c is not None:
        p_lo, p_hi = float(np.min(ideal_c)), float(np.max(ideal_c))
        for piece_name, zone in piece_zones.items():
            ch = channel_for_zone(profile, zone)
            if ch is None:
                rejections.append({
                    "type": "zone_not_identified",
                    "piece": piece_name, "zone": zone,
                    "message": f"工件 {piece_name} 指定区域 {zone} 无辨识成功的通道",
                })
                continue
            r = ch["applicable_range_c"]
            if p_lo < r["min_c"] - RANGE_TOL_C or p_hi > r["max_c"] + RANGE_TOL_C:
                rejections.append({
                    "type": "temperature_out_of_calibrated_range",
                    "piece": piece_name, "zone": zone,
                    "program_range_c": [round(p_lo, 2), round(p_hi, 2)],
                    "calibrated_range_c": [r["min_c"], r["max_c"]],
                    "message": (
                        f"工件 {piece_name}（区域 {zone}）程序温区 "
                        f"[{p_lo:.1f}, {p_hi:.1f}]°C 超出该通道校准范围 "
                        f"[{r['min_c']:.1f}, {r['max_c']:.1f}]°C，拒绝判定安全"
                    ),
                })
    return rejections
