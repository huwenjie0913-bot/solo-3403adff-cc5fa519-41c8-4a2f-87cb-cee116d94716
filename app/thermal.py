"""一维板坯非稳态导热求解器。

将熔合玻璃件视为双面受热的平板，沿半厚度方向离散：
- 中心节点：对称边界（绝热）
- 表面节点：与炉气的对流边界（换热系数 h 可配）

时间积分采用隐式欧拉（Backward Euler），对任意时间步无条件稳定，
适合窑炉程序这种长达数小时、时间步远大于热扩散特征步长的场景。
矩阵在时间上不变，预先 LU 分解后逐步回代。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import factorized

K_GLASS_W_MK = 1.05  # 钠钙玻璃导热系数近似值 W/(m·K)


def simulate_piece(
    alpha_m2s: float,
    thickness_mm: float,
    dt_s: float,
    kiln_c: np.ndarray,
    n_nodes: int = 61,
    h_w_m2k: float = 200.0,
    t_init_c: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """按给定炉温曲线求解玻璃中心与表面温度。

    参数:
        alpha_m2s: 热扩散率 α (m²/s)
        thickness_mm: 工件总厚度 (mm)，按双面受热取半厚度建模
        dt_s: 时间步长 (s)，kiln_c 为等间距采样
        kiln_c: 炉温序列 (°C)
        n_nodes: 半厚度方向节点数
        h_w_m2k: 表面对流换热系数
        t_init_c: 初始温度，缺省取 kiln_c[0]

    返回:
        (center_c, surface_c)：与 kiln_c 等长的中心/表面温度序列
    """
    kiln_c = np.asarray(kiln_c, dtype=float)
    nt = kiln_c.size
    if nt == 0:
        return np.empty(0), np.empty(0)

    L = thickness_mm / 1000.0 / 2.0  # 半厚度 m
    dx = L / (n_nodes - 1)
    beta = h_w_m2k * dx / K_GLASS_W_MK  # 网格 Biot 数

    # 离散算子 M：dT/dt = α/dx² · (M·T + b·T_kiln)
    main = np.full(n_nodes, -2.0)
    main[-1] = -2.0 * (1.0 + beta)  # 表面对流边界（半控制体）
    lower = np.ones(n_nodes - 1)
    lower[-1] = 2.0
    upper = np.ones(n_nodes - 1)
    upper[0] = 2.0  # 中心对称边界
    M = diags([lower, main, upper], [-1, 0, 1], format="csc")
    b = np.zeros(n_nodes)
    b[-1] = 2.0 * beta

    r = alpha_m2s * dt_s / dx**2
    A = (eye(n_nodes, format="csc") - r * M).tocsc()
    solve = factorized(A)
    src = r * b

    T = np.full(n_nodes, kiln_c[0] if t_init_c is None else t_init_c)
    center = np.empty(nt)
    surface = np.empty(nt)
    center[0] = T[0]
    surface[0] = T[-1]
    for n in range(1, nt):
        T = solve(T + src * kiln_c[n])
        center[n] = T[0]
        surface[n] = T[-1]
    return center, surface
