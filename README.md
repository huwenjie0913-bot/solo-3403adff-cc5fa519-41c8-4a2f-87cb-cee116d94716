# 退火曲线校核服务

供玻璃热加工工作室使用的本机 REST API：录入含不同厚度熔合玻璃件的作业与窑炉程序，
按时间步求解玻璃中心—表面温度场，校核退火曲线，自动调整排程，并支持实测炉温复核。

## 技术栈

Python 3.11 · FastAPI · Pydantic · NumPy · SciPy（稀疏 LU 回代）· SQLite（stdlib）

## 运行

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload        # http://127.0.0.1:8000/docs
python -m pytest tests/              # 测试
```

数据库默认写入 `./annealing.db`，可用环境变量 `ANNEALING_DB` 覆盖。

## 物理模型

- 每个工件按**双面受热平板**沿半厚度做一维非稳态导热离散（中心对称边界、
  表面对流边界 `surface_h_w_m2k`），隐式欧拉时间积分，无条件稳定；
- 准稳态下平板内外温差 ΔT = r·L²/(2α)，故各工件允许升降温速率
  `r_max = 2α·ΔT_allow/L²`（分析摘要与排程限制均由此导出）；
- **有效保温时长**：中心温度落在退火点 ±`soak_band_c` 内的累计时间；
- **穿越退火区间速率**：中心温度处于 [应变点, 退火点] 且炉温下降时的降温速率。

## 违规判定（均给出工件与时刻）

| 类型 | 含义 |
|---|---|
| `thermal_shock` | 热冲击：\|表面−中心\| 超过工件允许温差 |
| `fast_cooling` | 过快降温：中心穿越退火区间的速率超过该工件 r_max |
| `center_not_equalized` | 中心未均温：保温段（≥应变点）结束时中心仍滞后 |
| `program_jump` | 程序跳变：相邻段设定温度不连续 |
| `kiln_max_temp` | 程序温度超过窑炉最高温 |

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/jobs` | 创建作业（材料快照入库）并按名义程序分析 |
| GET | `/jobs` / `/jobs/{id}` | 列表 / 详情（含分析历史） |
| GET | `/jobs/{id}/analysis?kind=` | 最近分析（nominal/measured/scheduled） |
| POST | `/jobs/{id}/measurements` | 上传实测炉温并复核 |
| POST | `/jobs/{id}/schedule` | 自动排程 |
| GET | `/jobs/{id}/export/timeseries.json?kind=` | 逐时刻 JSON |
| GET | `/jobs/{id}/export/program.csv?source=` | 窑炉程序 CSV（nominal/scheduled） |

### 作业录入

```json
{
  "name": "熔合作业-1",
  "pieces": [{"name": "底座", "layers": 3, "max_thickness_mm": 19.0,
              "thermal_diffusivity_m2s": 4.5e-7, "anneal_point_c": 516,
              "strain_point_c": 470, "allowed_delta_t_c": 7}],
  "program": {"start_c": 25, "segments": [
    {"kind": "ramp", "target_c": 810, "rate_c_per_min": 4},
    {"kind": "hold", "duration_min": 20},
    {"kind": "cool", "target_c": 516, "rate_c_per_min": 5},
    {"kind": "hold", "duration_min": 45},
    {"kind": "cool", "target_c": 470, "rate_c_per_min": 3}
  ]},
  "constraints": {"kiln_max_temp_c": 870, "min_step_min": 5,
                  "max_total_duration_min": 900}
}
```

段可设 `locked: true`（排程不可修改）；`constraints` 约束窑炉最高温、
最小控温步长（段时长量化单位）与总时长上限。

### 排程

`POST /jobs/{id}/schedule` 迭代降低超速段斜率、延长不足的保温，
段时长对齐最小控温步长。成功返回 `{"status": "ok", "program", "analysis"}`；
不可行返回 `{"status": "infeasible", "conflicts"}`，冲突类型：

- `locked_segment`：锁定段速率/温度超限，附带受限材料名单；
- `duration_limit`：满足材料限制所需时长超过总时长上限，指出瓶颈工件；
- `kiln_max_temp`：退火点高于窑炉上限的材料。

### 实测复核

实测炉温覆盖名义曲线参与判定；相邻采样间隔超过 `max_gap_min` 的区间
**保持未知**——不插值、不判定安全。首个缺测点之后玻璃内部状态无法
诚实推演，评估截断至缺测前，结论为 `unknown`（已发现的违规照常报告）。

## 项目结构

```
app/
  models.py     Pydantic 数据模型
  thermal.py    一维板坯导热求解器（隐式欧拉 + 稀疏 LU）
  program.py    程序装配（跳变检测）与步长量化
  analysis.py   温差/穿越速率/有效保温/违规检测
  scheduler.py  约束排程与冲突报告
  db.py         SQLite 持久化
  main.py       FastAPI 入口
tests/test_api.py
```
