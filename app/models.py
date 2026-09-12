"""API 数据模型（Pydantic）。"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class PieceSpec(BaseModel):
    """单个熔合玻璃件的材料快照。一次作业可包含多个不同厚度的工件。"""

    name: str = Field(min_length=1, description="工件名称")
    layers: int = Field(ge=1, description="玻璃层数")
    max_thickness_mm: float = Field(gt=0, description="最大总厚度 mm")
    thermal_diffusivity_m2s: float = Field(gt=0, description="热扩散率 α, m²/s")
    anneal_point_c: float = Field(description="退火点 °C")
    strain_point_c: float = Field(description="应变点 °C")
    allowed_delta_t_c: float = Field(gt=0, description="允许的内外温差 °C")

    @model_validator(mode="after")
    def _check_points(self) -> "PieceSpec":
        if self.strain_point_c >= self.anneal_point_c:
            raise ValueError("strain_point_c 必须低于 anneal_point_c")
        return self


class Segment(BaseModel):
    """窑炉程序段：升温(ramp) / 保温(hold) / 降温(cool)。

    ramp/cool 需要 target_c 与 rate_c_per_min（速率取正值，方向由目标温度决定）；
    hold 需要 duration_min，可选 target_c（缺省表示保持上一段结束温度，
    若与当前温度之差超过跳变容差则记为程序跳变）。
    locked=True 的段在排程时不可修改。
    """

    kind: Literal["ramp", "hold", "cool"]
    target_c: Optional[float] = None
    rate_c_per_min: Optional[float] = Field(default=None, gt=0)
    duration_min: Optional[float] = Field(default=None, gt=0)
    locked: bool = False

    @model_validator(mode="after")
    def _check_fields(self) -> "Segment":
        if self.kind == "hold":
            if self.duration_min is None:
                raise ValueError("hold 段需要 duration_min")
        else:
            if self.target_c is None or self.rate_c_per_min is None:
                raise ValueError("ramp/cool 段需要 target_c 与 rate_c_per_min")
        return self


class KilnProgram(BaseModel):
    start_c: float = Field(default=25.0, description="装炉温度 °C")
    segments: List[Segment] = Field(min_length=1)


class Constraints(BaseModel):
    kiln_max_temp_c: float = Field(default=1000.0, description="窑炉最高温 °C")
    min_step_min: float = Field(default=5.0, gt=0, description="最小控温步长 min")
    max_total_duration_min: Optional[float] = Field(default=None, gt=0, description="总时长上限 min")


class AnalysisOptions(BaseModel):
    dt_s: float = Field(default=30.0, gt=0, description="时间步长 s")
    nodes: int = Field(default=61, ge=11, description="半厚度方向节点数")
    surface_h_w_m2k: float = Field(default=200.0, gt=0, description="表面对流换热系数 W/(m²·K)")
    soak_band_c: float = Field(default=5.0, gt=0, description="有效保温判定的退火点温度带宽 ±°C")
    equalization_tol_c: Optional[float] = Field(default=None, description="均温容差 °C，缺省取允许温差的一半")
    jump_tol_c: float = Field(default=1.0, gt=0, description="程序跳变容差 °C")


class JobCreate(BaseModel):
    name: str = Field(min_length=1)
    pieces: List[PieceSpec] = Field(min_length=1)
    program: KilnProgram
    constraints: Constraints = Field(default_factory=Constraints)
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)


class MeasurementPoint(BaseModel):
    time_min: float = Field(ge=0)
    temp_c: float


class MeasurementBatch(BaseModel):
    """实测炉温。相邻采样间隔超过 max_gap_min 的区间视为缺测，保持未知。"""

    samples: List[MeasurementPoint] = Field(min_length=2)
    max_gap_min: float = Field(default=5.0, gt=0)
    replace: bool = Field(default=True, description="True 覆盖旧实测，False 追加")


class ScheduleRequest(BaseModel):
    safety_factor: float = Field(default=0.9, gt=0, le=1.0, description="速率安全裕度系数")
    max_iterations: int = Field(default=40, ge=1, le=200)
