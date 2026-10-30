"""全流程共用的数据结构。所有模块只依赖这里，不互相 import 对方的结构。

文件职责边界：
  - 本文件：纯数据 + 序列化，无业务逻辑、无重依赖（只用标准库）。
  - io.py：断点续传的 jsonl 读写。
  - stats.py：Wilson/McNemar/DEFF/bootstrap。
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------- 区域与样本


@dataclass
class Box:
    """像素坐标框，xyxy，原点在左上角。"""

    x0: float
    y0: float
    x1: float
    y1: float
    label: str = ""

    def __post_init__(self) -> None:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError(f"非法框: {self.as_tuple()}")

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def center(self) -> tuple:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def as_tuple(self) -> tuple:
        return (self.x0, self.y0, self.x1, self.y1)

    def clip(self, w: int, h: int) -> "Box":
        return Box(max(0, self.x0), max(0, self.y0), min(w, self.x1), min(h, self.y1), self.label)

    def scaled(self, sx: float, sy: float) -> "Box":
        return Box(self.x0 * sx, self.y0 * sy, self.x1 * sx, self.y1 * sy, self.label)

    def to_json(self) -> list:
        return [self.x0, self.y0, self.x1, self.y1, self.label]

    @staticmethod
    def from_json(v: Any) -> "Box":
        if isinstance(v, dict):
            return Box(v["x0"], v["y0"], v["x1"], v["y1"], v.get("label", ""))
        return Box(*v[:4], v[4] if len(v) > 4 else "")


@dataclass
class Sample:
    """一个 prompt 单元：图 + 问题 + 金标答案 + 可选区域标注。

    tier: 'A' 有像素级区域标注 / 'B' 只有答案 / 'C' 纯文本
    answer_type: closed | short | long | measure   —— 决定可用的判分器（见 rl/grader.py）
    """

    id: str
    image: str
    question: str
    answer: str
    dataset: str = ""
    tier: str = "B"
    answer_type: str = "short"
    boxes: List[Box] = field(default_factory=list)
    masks: List[str] = field(default_factory=list)   # 掩膜文件路径
    license: str = ""
    commercial_ok: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)  # spacing / 原始尺寸 / 来源源等

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["boxes"] = [b.to_json() for b in self.boxes]
        return d

    @staticmethod
    def from_json(d: dict) -> "Sample":
        d = dict(d)
        d["boxes"] = [Box.from_json(b) for b in d.get("boxes", [])]
        known = {f.name for f in dataclasses.fields(Sample)}
        return Sample(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------- 工具与轨迹


@dataclass
class ToolCall:
    name: str                      # crop | zoom | window | measure
    args: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"name": self.name, "args": self.args}

    @staticmethod
    def from_json(d: dict) -> "ToolCall":
        return ToolCall(d["name"], d.get("args", {}))


@dataclass
class ToolResult:
    ok: bool
    image: Optional[str] = None    # 返回图的路径（必须是"重新注入"的图，见 PROPOSAL §7.5）
    text: str = ""                 # 文本观测（measure 的读数等）
    error: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)  # crop_box / elapsed / degree

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_json(d: dict) -> "ToolResult":
        return ToolResult(**d)


@dataclass
class Step:
    """一轮：模型文本（含 think）+ 可选工具调用 + 观测。"""

    text: str = ""
    call: Optional[ToolCall] = None
    result: Optional[ToolResult] = None

    @property
    def is_tool_turn(self) -> bool:
        return self.call is not None

    def to_json(self) -> dict:
        return {
            "text": self.text,
            "call": self.call.to_json() if self.call else None,
            "result": self.result.to_json() if self.result else None,
        }

    @staticmethod
    def from_json(d: dict) -> "Step":
        return Step(
            text=d.get("text", ""),
            call=ToolCall.from_json(d["call"]) if d.get("call") else None,
            result=ToolResult.from_json(d["result"]) if d.get("result") else None,
        )


@dataclass
class Probe:
    """一次前缀探针的读数。k=0 表示未做任何工具调用时的探针。"""

    k: int
    answer: str
    score: int                     # R_ORM ∈ {-1,0,1}；解析失败为 None 时已被过滤
    parse_ok: bool = True

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class Trajectory:
    id: str
    sample_id: str = ""
    steps: List[Step] = field(default_factory=list)
    final_answer: str = ""
    probes: List[Probe] = field(default_factory=list)   # k=0..K
    delta_k: List[int] = field(default_factory=list)     # 长度 K，Δ_k
    delta_total: int = 0
    source: str = "teacher"        # teacher | student | gold
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_calls(self) -> int:
        return sum(1 for s in self.steps if s.is_tool_turn)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "sample_id": self.sample_id,
            "steps": [s.to_json() for s in self.steps],
            "final_answer": self.final_answer,
            "probes": [p.to_json() for p in self.probes],
            "delta_k": self.delta_k,
            "delta_total": self.delta_total,
            "source": self.source,
            "meta": self.meta,
        }

    @staticmethod
    def from_json(d: dict) -> "Trajectory":
        return Trajectory(
            id=d["id"],
            sample_id=d.get("sample_id", ""),
            steps=[Step.from_json(s) for s in d.get("steps", [])],
            final_answer=d.get("final_answer", ""),
            probes=[Probe(**p) for p in d.get("probes", [])],
            delta_k=d.get("delta_k", []),
            delta_total=d.get("delta_total", 0),
            source=d.get("source", "teacher"),
            meta=d.get("meta", {}),
        )


# ---------------------------------------------------------------- 校验结论


@dataclass
class Verdict:
    """synth/validate.py 的输出：一条轨迹是否可用于训练。"""

    traj_id: str
    passed: bool
    reason: str = ""
    checks: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_json(d: dict) -> "Verdict":
        return Verdict(**d)
