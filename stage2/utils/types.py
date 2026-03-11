"""Shared dataclasses used across all stages."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ObjectSize:
    w: float   # width  (X)
    h: float   # height (Y)
    d: float   # depth  (Z)


@dataclass
class ObjectPlacement:
    name:       str
    layer:      str                    # foreground | midground | background
    x:          float = 0.0
    y:          float = 0.0
    z:          float = 0.0
    rot_y:      float = 0.0            # rotation around Y axis (degrees)
    size:       Optional[ObjectSize] = None
    surface:    Optional[str] = None   # floor | wall | on:<object>
    anchor_ref: Optional[str] = None   # semantic anchor from scene graph
    floating:   bool = False            # gravity=false mode
    environment_geometry: bool = False   # w or d > 3.0m; exempt from collision/bounds


@dataclass
class SceneGraph:
    scene_type:  str
    gravity:     bool
    camera_facing: str
    relations:   list[dict]            # raw LLM output, algo consumes this


@dataclass
class VLMIssue:
    object:      str
    group:       str                   # occlusion | gravity | collision | theme | composition
    severity:    str                   # critical | warning | minor
    description: str
    fix_hint:    str


@dataclass
class VLMResult:
    issues:            list[VLMIssue]
    composition_score: float
    approved:          bool


# ── Q-based validation types ────────────────────────────────────────────────

@dataclass
class Question:
    id:         str
    text:       str
    object_ids: Optional[list[str]] = None


@dataclass
class QuestionResult:
    question_id:  str
    passed:       bool
    observation: str


@dataclass
class QValidationResult:
    results:             list[QuestionResult]
    approved:            bool
    failed_question_ids: list[str]
    parse_failed:        bool = False


@dataclass
class DeltaResult:
    deltas: dict[str, tuple[float, float, float]]  # obj_id -> (dx, dy, dz)
