from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


Matrix = list[list[float]]
Vector = list[float]
ImageTensor = list[list[list[float]]]


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class QueryImage:
    path: Path
    image: ImageTensor
    intrinsics: CameraIntrinsics | None = None


@dataclass
class ReferenceImage:
    name: str
    path: Path
    global_descriptor: Vector
    local_features: list[dict[str, Any]]
    world_points: list[Vector]
    pose: Matrix | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneAssets:
    scene_dir: Path
    references: list[ReferenceImage]
    renderer_config: dict[str, Any]
    intrinsics: CameraIntrinsics | None = None
    query_poses: dict[str, Matrix] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PoseEstimate:
    matrix: Matrix
    inliers: int = 0
    success: bool = True
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalizationResult:
    query_path: Path
    init_pose: PoseEstimate
    refined_pose: PoseEstimate
    init_inliers: int
    refinement_success: bool
    final_loss: float | None
    timings: dict[str, float]
    debug_artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
