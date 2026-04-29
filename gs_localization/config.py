from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RetrievalConfig:
    top_k: int = 10


@dataclass
class MatchingConfig:
    max_descriptor_distance: float = 1.0
    min_matches: int = 6


@dataclass
class PnPConfig:
    min_inliers: int = 6
    reprojection_error: float = 8.0
    confidence: float = 0.99
    max_iterations: int = 100


@dataclass
class RefinementConfig:
    num_iters: int = 12
    learning_rate: float = 0.08
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    opacity_threshold: float = 0.15
    gradient_quantile: float = 0.65
    blur_sigma_start: float = 1.8
    blur_sigma_end: float = 0.0
    divergence_ratio: float = 1.5
    finite_difference_eps: float = 1e-3
    max_resolution_side: int = 10000  # needs adjustment for lower-end machines
    max_loss_pixels: int = 4096
    min_relative_improvement: float = 0.002
    accept_loss_ratio: float = 0.995
    absolute_accept_loss: float = 0.9
    retry_loss_threshold: float = 0.15
    max_init_retries: int = 3


@dataclass
class EvaluationConfig:
    success_translation_threshold: float = 0.25
    success_rotation_threshold_deg: float = 5.0


@dataclass
class PipelineConfig:
    scene_dir: Path
    debug_dir: Path | None = None
    gs_model_dir: Path | None = None
    gs_repo_dir: Path | None = None
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    pnp: PnPConfig = field(default_factory=PnPConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_scene_dir(
        cls,
        scene_dir: str | Path,
        debug_dir: str | Path | None = None,
        gs_model_dir: str | Path | None = None,
        gs_repo_dir: str | Path | None = None,
    ) -> "PipelineConfig":
        return cls(
            scene_dir=Path(scene_dir),
            debug_dir=Path(debug_dir) if debug_dir else None,
            gs_model_dir=Path(gs_model_dir) if gs_model_dir else None,
            gs_repo_dir=Path(gs_repo_dir) if gs_repo_dir else None,
        )
