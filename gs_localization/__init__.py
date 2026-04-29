"""Gaussian Splatting based localization prototype."""

from .config import PipelineConfig
from .data import LocalizationResult, PoseEstimate, SceneAssets
from .pipeline import LocalizationPipeline

__all__ = [
    "LocalizationPipeline",
    "LocalizationResult",
    "PipelineConfig",
    "PoseEstimate",
    "SceneAssets",
]
