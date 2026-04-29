from __future__ import annotations

import math
import sys
from pathlib import Path

try:
    import torch
except Exception:
    torch = None

from .data import CameraIntrinsics, ImageTensor


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _focal_to_fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(pixels / (2.0 * focal))


class GaussianSplattingRenderer:
    def __init__(
        self,
        renderer_config: dict,
        intrinsics: CameraIntrinsics | None = None,
        gs_model_dir: Path | None = None,
        gs_repo_dir: Path | None = None,
    ) -> None:
        self.config = renderer_config or {}
        self.intrinsics = intrinsics
        self.width = int(
            self.config.get("width", intrinsics.width if intrinsics else 16)
        )
        self.height = int(
            self.config.get("height", intrinsics.height if intrinsics else 16)
        )
        self.frequency = float(self.config.get("frequency", 0.35))
        self.opacity_bias = float(self.config.get("opacity_bias", 0.6))
        self.gs_model_dir = gs_model_dir
        self.gs_repo_dir = gs_repo_dir
        self.backend_error: str | None = None
        self.gs_backend = None
        self._maybe_initialize_backend()

    def render(
        self,
        camera_pose: list[list[float]],
        target_width: int | None = None,
        target_height: int | None = None,
    ) -> tuple[ImageTensor, list[list[float]]]:
        if self.gs_backend is None:
            raise RuntimeError(
                "The Gaussian Splatting backend is required for rendering."
            )
        return self.gs_backend.render(
            camera_pose,
            target_width=target_width or self.width,
            target_height=target_height or self.height,
        )

    def render_torch(
        self,
        camera_pose: torch.Tensor,
        target_width: int | None = None,
        target_height: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.gs_backend is None:
            raise RuntimeError(
                "Differentiable rendering requires the real gaussian-splatting backend."
            )
        return self.gs_backend.render_torch(
            camera_pose,
            target_width=target_width or self.width,
            target_height=target_height or self.height,
        )

    def _maybe_initialize_backend(self) -> None:
        if self.gs_model_dir is None or self.gs_repo_dir is None:
            self.backend_error = (
                "gaussian-splatting repo or checkpoint directory not configured"
            )
            return
        try:
            self.gs_backend = _GaussianSplattingBackend(
                self.gs_model_dir, self.gs_repo_dir, self.intrinsics
            )
        except Exception as exc:
            self.backend_error = str(exc)
            self.gs_backend = None


class _GaussianSplattingBackend:
    def __init__(
        self, gs_model_dir: Path, gs_repo_dir: Path, intrinsics: CameraIntrinsics | None
    ) -> None:
        self.gs_model_dir = gs_model_dir
        self.gs_repo_dir = gs_repo_dir
        self.intrinsics = intrinsics

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available for gaussian-splatting rendering."
            )
        if not gs_repo_dir.exists():
            raise RuntimeError(f"gaussian-splatting repo not found: {gs_repo_dir}")

        sys.path.insert(0, str(gs_repo_dir))
        try:
            from gaussian_renderer import render as gs_render  # type: ignore
            from diff_gaussian_rasterization import (  # type: ignore
                GaussianRasterizationSettings,
                GaussianRasterizer,
            )
            from scene.gaussian_model import GaussianModel  # type: ignore
            from utils.general_utils import build_scaling_rotation, strip_symmetric  # type: ignore
            from utils.graphics_utils import getProjectionMatrix  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Could not import gaussian-splatting backend. "
                "The CUDA extensions likely need to be built and plyfile installed. "
                f"Original import error: {exc!r}"
            ) from exc

        self.gs_render = gs_render
        self.GaussianRasterizationSettings = GaussianRasterizationSettings
        self.GaussianRasterizer = GaussianRasterizer
        self.GaussianModel = GaussianModel
        self.build_scaling_rotation = build_scaling_rotation
        self.strip_symmetric = strip_symmetric
        self.getProjectionMatrix = getProjectionMatrix
        self.bg_color = torch.tensor(
            [0.0, 0.0, 0.0], dtype=torch.float32, device="cuda"
        )
        self.model = self._load_model()
        self.pipeline = type(
            "PipelineParamsLite",
            (),
            {
                "debug": False,
                "antialiasing": False,
                "compute_cov3D_python": False,
                "convert_SHs_python": False,
            },
        )()

    def _load_model(self):
        checkpoint_path = self.gs_model_dir / "checkpoint" / "chkpnt-30000.pth"
        if not checkpoint_path.exists():
            raise RuntimeError(f"Expected checkpoint not found at {checkpoint_path}")

        state, _ = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
        sh_degree = int(state[0])
        model = self.GaussianModel(sh_degree)
        model.active_sh_degree = int(state[0])
        model._xyz = state[1].detach().cuda()
        model._features_dc = state[2].detach().cuda()
        model._features_rest = state[3].detach().cuda()
        model._scaling = state[4].detach().cuda()
        model._rotation = state[5].detach().cuda()
        model._opacity = state[6].detach().cuda()
        model.max_radii2D = state[7].detach().cuda()
        model.xyz_gradient_accum = state[8].detach().cuda()
        model.denom = state[9].detach().cuda()
        model.optimizer = None
        model.spatial_lr_scale = float(state[11])
        return model

    def _make_camera(
        self,
        camera_pose: list[list[float]],
        target_width: int,
        target_height: int,
    ):
        if self.intrinsics is None:
            raise RuntimeError(
                "Camera intrinsics are required for gaussian-splatting rendering."
            )

        world_view_transform = torch.tensor(
            camera_pose, dtype=torch.float32, device="cuda"
        ).transpose(0, 1)
        fovx = _focal_to_fov(self.intrinsics.fx, self.intrinsics.width)
        fovy = _focal_to_fov(self.intrinsics.fy, self.intrinsics.height)
        projection = (
            self.getProjectionMatrix(
                znear=0.01,
                zfar=100.0,
                fovX=fovx,
                fovY=fovy,
            )
            .transpose(0, 1)
            .cuda()
        )
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
        )
        camera_center = torch.inverse(world_view_transform)[3, :3]
        return type(
            "LocalizationMiniCam",
            (),
            {
                "FoVx": fovx,
                "FoVy": fovy,
                "image_width": target_width,
                "image_height": target_height,
                "world_view_transform": world_view_transform,
                "full_proj_transform": full_proj_transform,
                "camera_center": camera_center,
                "image_name": "query",
            },
        )()

    def render(
        self,
        camera_pose: list[list[float]],
        target_width: int,
        target_height: int,
    ) -> tuple[ImageTensor, list[list[float]]]:
        camera = self._make_camera(camera_pose, target_width, target_height)
        with torch.no_grad():
            output = self.gs_render(camera, self.model, self.pipeline, self.bg_color)
        rendered = output["render"].detach().clamp(0.0, 1.0).cpu()
        depth = output["depth"].detach().cpu()

        image: ImageTensor = []
        opacity_map: list[list[float]] = []
        for y in range(rendered.shape[1]):
            row = []
            opacity_row = []
            for x in range(rendered.shape[2]):
                row.append([float(rendered[channel, y, x]) for channel in range(3)])
                opacity_row.append(1.0 if float(depth[0, y, x]) > 0.0 else 0.0)
            image.append(row)
            opacity_map.append(opacity_row)
        return image, opacity_map

    def render_torch(
        self,
        camera_pose: torch.Tensor,
        target_width: int,
        target_height: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.intrinsics is None:
            raise RuntimeError(
                "Camera intrinsics are required for gaussian-splatting rendering."
            )

        pose = camera_pose.to(device="cuda", dtype=torch.float32)
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        means3d = self.model.get_xyz @ rotation.transpose(0, 1) + translation.unsqueeze(
            0
        )

        scaling = self.model.get_scaling
        world_rotation = self.model.get_rotation
        covariance_factor = self.build_scaling_rotation(scaling, world_rotation)
        covariance_world = covariance_factor @ covariance_factor.transpose(1, 2)
        rotation_batch = rotation.unsqueeze(0).expand(covariance_world.shape[0], -1, -1)
        covariance_camera = (
            rotation_batch @ covariance_world @ rotation_batch.transpose(1, 2)
        )
        covariance_precomp = self.strip_symmetric(covariance_camera)

        fovx = _focal_to_fov(self.intrinsics.fx, self.intrinsics.width)
        fovy = _focal_to_fov(self.intrinsics.fy, self.intrinsics.height)
        viewmatrix = torch.eye(4, dtype=torch.float32, device="cuda")
        projmatrix = (
            self.getProjectionMatrix(
                znear=0.01,
                zfar=100.0,
                fovX=fovx,
                fovY=fovy,
            )
            .transpose(0, 1)
            .to(device="cuda", dtype=torch.float32)
        )
        raster_settings = self.GaussianRasterizationSettings(
            image_height=int(target_height),
            image_width=int(target_width),
            tanfovx=math.tan(fovx * 0.5),
            tanfovy=math.tan(fovy * 0.5),
            bg=self.bg_color,
            scale_modifier=1.0,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            sh_degree=self.model.active_sh_degree,
            campos=torch.zeros(3, dtype=torch.float32, device="cuda"),
            prefiltered=False,
            debug=False,
            antialiasing=False,
        )
        rasterizer = self.GaussianRasterizer(raster_settings=raster_settings)
        means2d = torch.zeros_like(
            means3d, dtype=means3d.dtype, requires_grad=True, device="cuda"
        )
        rendered, _, depth = rasterizer(
            means3D=means3d,
            means2D=means2d,
            shs=self.model.get_features,
            colors_precomp=None,
            opacities=self.model.get_opacity,
            scales=None,
            rotations=None,
            cov3D_precomp=covariance_precomp,
        )
        rendered = rendered.clamp(0.0, 1.0)
        opacity = (depth > 0.0).float()
        return rendered, opacity
