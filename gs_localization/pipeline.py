import argparse
import json
import time
from pathlib import Path

from .config import PipelineConfig
from .data import ImageTensor, LocalizationResult, PoseEstimate
from .features import (
    FeatureExtractor,
    downsample_image,
    h5py as features_h5py,
    hloc_extract_features,
    load_image,
    np as features_np,
)
from .init_pose import InitialPoseEstimator, cv2, np as init_pose_np
from .refine import PoseRefiner
from .renderer import GaussianSplattingRenderer
from .refine import F as torch_functional, torch as refine_torch
from .sfm import SfMDatabase, load_scene_assets, pycolmap


class LocalizationPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._validate_runtime_requirements(config)
        self.assets = load_scene_assets(config.scene_dir)
        self.database = SfMDatabase(self.assets)
        self.feature_extractor = FeatureExtractor(
            cache_dir=config.scene_dir / ".gs_localization_cache"
        )
        self.initial_estimator = InitialPoseEstimator(
            self.database,
            config.retrieval,
            config.matching,
            config.pnp,
        )
        scene_name = config.scene_dir.name
        inferred_gs_model_dir = config.gs_model_dir or (
            config.scene_dir.parent / f"{scene_name}-gs"
        )
        inferred_gs_repo_dir = config.gs_repo_dir or (
            config.scene_dir.parent / "gaussian-splatting"
        )
        self.renderer = GaussianSplattingRenderer(
            self.assets.renderer_config,
            self.assets.intrinsics,
            gs_model_dir=(
                inferred_gs_model_dir if inferred_gs_model_dir.exists() else None
            ),
            gs_repo_dir=inferred_gs_repo_dir if inferred_gs_repo_dir.exists() else None,
        )
        if self.renderer.gs_backend is None:
            raise RuntimeError(
                "The Gaussian Splatting backend is required for localization. "
                f"Backend initialization failed: {self.renderer.backend_error}"
            )
        self.pose_refiner = PoseRefiner(self.renderer, config.refinement)
        print(
            "[init] scene=",
            config.scene_dir,
            "references=",
            len(self.assets.references),
            "renderer_backend=",
            "real",
        )

    def _validate_runtime_requirements(self, config: PipelineConfig) -> None:
        missing: list[str] = []
        if hloc_extract_features is None:
            missing.append("hloc")
        if features_h5py is None:
            missing.append("h5py")
        if features_np is None or init_pose_np is None:
            missing.append("numpy")
        if cv2 is None:
            missing.append("opencv-python")
        if refine_torch is None or torch_functional is None:
            missing.append("torch")
        if missing:
            raise RuntimeError(
                "Missing required dependencies for localization: "
                + ", ".join(sorted(set(missing)))
            )
        scene_path = config.scene_dir
        if (scene_path / "sparse" / "0").exists() and pycolmap is None:
            raise RuntimeError("pycolmap is required to load COLMAP-backed scenes.")

    def _should_skip_refinement(self, query_image, init_pose: PoseEstimate) -> bool:
        return init_pose.source == "retrieval-pose"

    def _prepare_refinement_query(
        self, query_image, max_side: int | None = None
    ) -> tuple[list, bool]:
        active_max_side = max_side or self.config.refinement.max_resolution_side
        height = len(query_image)
        width = len(query_image[0]) if height else 0
        original_max_side = max(height, width) if height else 0
        if original_max_side <= active_max_side:
            return query_image, False
        return downsample_image(query_image, max_side=active_max_side), True

    def render_pose_image(
        self,
        pose: list[list[float]],
        target_width: int,
        target_height: int,
    ) -> ImageTensor:
        rendered, _ = self.renderer.render(
            pose,
            target_width=target_width,
            target_height=target_height,
        )
        return rendered

    def _refinement_needs_retry(
        self,
        refined_pose: PoseEstimate,
        final_loss: float | None,
    ) -> bool:
        if not refined_pose.success:
            return True
        if final_loss is None:
            return False
        return final_loss > self.config.refinement.retry_loss_threshold

    def _collect_reference_pose_candidates(
        self,
        candidates,
        matches: dict[str, list[tuple[int, int]]],
        local_features,
        active_intrinsics,
    ) -> list[dict]:
        reference_pose_candidates: list[dict] = []
        for candidate in candidates:
            reference_matches = matches.get(candidate.name)
            if not reference_matches:
                continue
            reference_points_2d, reference_points_3d, reference_provenance = (
                self.database.assemble_correspondences(
                    local_features,
                    {candidate.name: reference_matches},
                )
            )
            candidate_pose = self.initial_estimator.estimate_pose_pnp(
                reference_points_2d,
                reference_points_3d,
                active_intrinsics,
            )
            if not candidate_pose.success:
                continue
            reference_pose_candidates.append(
                {
                    "reference": candidate.name,
                    "pose": candidate_pose,
                    "points_2d": reference_points_2d,
                    "provenance": reference_provenance,
                    "match_count": len(reference_points_2d),
                }
            )
        reference_pose_candidates.sort(
            key=lambda item: (
                item["pose"].inliers,
                item["match_count"],
            ),
            reverse=True,
        )
        return reference_pose_candidates

    def localize(self, query_image_path: str | Path) -> LocalizationResult:
        start = time.perf_counter()
        query = load_image(query_image_path)
        query_h = len(query.image)
        query_w = len(query.image[0]) if query_h else 0
        print(f"[localize] query={query.path} size={query_w}x{query_h}")

        retrieval_start = time.perf_counter()
        print("[retrieval] preparing global descriptors for references")
        self.feature_extractor.ensure_reference_globals(self.assets.references)
        print("[retrieval] extracting query descriptors")
        global_descriptor = self.feature_extractor.extract_query_global(query)
        local_features = self.feature_extractor.extract_query_local(query)
        print(f"[retrieval] query local features={len(local_features)}")
        candidates = self.initial_estimator.retrieve_candidates(global_descriptor)
        print(
            "[retrieval] top candidates:",
            [candidate.name for candidate in candidates[: min(5, len(candidates))]],
        )
        self.feature_extractor.ensure_reference_locals(candidates)
        matches = self.initial_estimator.match_features(local_features, candidates)
        points_2d, points_3d, provenance = self.database.assemble_correspondences(
            local_features, matches
        )
        retrieval_time = time.perf_counter() - retrieval_start
        match_summary = {name: len(pairs) for name, pairs in matches.items()}
        print("[retrieval] per-reference matches:", match_summary)
        print(
            "[retrieval] cache stats:",
            {
                "hits": dict(self.feature_extractor.cache_hits),
                "misses": dict(self.feature_extractor.cache_misses),
            },
        )
        print(
            f"[retrieval] total correspondences={len(points_2d)} elapsed={retrieval_time:.3f}s"
        )

        pnp_start = time.perf_counter()
        active_intrinsics = self.assets.intrinsics or query.intrinsics
        reference_pose_candidates = self._collect_reference_pose_candidates(
            candidates,
            matches,
            local_features,
            active_intrinsics,
        )
        init_pose = self.initial_estimator.estimate_pose_pnp(
            points_2d, points_3d, active_intrinsics
        )
        if not init_pose.success and matches:
            best_pose = init_pose
            best_reference = None
            best_reference_points_2d: list[list[float]] = []
            best_reference_provenance: list[dict] = []
            for candidate_info in reference_pose_candidates:
                candidate_pose = candidate_info["pose"]
                if candidate_pose.inliers > best_pose.inliers:
                    best_pose = candidate_pose
                    best_reference = candidate_info["reference"]
                    best_reference_points_2d = candidate_info["points_2d"]
                    best_reference_provenance = candidate_info["provenance"]
            if best_pose.success:
                print(
                    "[pnp] aggregate correspondences failed; using best single-reference PnP:",
                    best_reference,
                    f"inliers={best_pose.inliers}",
                )
                init_pose = PoseEstimate(
                    matrix=best_pose.matrix,
                    inliers=best_pose.inliers,
                    success=True,
                    source=best_pose.source,
                    metadata={
                        **best_pose.metadata,
                        "reference_image": best_reference,
                        "match_count": len(best_reference_points_2d),
                        "pnp_strategy": "single-reference-fallback",
                    },
                )
                points_2d = best_reference_points_2d
                provenance = best_reference_provenance
        if not init_pose.success and candidates and candidates[0].pose is not None:
            print(
                "[pnp] PnP failed, falling back to retrieved reference pose:",
                candidates[0].name,
            )
            init_pose = PoseEstimate(
                matrix=candidates[0].pose,
                inliers=0,
                success=True,
                source="retrieval-pose",
                metadata={"reference_image": candidates[0].name},
            )
        pnp_time = time.perf_counter() - pnp_start
        print(
            f"[pnp] success={init_pose.success} source={init_pose.source} "
            f"inliers={init_pose.inliers} elapsed={pnp_time:.3f}s"
        )

        if not init_pose.success:
            print("[localize] initialization failed, returning without refinement")
            return LocalizationResult(
                query_path=Path(query_image_path),
                init_pose=init_pose,
                refined_pose=PoseEstimate(
                    matrix=init_pose.matrix,
                    inliers=init_pose.inliers,
                    success=False,
                    source="init-pose-failed",
                    metadata={"reason": "initial_pose_failed"},
                ),
                init_inliers=init_pose.inliers,
                refinement_success=False,
                final_loss=None,
                timings={
                    "feature_and_retrieval": retrieval_time,
                    "pnp": pnp_time,
                    "refinement": 0.0,
                    "total": time.perf_counter() - start,
                },
                debug_artifacts={},
                metadata={
                    "candidate_names": [candidate.name for candidate in candidates],
                    "match_count": len(points_2d),
                    "provenance": provenance,
                    "renderer_backend_error": self.renderer.backend_error,
                    "hloc_global_error": self.feature_extractor.hloc_global_error,
                    "hloc_local_error": self.feature_extractor.hloc_local_error,
                    "feature_backends": sorted(
                        {
                            reference.metadata.get("feature_backend", "unknown")
                            for reference in candidates
                        }
                    ),
                },
            )

        if self._should_skip_refinement(query.image, init_pose):
            print(
                "[refine] skipped:",
                {
                    "reason": "retrieval_only",
                    "init_source": init_pose.source,
                    "query_size": [query_w, query_h],
                    "backend": "real",
                },
            )
            refined_pose = PoseEstimate(
                matrix=init_pose.matrix,
                inliers=init_pose.inliers,
                success=True,
                source="refinement-skipped",
                metadata={"reason": "retrieval_only_or_large_query"},
            )
            final_loss = None
            debug_artifacts = {}
            refinement_time = 0.0
        else:
            refinement_query, was_downscaled = self._prepare_refinement_query(
                query.image
            )
            if was_downscaled:
                refine_h = len(refinement_query)
                refine_w = len(refinement_query[0]) if refine_h else 0
                print(
                    f"[refine] downscaling query for refinement: "
                    f"{query_w}x{query_h} -> {refine_w}x{refine_h}"
                )
            print("[refine] starting photometric refinement")
            refine_start = time.perf_counter()
            refined_pose, final_loss, debug_artifacts = self.pose_refiner.refine_pose(
                refinement_query,
                init_pose.matrix,
                debug_dir=self.config.debug_dir,
                intrinsics=query.intrinsics,
            )
            if self._refinement_needs_retry(refined_pose, final_loss):
                current_reference = init_pose.metadata.get("reference_image")
                attempts = 0
                best_retry_pose = refined_pose
                best_retry_loss = final_loss
                best_retry_artifacts = debug_artifacts
                best_retry_init = init_pose
                for candidate_info in reference_pose_candidates:
                    if attempts >= self.config.refinement.max_init_retries:
                        break
                    if candidate_info["reference"] == current_reference:
                        continue
                    attempts += 1
                    candidate_pose = candidate_info["pose"]
                    print(
                        "[refine] retrying from alternative init:",
                        candidate_info["reference"],
                        f"inliers={candidate_pose.inliers}",
                    )
                    retry_pose, retry_loss, retry_artifacts = (
                        self.pose_refiner.refine_pose(
                            refinement_query,
                            candidate_pose.matrix,
                            debug_dir=self.config.debug_dir,
                            intrinsics=query.intrinsics,
                        )
                    )
                    print(
                        f"[refine] retry result reference={candidate_info['reference']} "
                        f"success={retry_pose.success} final_loss={retry_loss}"
                    )
                    if retry_loss is None:
                        continue
                    if best_retry_loss is None or retry_loss < best_retry_loss:
                        best_retry_pose = retry_pose
                        best_retry_loss = retry_loss
                        best_retry_artifacts = retry_artifacts
                        best_retry_init = PoseEstimate(
                            matrix=candidate_pose.matrix,
                            inliers=candidate_pose.inliers,
                            success=True,
                            source=candidate_pose.source,
                            metadata={
                                **candidate_pose.metadata,
                                "reference_image": candidate_info["reference"],
                                "match_count": candidate_info["match_count"],
                                "pnp_strategy": "retry-reference-init",
                            },
                        )
                if best_retry_loss is not None and (
                    final_loss is None or best_retry_loss < final_loss
                ):
                    print(
                        "[refine] selected alternative initialization result:",
                        best_retry_init.metadata.get("reference_image"),
                        f"final_loss={best_retry_loss}",
                    )
                    init_pose = best_retry_init
                    points_2d = next(
                        (
                            item["points_2d"]
                            for item in reference_pose_candidates
                            if item["reference"]
                            == best_retry_init.metadata.get("reference_image")
                        ),
                        points_2d,
                    )
                    provenance = next(
                        (
                            item["provenance"]
                            for item in reference_pose_candidates
                            if item["reference"]
                            == best_retry_init.metadata.get("reference_image")
                        ),
                        provenance,
                    )
                    refined_pose = best_retry_pose
                    final_loss = best_retry_loss
                    debug_artifacts = best_retry_artifacts
            refinement_time = time.perf_counter() - refine_start
            print(
                f"[refine] success={refined_pose.success} final_loss={final_loss} "
                f"elapsed={refinement_time:.3f}s"
            )

        if not refined_pose.success:
            print("[refine] refinement failed, falling back to init pose")
            refined_pose = PoseEstimate(
                matrix=init_pose.matrix,
                inliers=init_pose.inliers,
                success=False,
                source="refinement-fallback",
                metadata={"reason": "refinement_failed"},
            )

        print(
            f"[done] total={time.perf_counter() - start:.3f}s "
            f"init_source={init_pose.source} refined_source={refined_pose.source}"
        )

        return LocalizationResult(
            query_path=Path(query_image_path),
            init_pose=init_pose,
            refined_pose=refined_pose,
            init_inliers=init_pose.inliers,
            refinement_success=refined_pose.success,
            final_loss=final_loss,
            timings={
                "feature_and_retrieval": retrieval_time,
                "pnp": pnp_time,
                "refinement": refinement_time,
                "total": time.perf_counter() - start,
            },
            debug_artifacts=debug_artifacts,
            metadata={
                "candidate_names": [candidate.name for candidate in candidates],
                "match_count": len(points_2d),
                "provenance": provenance,
                "renderer_backend_error": self.renderer.backend_error,
                "hloc_global_error": self.feature_extractor.hloc_global_error,
                "hloc_local_error": self.feature_extractor.hloc_local_error,
                "feature_backends": sorted(
                    {
                        reference.metadata.get("feature_backend", "unknown")
                        for reference in candidates
                    }
                ),
            },
        )

    def localize_with_init(
        self, query_image_path: str | Path, init_pose: list[list[float]]
    ) -> LocalizationResult:
        query = load_image(query_image_path)
        refined_pose, final_loss, debug_artifacts = self.pose_refiner.refine_pose(
            query.image,
            init_pose,
            debug_dir=self.config.debug_dir,
            intrinsics=query.intrinsics,
        )
        return LocalizationResult(
            query_path=Path(query_image_path),
            init_pose=PoseEstimate(
                matrix=init_pose, inliers=0, success=True, source="provided-init"
            ),
            refined_pose=refined_pose,
            init_inliers=0,
            refinement_success=refined_pose.success,
            final_loss=final_loss,
            timings={},
            debug_artifacts=debug_artifacts,
        )


def _result_to_dict(result: LocalizationResult) -> dict:
    return {
        "query_path": str(result.query_path),
        "init_pose": result.init_pose.matrix,
        "refined_pose": result.refined_pose.matrix,
        "init_inliers": result.init_inliers,
        "refinement_success": result.refinement_success,
        "final_loss": result.final_loss,
        "timings": result.timings,
        "debug_artifacts": result.debug_artifacts,
        "metadata": result.metadata,
    }


def _save_image_tensor(image: ImageTensor, output_path: Path) -> None:
    from PIL import Image

    height = len(image)
    width = len(image[0]) if height else 0
    pixels: list[tuple[int, int, int]] = []
    for row in image:
        for pixel in row:
            pixels.append(
                tuple(
                    max(0, min(255, int(round(channel * 255.0))))
                    for channel in pixel[:3]
                )
            )
    rendered = Image.new("RGB", (width, height))
    if pixels:
        rendered.putdata(pixels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gaussian Splatting based localization prototype"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    localize_parser = subparsers.add_parser("localize")
    localize_parser.add_argument("--scene", required=True)
    localize_parser.add_argument("--query", required=True)
    localize_parser.add_argument("--output", required=True)
    localize_parser.add_argument("--debug-dir")
    localize_parser.add_argument("--gs-model")
    localize_parser.add_argument("--gs-repo")

    args = parser.parse_args(argv)
    config = PipelineConfig.from_scene_dir(
        args.scene,
        getattr(args, "debug_dir", None),
        getattr(args, "gs_model", None),
        getattr(args, "gs_repo", None),
    )
    pipeline = LocalizationPipeline(config)

    result = pipeline.localize(args.query)
    query = load_image(args.query)
    query_height = len(query.image)
    query_width = len(query.image[0]) if query_height else 0
    render_output = Path(args.output).with_name(
        f"{Path(args.output).stem}_final_render.png"
    )
    final_render = pipeline.render_pose_image(
        result.refined_pose.matrix,
        target_width=query_width,
        target_height=query_height,
    )
    _save_image_tensor(final_render, render_output)
    result.debug_artifacts["final_render"] = str(render_output)
    Path(args.output).write_text(json.dumps(_result_to_dict(result), indent=2))
    return 0
