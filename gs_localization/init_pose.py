import math

try:
    import cv2
except Exception:
    cv2 = None

try:
    import numpy as np
except Exception:
    np = None

from .config import MatchingConfig, PnPConfig, RetrievalConfig
from .data import CameraIntrinsics, PoseEstimate, ReferenceImage
from .sfm import IDENTITY_4X4, SfMDatabase


def _identity_pose(
    source: str, success: bool = False, metadata: dict | None = None
) -> PoseEstimate:
    return PoseEstimate(
        matrix=[row[:] for row in IDENTITY_4X4],
        inliers=0,
        success=success,
        source=source,
        metadata=metadata or {},
    )


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    numerator = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return numerator / (norm_a * norm_b)


def _descriptor_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class InitialPoseEstimator:
    def __init__(
        self,
        database: SfMDatabase,
        retrieval_config: RetrievalConfig,
        matching_config: MatchingConfig,
        pnp_config: PnPConfig,
    ) -> None:
        self.database = database
        self.retrieval_config = retrieval_config
        self.matching_config = matching_config
        self.pnp_config = pnp_config

    def retrieve_candidates(
        self, global_desc: list[float], top_k: int | None = None
    ) -> list[ReferenceImage]:
        k = top_k or self.retrieval_config.top_k
        scored = []
        for reference in self.database.assets.references:
            score = _cosine_similarity(global_desc, reference.global_descriptor)
            scored.append((score, reference))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [reference for _, reference in scored[:k]]

    def match_features(
        self,
        query_features: list[dict],
        candidate_features: list[ReferenceImage],
    ) -> dict[str, list[tuple[int, int]]]:
        matches: dict[str, list[tuple[int, int]]] = {}
        for reference in candidate_features:
            ref_matches = self._match_reference(query_features, reference)
            if len(ref_matches) >= self.matching_config.min_matches:
                matches[reference.name] = ref_matches
        return matches

    def _match_reference(
        self,
        query_features: list[dict],
        reference: ReferenceImage,
    ) -> list[tuple[int, int]]:
        if not query_features or not reference.local_features:
            return []
        descriptor_dim = len(query_features[0].get("descriptor", []))
        if descriptor_dim <= 8:
            return [
                (
                    query_idx,
                    min(
                        range(len(reference.local_features)),
                        key=lambda ref_idx: _descriptor_distance(
                            query_feature["descriptor"],
                            reference.local_features[ref_idx]["descriptor"],
                        ),
                    ),
                )
                for query_idx, query_feature in enumerate(query_features)
            ]
        if len(query_features) >= 8 and len(reference.local_features) >= 8:
            return self._match_reference_numpy(query_features, reference)
        ref_matches: list[tuple[int, int]] = []
        used_reference_indices: set[int] = set()
        for query_idx, query_feature in enumerate(query_features):
            best_index = None
            best_distance = float("inf")
            second_distance = float("inf")
            for ref_idx, ref_feature in enumerate(reference.local_features):
                distance = _descriptor_distance(
                    query_feature["descriptor"], ref_feature["descriptor"]
                )
                if distance < best_distance:
                    second_distance = best_distance
                    best_distance = distance
                    best_index = ref_idx
                elif distance < second_distance:
                    second_distance = distance
            ratio = best_distance / max(second_distance, 1e-6)
            if (
                best_index is not None
                and best_distance <= self.matching_config.max_descriptor_distance
                and ratio <= 0.92
                and best_index not in used_reference_indices
            ):
                used_reference_indices.add(best_index)
                ref_matches.append((query_idx, best_index))
        return ref_matches

    def _match_reference_numpy(
        self,
        query_features: list[dict],
        reference: ReferenceImage,
    ) -> list[tuple[int, int]]:
        query_descriptors = np.asarray(
            [feature["descriptor"] for feature in query_features],
            dtype=np.float32,
        )
        reference_descriptors = np.asarray(
            [feature["descriptor"] for feature in reference.local_features],
            dtype=np.float32,
        )
        if query_descriptors.size == 0 or reference_descriptors.size == 0:
            return []
        if query_descriptors.shape[1] <= 8:
            matches: list[tuple[int, int]] = []
            for query_idx in range(query_descriptors.shape[0]):
                distances = np.linalg.norm(
                    reference_descriptors - query_descriptors[query_idx],
                    axis=1,
                )
                ref_idx = int(distances.argmin())
                matches.append((query_idx, ref_idx))
            return matches
        query_norms = np.linalg.norm(query_descriptors, axis=1, keepdims=True)
        ref_norms = np.linalg.norm(reference_descriptors, axis=1, keepdims=True)
        cosine_ready = (
            query_descriptors.shape[1] > 16
            and np.all(query_norms > 0.0)
            and np.all(ref_norms > 0.0)
        )
        if cosine_ready:
            query_descriptors = query_descriptors / np.maximum(query_norms, 1e-6)
            reference_descriptors = reference_descriptors / np.maximum(ref_norms, 1e-6)
            similarity = query_descriptors @ reference_descriptors.T
            query_best = similarity.argmax(axis=1)
            query_scores = similarity[np.arange(similarity.shape[0]), query_best]
            query_second = (
                np.partition(similarity, -2, axis=1)[:, -2]
                if similarity.shape[1] > 1
                else np.full(similarity.shape[0], -1.0, dtype=np.float32)
            )
            ref_best = similarity.argmax(axis=0)
            matches: list[tuple[int, int]] = []
            for query_idx, ref_idx in enumerate(query_best.tolist()):
                if ref_best[ref_idx] != query_idx:
                    continue
                score = float(query_scores[query_idx])
                ratio = score / max(float(query_second[query_idx]), 1e-6)
                if score < 0.70 or ratio < 1.02:
                    continue
                matches.append((query_idx, ref_idx))
            return matches

        distances = np.linalg.norm(
            query_descriptors[:, None, :] - reference_descriptors[None, :, :],
            axis=2,
        )
        query_best = distances.argmin(axis=1)
        query_scores = distances[np.arange(distances.shape[0]), query_best]
        query_second = (
            np.partition(distances, 1, axis=1)[:, 1]
            if distances.shape[1] > 1
            else np.full(distances.shape[0], float("inf"), dtype=np.float32)
        )
        ref_best = distances.argmin(axis=0)
        matches = []
        for query_idx, ref_idx in enumerate(query_best.tolist()):
            if ref_best[ref_idx] != query_idx:
                continue
            distance = float(query_scores[query_idx])
            ratio = distance / max(float(query_second[query_idx]), 1e-6)
            if distance > self.matching_config.max_descriptor_distance or ratio > 0.92:
                continue
            matches.append((query_idx, ref_idx))
        return matches

    def estimate_pose_pnp(
        self,
        points_2d: list[list[float]],
        points_3d: list[list[float]],
        intrinsics: CameraIntrinsics | None,
    ) -> PoseEstimate:
        if (
            len(points_2d) < self.pnp_config.min_inliers
            or len(points_3d) < self.pnp_config.min_inliers
        ):
            return _identity_pose(
                "pnp-ransac", success=False, metadata={"reason": "not_enough_matches"}
            )

        if intrinsics is None:
            return _identity_pose(
                "pnp-ransac",
                success=False,
                metadata={"reason": "missing_intrinsics"},
            )

        object_points = np.asarray(points_3d, dtype=np.float32).reshape(-1, 1, 3)
        image_points = np.asarray(points_2d, dtype=np.float32).reshape(-1, 1, 2)
        camera_matrix = np.asarray(
            [
                [intrinsics.fx, 0.0, intrinsics.cx],
                [0.0, intrinsics.fy, intrinsics.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=object_points,
                imagePoints=image_points,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs,
                iterationsCount=self.pnp_config.max_iterations,
                reprojectionError=self.pnp_config.reprojection_error,
                confidence=self.pnp_config.confidence,
            )
        except Exception as exc:
            return _identity_pose(
                "pnp-ransac", success=False, metadata={"reason": str(exc)}
            )

        if not success:
            return _identity_pose(
                "pnp-ransac", success=False, metadata={"reason": "opencv_failed"}
            )

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        pose = [row[:] for row in IDENTITY_4X4]
        for row in range(3):
            for col in range(3):
                pose[row][col] = float(rotation_matrix[row][col])
            pose[row][3] = float(tvec[row][0])
        return PoseEstimate(
            matrix=pose, inliers=len(inliers), success=True, source="pnp-ransac"
        )
