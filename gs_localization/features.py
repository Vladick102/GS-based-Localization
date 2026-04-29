import json
import math
import os
import shutil
from hashlib import sha1
from pathlib import Path

try:
    import h5py
except Exception:
    h5py = None

try:
    from hloc import extract_features as hloc_extract_features
except Exception:
    hloc_extract_features = None

try:
    import numpy as np
except Exception:
    np = None

from .data import CameraIntrinsics, ImageTensor, QueryImage, ReferenceImage, Vector

_HLOC_GLOBAL_CONF = "netvlad"
_HLOC_LOCAL_CONF = "superpoint_aachen"
_HLOC_MAX_LOCAL_FEATURES = 1024
_HLOC_COLMAP_MATCH_RADIUS = 6.0
_FEATURE_CACHE_VERSION = 3


def _ensure_rgb(image: list) -> ImageTensor:
    if image and image[0] and isinstance(image[0][0], (int, float)):
        return [[[float(v), float(v), float(v)] for v in row] for row in image]
    return [
        [[float(pixel[0]), float(pixel[1]), float(pixel[2])] for pixel in row]
        for row in image
    ]


def _default_intrinsics(width: int, height: int) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=float(width),
        fy=float(height),
        cx=float(width) / 2.0,
        cy=float(height) / 2.0,
        width=width,
        height=height,
    )


def load_image(path: str | Path) -> QueryImage:
    image_path = Path(path)
    suffix = image_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(image_path.read_text())
        image = _ensure_rgb(payload["image"])
        intr = payload.get("intrinsics")
        if intr:
            intrinsics = CameraIntrinsics(**intr)
        else:
            intrinsics = _default_intrinsics(len(image[0]), len(image))
        return QueryImage(path=image_path, image=image, intrinsics=intrinsics)

    from PIL import Image

    pil = Image.open(image_path).convert("RGB")
    width, height = pil.size
    pixels = list(pil.getdata())
    image: ImageTensor = []
    for row in range(height):
        start = row * width
        image.append(
            [
                [channel / 255.0 for channel in pixels[start + col]]
                for col in range(width)
            ]
        )
    return QueryImage(
        path=image_path, image=image, intrinsics=_default_intrinsics(width, height)
    )


def flatten_image(image: ImageTensor) -> list[float]:
    values: list[float] = []
    for row in image:
        for pixel in row:
            values.extend(pixel)
    return values


def grayscale(image: ImageTensor) -> list[list[float]]:
    return [
        [0.2989 * pixel[0] + 0.5870 * pixel[1] + 0.1140 * pixel[2] for pixel in row]
        for row in image
    ]


def downsample_image(image: ImageTensor, max_side: int = 256) -> ImageTensor:
    height = len(image)
    width = len(image[0]) if height else 0
    if height == 0 or width == 0:
        return image
    scale = max(height, width) / float(max_side)
    if scale <= 1.0:
        return image
    target_height = max(1, int(height / scale))
    target_width = max(1, int(width / scale))
    resized: ImageTensor = []
    for y in range(target_height):
        source_y = min(height - 1, int(y * height / target_height))
        row = []
        for x in range(target_width):
            source_x = min(width - 1, int(x * width / target_width))
            row.append(list(image[source_y][source_x]))
        resized.append(row)
    return resized


def sample_descriptors_at_points(
    image: ImageTensor, points: list[list[float]]
) -> list[dict[str, Vector]]:
    height = len(image)
    width = len(image[0]) if height else 0
    if height == 0 or width == 0:
        return []
    gradients = gradient_map(image)
    features: list[dict[str, Vector]] = []
    for point in points:
        x = min(width - 1, max(0, int(round(point[0]))))
        y = min(height - 1, max(0, int(round(point[1]))))
        pixel = image[y][x]
        descriptor = [
            pixel[0],
            pixel[1],
            pixel[2],
            gradients[y][x],
            x / max(width - 1, 1),
            y / max(height - 1, 1),
        ]
        features.append({"point": [float(x), float(y)], "descriptor": descriptor})
    return features


def gradient_map(image: ImageTensor) -> list[list[float]]:
    gray = grayscale(image)
    height = len(gray)
    width = len(gray[0]) if height else 0
    grads: list[list[float]] = []
    for y in range(height):
        row: list[float] = []
        for x in range(width):
            left = gray[y][max(0, x - 1)]
            right = gray[y][min(width - 1, x + 1)]
            up = gray[max(0, y - 1)][x]
            down = gray[min(height - 1, y + 1)][x]
            row.append(abs(right - left) + abs(down - up))
        grads.append(row)
    return grads


def _vector_from_dataset(dataset) -> list[float]:
    values = dataset[()]
    if np is not None:
        return np.asarray(values, dtype=np.float32).reshape(-1).astype(float).tolist()
    if hasattr(values, "reshape"):
        return values.reshape(-1).tolist()
    if isinstance(values, (list, tuple)):
        flattened: list[float] = []
        for item in values:
            if isinstance(item, (list, tuple)):
                flattened.extend(float(v) for v in item)
            else:
                flattened.append(float(item))
        return flattened
    return [float(values)]


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


class FeatureExtractor:
    def __init__(self, cache_dir: Path) -> None:
        self.hloc_global_error: str | None = None
        self.hloc_local_error: str | None = None
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hloc_root = self.cache_dir / "hloc"
        self.hloc_image_root = self.hloc_root / "images"
        self.hloc_output_root = self.hloc_root / "outputs"
        self.hloc_global_path = self.hloc_output_root / "netvlad_features.h5"
        self.hloc_local_path = self.hloc_output_root / "superpoint_features.h5"
        self.cache_hits = {"global": 0, "local": 0}
        self.cache_misses = {"global": 0, "local": 0}
        self.hloc_image_root.mkdir(parents=True, exist_ok=True)
        self.hloc_output_root.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, reference: ReferenceImage, suffix: str) -> Path:
        cache_key = sha1(str(reference.path.resolve()).encode("utf-8")).hexdigest()[:16]
        safe_name = Path(reference.name).stem
        return self.cache_dir / f"{safe_name}_{cache_key}_{suffix}.json"

    def _load_cached_payload(
        self, cache_path: Path, reference: ReferenceImage
    ) -> dict | None:
        try:
            payload = json.loads(cache_path.read_text())
        except Exception:
            return None
        if not reference.path.exists():
            return None
        if payload.get("source_mtime_ns") != reference.path.stat().st_mtime_ns:
            return None
        if payload.get("reference_points_count") != len(
            reference.metadata.get("reference_points", [])
        ):
            return None
        if payload.get("cache_version") != _FEATURE_CACHE_VERSION:
            return None
        return payload

    def _write_cached_payload(
        self, cache_path: Path, reference: ReferenceImage, payload: dict
    ) -> None:
        if not reference.path.exists():
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = {
            "cache_version": _FEATURE_CACHE_VERSION,
            "source_path": str(reference.path),
            "source_mtime_ns": reference.path.stat().st_mtime_ns,
            "reference_points_count": len(
                reference.metadata.get("reference_points", [])
            ),
            **payload,
        }
        cache_path.write_text(json.dumps(serialized))

    def _stage_hloc_image(self, image_path: Path, namespace: str) -> str:
        suffix = image_path.suffix or ".png"
        safe_name = f"{image_path.stem}_{sha1(str(image_path.resolve()).encode('utf-8')).hexdigest()[:12]}{suffix}"
        relative_name = f"{namespace}/{safe_name}"
        target_path = self.hloc_image_root / relative_name
        _copy_or_link(image_path.resolve(), target_path)
        return relative_name

    def _run_hloc_extractor(
        self,
        conf_name: str,
        feature_path: Path,
        image_list: list[str],
    ) -> None:
        if not image_list:
            return
        conf = hloc_extract_features.confs[conf_name]
        attempts = [
            {
                "conf": conf,
                "image_dir": self.hloc_image_root,
                "export_dir": self.hloc_output_root,
                "feature_path": feature_path,
                "image_list": image_list,
                "overwrite": False,
            },
            {
                "conf": conf,
                "image_dir": self.hloc_image_root,
                "export_dir": self.hloc_output_root,
                "feature_path": feature_path,
                "image_list": image_list,
            },
            {
                "conf": conf,
                "image_dir": self.hloc_image_root,
                "export_dir": self.hloc_output_root,
                "image_list": image_list,
            },
        ]
        last_exc: Exception | None = None
        for kwargs in attempts:
            try:
                hloc_extract_features.main(**kwargs)
                return
            except TypeError as exc:
                last_exc = exc
                continue
            except Exception as exc:
                last_exc = exc
                break
        if last_exc is not None:
            raise last_exc

    def _record_hloc_failure(self, kind: str, exc: Exception) -> None:
        message = f"{type(exc).__name__}: {exc}"
        if kind == "global":
            self.hloc_global_error = message
        else:
            self.hloc_local_error = message
        raise RuntimeError(f"HLOC {kind} feature extraction failed: {message}") from exc

    def _load_hloc_global_descriptor(self, image_name: str) -> list[float] | None:
        if not self.hloc_global_path.exists():
            return None
        with h5py.File(self.hloc_global_path, "r") as handle:
            if image_name not in handle:
                return None
            group = handle[image_name]
            if "global_descriptor" not in group:
                return None
            return _vector_from_dataset(group["global_descriptor"])

    def _load_hloc_local_features(
        self,
        image_name: str,
        max_features: int | None = _HLOC_MAX_LOCAL_FEATURES,
    ) -> list[dict[str, Vector]] | None:
        if not self.hloc_local_path.exists():
            return None
        with h5py.File(self.hloc_local_path, "r") as handle:
            if image_name not in handle:
                return None
            group = handle[image_name]
            if "keypoints" not in group or "descriptors" not in group:
                return None
            keypoints = group["keypoints"][()]
            descriptors = group["descriptors"][()]
            scores = group["scores"][()] if "scores" in group else None
            if descriptors.ndim == 2 and descriptors.shape[1] == len(keypoints):
                descriptors = descriptors.transpose(1, 0)
            elif descriptors.ndim == 2 and descriptors.shape[0] == len(keypoints):
                pass
            else:
                descriptors = descriptors.reshape(len(keypoints), -1)
            if scores is not None:
                order = list(range(len(keypoints)))
                order.sort(key=lambda idx: float(scores[idx]), reverse=True)
            else:
                order = list(range(len(keypoints)))
            if max_features is not None:
                order = order[: min(max_features, len(order))]
            features: list[dict[str, Vector]] = []
            for idx in order:
                point = keypoints[idx]
                descriptor = descriptors[idx]
                feature = {
                    "point": [float(point[0]), float(point[1])],
                    "descriptor": [float(value) for value in descriptor],
                    "backend": "hloc-superpoint",
                }
                if scores is not None:
                    feature["score"] = float(scores[idx])
                features.append(feature)
            return features

    def _map_local_features_to_colmap(
        self,
        local_features: list[dict[str, Vector]],
        reference_points: list[list[float]],
    ) -> list[int | None]:
        if not local_features or not reference_points:
            return []
        if np is not None:
            query_points = np.asarray(
                [feature["point"] for feature in local_features], dtype=np.float32
            )
            colmap_points = np.asarray(reference_points, dtype=np.float32)
            distances = (
                (query_points[:, None, :] - colmap_points[None, :, :]) ** 2
            ).sum(axis=2)
            best_indices = distances.argmin(axis=1)
            best_distances = distances[np.arange(len(query_points)), best_indices]
            return [
                (
                    int(best_idx)
                    if float(best_distance) <= _HLOC_COLMAP_MATCH_RADIUS**2
                    else None
                )
                for best_idx, best_distance in zip(
                    best_indices.tolist(), best_distances.tolist()
                )
            ]
        mapping: list[int | None] = []
        for feature in local_features:
            best_idx = None
            best_distance = float("inf")
            fx, fy = feature["point"]
            for idx, point in enumerate(reference_points):
                dx = fx - point[0]
                dy = fy - point[1]
                distance = dx * dx + dy * dy
                if distance < best_distance:
                    best_distance = distance
                    best_idx = idx
            if best_idx is not None and best_distance <= _HLOC_COLMAP_MATCH_RADIUS**2:
                mapping.append(best_idx)
            else:
                mapping.append(None)
        return mapping

    def _align_local_features_to_reference_points(
        self,
        local_features: list[dict[str, Vector]],
        reference_points: list[list[float]],
    ) -> tuple[list[dict[str, Vector]], list[int]]:
        mapping = self._map_local_features_to_colmap(local_features, reference_points)
        if not mapping:
            return [], []

        best_by_reference: dict[int, tuple[float, int, dict[str, Vector]]] = {}
        for feature_idx, mapped_index in enumerate(mapping):
            if mapped_index is None:
                continue
            feature = local_features[feature_idx]
            fx, fy = feature["point"]
            rx, ry = reference_points[mapped_index]
            distance_sq = (fx - rx) * (fx - rx) + (fy - ry) * (fy - ry)
            current = best_by_reference.get(mapped_index)
            if current is None or distance_sq < current[0]:
                best_by_reference[mapped_index] = (distance_sq, feature_idx, feature)

        if not best_by_reference:
            return [], []

        aligned_features: list[dict[str, Vector]] = []
        local_to_world_index: list[int] = []
        for mapped_index in sorted(best_by_reference):
            _, _, feature = best_by_reference[mapped_index]
            aligned_feature = {
                "point": [
                    float(reference_points[mapped_index][0]),
                    float(reference_points[mapped_index][1]),
                ],
                "descriptor": [float(value) for value in feature["descriptor"]],
                "backend": feature.get("backend", "hloc-superpoint"),
            }
            if "score" in feature:
                aligned_feature["score"] = float(feature["score"])
            aligned_feature["source_point"] = [
                float(feature["point"][0]),
                float(feature["point"][1]),
            ]
            aligned_features.append(aligned_feature)
            local_to_world_index.append(int(mapped_index))
        return aligned_features, local_to_world_index

    def _extract_hloc_global_for_references(
        self, references: list[ReferenceImage]
    ) -> bool:
        staged: list[tuple[ReferenceImage, str]] = []
        for reference in references:
            if not reference.path.exists() or reference.global_descriptor:
                continue
            image_name = self._stage_hloc_image(reference.path, "references")
            reference.metadata["hloc_image_name"] = image_name
            staged.append((reference, image_name))
        if not staged:
            return False
        try:
            self._run_hloc_extractor(
                _HLOC_GLOBAL_CONF,
                self.hloc_global_path,
                [image_name for _, image_name in staged],
            )
        except Exception as exc:
            self._record_hloc_failure("global", exc)
        for reference, image_name in staged:
            descriptor = self._load_hloc_global_descriptor(image_name)
            if descriptor:
                reference.global_descriptor = descriptor
                reference.metadata["feature_backend"] = "hloc-netvlad"
        return any(reference.global_descriptor for reference, _ in staged)

    def _extract_hloc_local_for_references(
        self, references: list[ReferenceImage]
    ) -> bool:
        staged: list[tuple[ReferenceImage, str]] = []
        for reference in references:
            if not reference.path.exists() or reference.local_features:
                continue
            image_name = reference.metadata.get(
                "hloc_image_name"
            ) or self._stage_hloc_image(reference.path, "references")
            reference.metadata["hloc_image_name"] = image_name
            staged.append((reference, image_name))
        if not staged:
            return False
        try:
            self._run_hloc_extractor(
                _HLOC_LOCAL_CONF,
                self.hloc_local_path,
                [image_name for _, image_name in staged],
            )
        except Exception as exc:
            self._record_hloc_failure("local", exc)
        any_loaded = False
        for reference, image_name in staged:
            local_features = self._load_hloc_local_features(
                image_name, max_features=None
            )
            if not local_features:
                continue
            reference_points = reference.metadata.get("reference_points", [])
            if reference_points:
                aligned_features, mapping = (
                    self._align_local_features_to_reference_points(
                        local_features,
                        reference_points,
                    )
                )
                if not aligned_features:
                    continue
                reference.local_features = aligned_features
                reference.metadata["local_to_world_index"] = mapping
            else:
                reference.local_features = local_features
            reference.metadata["feature_backend"] = "hloc-superpoint"
            any_loaded = True
        return any_loaded

    def extract_global_descriptor(self, image: ImageTensor) -> Vector:
        image = downsample_image(image, max_side=192)
        flat = flatten_image(image)
        count = max(len(flat), 1)
        mean_value = sum(flat) / count
        variance = sum((value - mean_value) ** 2 for value in flat) / count
        std_value = math.sqrt(variance)
        height = len(image)
        width = len(image[0]) if height else 0
        center_pixel = (
            image[height // 2][width // 2] if height and width else [0.0, 0.0, 0.0]
        )
        corners = [
            image[0][0] if height and width else [0.0, 0.0, 0.0],
            image[0][width - 1] if height and width else [0.0, 0.0, 0.0],
            image[height - 1][0] if height and width else [0.0, 0.0, 0.0],
            image[height - 1][width - 1] if height and width else [0.0, 0.0, 0.0],
        ]
        corner_mean = sum(sum(pixel) / 3.0 for pixel in corners) / 4.0
        gradients = gradient_map(image)
        grad_values = [value for row in gradients for value in row]
        grad_mean = sum(grad_values) / max(len(grad_values), 1)
        return [
            mean_value,
            std_value,
            center_pixel[0],
            center_pixel[1],
            center_pixel[2],
            corner_mean,
            grad_mean,
        ]

    def extract_local_features(self, image: ImageTensor) -> list[dict[str, Vector]]:
        image = downsample_image(image, max_side=160)
        gradients = gradient_map(image)
        height = len(image)
        width = len(image[0]) if height else 0
        candidates: list[tuple[float, int, int]] = []
        for y in range(height):
            for x in range(width):
                candidates.append((gradients[y][x], x, y))
        candidates.sort(reverse=True)
        selected = candidates[
            : max(8, min(64, len(candidates) // 3 or len(candidates)))
        ]

        features: list[dict[str, Vector]] = []
        for _, x, y in selected:
            pixel = image[y][x]
            descriptor = [
                pixel[0],
                pixel[1],
                pixel[2],
                gradients[y][x],
                x / max(width - 1, 1),
                y / max(height - 1, 1),
            ]
            features.append({"point": [float(x), float(y)], "descriptor": descriptor})
        return features

    def extract_query_global(self, query: QueryImage) -> Vector:
        if query.path.suffix.lower() == ".json" or not query.path.exists():
            raise RuntimeError("Query image must be an existing raster image file.")
        image_name = self._stage_hloc_image(query.path, "queries")
        try:
            self._run_hloc_extractor(
                _HLOC_GLOBAL_CONF,
                self.hloc_global_path,
                [image_name],
            )
            descriptor = self._load_hloc_global_descriptor(image_name)
            if descriptor:
                return descriptor
        except Exception as exc:
            self._record_hloc_failure("global", exc)
        raise RuntimeError("Failed to load HLOC global descriptor for query image.")

    def extract_query_local(self, query: QueryImage) -> list[dict[str, Vector]]:
        if query.path.suffix.lower() == ".json" or not query.path.exists():
            raise RuntimeError("Query image must be an existing raster image file.")
        image_name = self._stage_hloc_image(query.path, "queries")
        try:
            self._run_hloc_extractor(
                _HLOC_LOCAL_CONF,
                self.hloc_local_path,
                [image_name],
            )
            local_features = self._load_hloc_local_features(image_name)
            if local_features:
                return local_features
        except Exception as exc:
            self._record_hloc_failure("local", exc)
        raise RuntimeError("Failed to load HLOC local features for query image.")

    def ensure_reference_globals(self, references: list[ReferenceImage]) -> None:
        missing_cache = []
        for reference in references:
            if reference.global_descriptor:
                continue
            if not reference.path.exists():
                raise RuntimeError(f"Reference image is missing: {reference.name}")
            cache_path = self._cache_path(reference, "global")
            if cache_path.exists():
                cached = self._load_cached_payload(cache_path, reference)
                if cached is not None:
                    reference.global_descriptor = [
                        float(value) for value in cached["global_descriptor"]
                    ]
                    if "feature_backend" in cached:
                        reference.metadata["feature_backend"] = cached["feature_backend"]
                    self.cache_hits["global"] += 1
                    continue
            self.cache_misses["global"] += 1
            missing_cache.append(reference)

        if missing_cache:
            self._extract_hloc_global_for_references(missing_cache)
            for reference in missing_cache:
                if reference.global_descriptor:
                    self._write_cached_payload(
                        self._cache_path(reference, "global"),
                        reference,
                        {
                            "global_descriptor": reference.global_descriptor,
                            "feature_backend": reference.metadata.get(
                                "feature_backend", "hloc-netvlad"
                            ),
                        },
                    )

    def ensure_reference_locals(self, references: list[ReferenceImage]) -> None:
        missing_cache = []
        for reference in references:
            if reference.local_features:
                continue
            if not reference.path.exists():
                raise RuntimeError(f"Reference image is missing: {reference.name}")
            cache_path = self._cache_path(reference, "local")
            if cache_path.exists():
                cached = self._load_cached_payload(cache_path, reference)
                if cached is not None:
                    reference.local_features = cached["local_features"]
                    if "local_to_world_index" in cached:
                        reference.metadata["local_to_world_index"] = cached[
                            "local_to_world_index"
                        ]
                    if "feature_backend" in cached:
                        reference.metadata["feature_backend"] = cached["feature_backend"]
                    self.cache_hits["local"] += 1
                    continue
            self.cache_misses["local"] += 1
            missing_cache.append(reference)

        if missing_cache:
            self._extract_hloc_local_for_references(missing_cache)
            for reference in missing_cache:
                if reference.local_features:
                    self._write_cached_payload(
                        self._cache_path(reference, "local"),
                        reference,
                        {
                            "local_features": reference.local_features,
                            "local_to_world_index": reference.metadata.get(
                                "local_to_world_index", []
                            ),
                            "feature_backend": reference.metadata.get(
                                "feature_backend", "hloc-superpoint"
                            ),
                        },
                    )


