import json
from pathlib import Path

try:
    import pycolmap
except Exception:
    pycolmap = None

from .data import CameraIntrinsics, ReferenceImage, SceneAssets

IDENTITY_4X4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


class SfMDatabase:
    def __init__(self, assets: SceneAssets) -> None:
        self.assets = assets
        self.reference_lookup = {
            reference.name: reference for reference in assets.references
        }

    def assemble_correspondences(
        self,
        query_features: list[dict],
        matches_by_reference: dict[str, list[tuple[int, int]]],
    ) -> tuple[list[list[float]], list[list[float]], list[dict]]:
        points_2d: list[list[float]] = []
        points_3d: list[list[float]] = []
        provenance: list[dict] = []
        for ref_name, matches in matches_by_reference.items():
            reference = self.reference_lookup.get(ref_name)
            if reference is None:
                continue
            local_to_world = reference.metadata.get("local_to_world_index", [])
            for query_index, reference_index in matches:
                if query_index >= len(query_features):
                    continue
                if local_to_world:
                    if reference_index >= len(local_to_world):
                        continue
                    mapped_index = local_to_world[reference_index]
                    if mapped_index is None:
                        continue
                    reference_world_index = mapped_index
                else:
                    reference_world_index = reference_index
                if reference_world_index >= len(reference.world_points):
                    continue
                points_2d.append(list(query_features[query_index]["point"]))
                points_3d.append(list(reference.world_points[reference_world_index]))
                provenance.append(
                    {
                        "reference": ref_name,
                        "query_index": query_index,
                        "reference_index": reference_index,
                        "world_index": reference_world_index,
                    }
                )
        return points_2d, points_3d, provenance


def _load_scene_json(scene_dir: Path) -> SceneAssets:
    payload = json.loads((scene_dir / "scene.json").read_text())
    intr_payload = payload.get("intrinsics")
    intrinsics = CameraIntrinsics(**intr_payload) if intr_payload else None
    references: list[ReferenceImage] = []
    for item in payload.get("references", []):
        references.append(
            ReferenceImage(
                name=item["name"],
                path=scene_dir / item["path"] if item.get("path") else None,
                global_descriptor=[float(v) for v in item.get("global_descriptor", [])],
                local_features=item.get("local_features", []),
                world_points=item.get("world_points", []),
                pose=item.get("pose"),
            )
        )
    return SceneAssets(
        scene_dir=scene_dir,
        references=references,
        renderer_config=payload.get("renderer", {}),
        intrinsics=intrinsics,
        query_poses=payload.get("query_poses", {}),
    )


def _pose_from_camera(camera: dict) -> list[list[float]]:
    pose = [row[:] for row in IDENTITY_4X4]
    rotation = camera.get(
        "rotation", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    position = camera.get("position", [0.0, 0.0, 0.0])
    for row in range(3):
        for col in range(3):
            pose[row][col] = float(rotation[row][col])
        pose[row][3] = float(position[row])
    return pose


def _load_3dgs_eval_scene(scene_dir: Path) -> SceneAssets:
    cameras = json.loads((scene_dir / "cameras.json").read_text())
    gt_dir = scene_dir / "test" / "ours_30000" / "gt"
    render_dir = scene_dir / "test" / "ours_30000" / "renders"
    gt_images = sorted(gt_dir.glob("*.png"))
    render_images = sorted(render_dir.glob("*.png"))
    references: list[ReferenceImage] = []

    image_paths = gt_images if gt_images else render_images
    if image_paths:
        if len(image_paths) == 1:
            camera_indices = [0]
        else:
            camera_indices = [
                round(idx * (len(cameras) - 1) / max(len(image_paths) - 1, 1))
                for idx in range(len(image_paths))
            ]
        for idx, image_path in enumerate(image_paths):
            camera = cameras[camera_indices[idx]]
            references.append(
                ReferenceImage(
                    name=image_path.name,
                    path=image_path,
                    global_descriptor=[],
                    local_features=[],
                    world_points=[],
                    pose=_pose_from_camera(camera),
                )
            )

    intrinsics = None
    if cameras:
        first = cameras[0]
        intrinsics = CameraIntrinsics(
            fx=float(first["fx"]),
            fy=float(first["fy"]),
            cx=float(first["width"]) / 2.0,
            cy=float(first["height"]) / 2.0,
            width=int(first["width"]),
            height=int(first["height"]),
        )
    return SceneAssets(
        scene_dir=scene_dir,
        references=references,
        renderer_config={
            "width": intrinsics.width if intrinsics else 16,
            "height": intrinsics.height if intrinsics else 16,
        },
        intrinsics=intrinsics,
        query_poses={},
    )


def _load_colmap_scene(scene_dir: Path) -> SceneAssets:

    reconstruction = pycolmap.Reconstruction(scene_dir / "sparse" / "0")
    references: list[ReferenceImage] = []
    intrinsics = None

    for image_id in reconstruction.images:
        image = reconstruction.images[image_id]
        camera = reconstruction.cameras[image.camera_id]
        if intrinsics is None:
            params = camera.params
            intrinsics = CameraIntrinsics(
                fx=float(params[0]),
                fy=float(params[1] if len(params) > 1 else params[0]),
                cx=float(params[2]),
                cy=float(params[3]),
                width=int(camera.width),
                height=int(camera.height),
            )

        rigid = image.cam_from_world()
        matrix34 = rigid.matrix()
        pose = [list(row) for row in matrix34]
        pose.append([0.0, 0.0, 0.0, 1.0])

        reference_points: list[list[float]] = []
        world_points: list[list[float]] = []
        for point2d in image.points2D:
            if not point2d.has_point3D():
                continue
            point3d = reconstruction.points3D[point2d.point3D_id]
            xy = point2d.xy
            reference_points.append([float(xy[0]), float(xy[1])])
            world_points.append([float(value) for value in point3d.xyz])

        references.append(
            ReferenceImage(
                name=image.name,
                path=scene_dir / "images" / image.name,
                global_descriptor=[],
                local_features=[],
                world_points=world_points,
                pose=pose,
                metadata={
                    "reference_points": reference_points,
                    "image_id": int(image_id),
                    "camera_id": int(image.camera_id),
                },
            )
        )

    return SceneAssets(
        scene_dir=scene_dir,
        references=references,
        renderer_config={
            "type": "gaussian-splatting",
            "width": intrinsics.width if intrinsics else 16,
            "height": intrinsics.height if intrinsics else 16,
        },
        intrinsics=intrinsics,
        query_poses={},
        metadata={"scene_type": "colmap_sparse"},
    )


def load_scene_assets(scene_dir: str | Path) -> SceneAssets:
    scene_path = Path(scene_dir)
    scene_json = scene_path / "scene.json"
    if scene_json.exists():
        return _load_scene_json(scene_path)

    if (scene_path / "cameras.json").exists() and (
        scene_path / "test" / "ours_30000"
    ).exists():
        return _load_3dgs_eval_scene(scene_path)

    if (scene_path / "sparse" / "0").exists() and (scene_path / "images").exists():
        return _load_colmap_scene(scene_path)

    if pycolmap is not None and (scene_path / "sparse").exists():
        raise NotImplementedError(
            "COLMAP-backed scene loading is reserved for environments with a prepared adapter. "
            "Provide scene.json for the prototype path."
        )

    return SceneAssets(
        scene_dir=scene_path,
        references=[],
        renderer_config={},
        intrinsics=None,
        query_poses={},
    )
