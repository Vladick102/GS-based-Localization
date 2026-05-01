import json
import math
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

from .config import RefinementConfig
from .data import ImageTensor, PoseEstimate
from .renderer import GaussianSplattingRenderer
from .sfm import IDENTITY_4X4


def _zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def _identity(size: int) -> list[list[float]]:
    matrix = _zeros(size, size)
    for idx in range(size):
        matrix[idx][idx] = 1.0
    return matrix


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows = len(a)
    cols = len(b[0])
    inner = len(b)
    result = _zeros(rows, cols)
    for row in range(rows):
        for col in range(cols):
            result[row][col] = sum(a[row][k] * b[k][col] for k in range(inner))
    return result


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*matrix)]


def _vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _skew(vector: list[float]) -> list[list[float]]:
    wx, wy, wz = vector
    return [
        [0.0, -wz, wy],
        [wz, 0.0, -wx],
        [-wy, wx, 0.0],
    ]


def _matrix_add(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [left + right for left, right in zip(row_a, row_b)]
        for row_a, row_b in zip(a, b)
    ]


def _matrix_scalar(matrix: list[list[float]], scalar: float) -> list[list[float]]:
    return [[value * scalar for value in row] for row in matrix]


def _gaussian_kernel_1d(sigma: float) -> list[float]:
    if sigma <= 1e-6:
        return [1.0]
    radius = max(1, int(math.ceil(2 * sigma)))
    kernel = []
    for idx in range(-radius, radius + 1):
        kernel.append(math.exp(-(idx * idx) / (2.0 * sigma * sigma)))
    total = sum(kernel)
    return [value / total for value in kernel]


def _apply_kernel_gray(
    image: list[list[float]], kernel: list[float]
) -> list[list[float]]:
    radius = len(kernel) // 2
    height = len(image)
    width = len(image[0]) if height else 0
    tmp = _zeros(height, width)
    for y in range(height):
        for x in range(width):
            value = 0.0
            for offset, weight in enumerate(kernel):
                dx = offset - radius
                value += image[y][min(width - 1, max(0, x + dx))] * weight
            tmp[y][x] = value
    out = _zeros(height, width)
    for y in range(height):
        for x in range(width):
            value = 0.0
            for offset, weight in enumerate(kernel):
                dy = offset - radius
                value += tmp[min(height - 1, max(0, y + dy))][x] * weight
            out[y][x] = value
    return out


def _image_channels(image: ImageTensor, channel: int) -> list[list[float]]:
    return [[pixel[channel] for pixel in row] for row in image]


def _rebuild_rgb(channels: list[list[list[float]]]) -> ImageTensor:
    height = len(channels[0])
    width = len(channels[0][0]) if height else 0
    image: ImageTensor = []
    for y in range(height):
        row = []
        for x in range(width):
            row.append([channels[0][y][x], channels[1][y][x], channels[2][y][x]])
        image.append(row)
    return image


def _resize_gray(
    image: list[list[float]], target_height: int, target_width: int
) -> list[list[float]]:
    if not image or not image[0]:
        return _zeros(target_height, target_width)
    source_height = len(image)
    source_width = len(image[0])
    resized = _zeros(target_height, target_width)
    for y in range(target_height):
        source_y = min(
            source_height - 1, int(y * source_height / max(target_height, 1))
        )
        for x in range(target_width):
            source_x = min(
                source_width - 1, int(x * source_width / max(target_width, 1))
            )
            resized[y][x] = image[source_y][source_x]
    return resized


def _resize_rgb(
    image: ImageTensor, target_height: int, target_width: int
) -> ImageTensor:
    if not image or not image[0]:
        return [
            [[0.0, 0.0, 0.0] for _ in range(target_width)] for _ in range(target_height)
        ]
    source_height = len(image)
    source_width = len(image[0])
    resized: ImageTensor = []
    for y in range(target_height):
        source_y = min(
            source_height - 1, int(y * source_height / max(target_height, 1))
        )
        row = []
        for x in range(target_width):
            source_x = min(
                source_width - 1, int(x * source_width / max(target_width, 1))
            )
            row.append(list(image[source_y][source_x]))
        resized.append(row)
    return resized


def _masked_sample_coordinates(
    mask: list[list[float]], max_samples: int
) -> list[tuple[int, int, float]]:
    coords: list[tuple[int, int, float]] = []
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if value > 0.0:
                coords.append((y, x, value))
    if len(coords) <= max_samples:
        return coords
    stride = max(1, len(coords) // max_samples)
    sampled = coords[::stride]
    return sampled[:max_samples]


def _compose_pose(
    update: list[list[float]], base: list[list[float]]
) -> list[list[float]]:
    return _matmul(update, base)


def _rotation_error_degrees(rotation: list[list[float]]) -> float:
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    value = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.degrees(math.acos(value))


class PoseRefiner:
    def __init__(
        self, renderer: GaussianSplattingRenderer, config: RefinementConfig
    ) -> None:
        self.renderer = renderer
        self.config = config

    def exp_map_se3(self, xi: list[float]) -> list[list[float]]:
        omega = xi[:3]
        upsilon = xi[3:]
        theta = _vector_norm(omega)
        rotation = _identity(3)
        V = _identity(3)
        if theta > 1e-9:
            omega_hat = _skew(omega)
            omega_hat_sq = _matmul(omega_hat, omega_hat)
            sin_term = math.sin(theta) / theta
            cos_term = (1.0 - math.cos(theta)) / (theta * theta)
            rotation = _matrix_add(
                _matrix_add(_identity(3), _matrix_scalar(omega_hat, sin_term)),
                _matrix_scalar(omega_hat_sq, cos_term),
            )
            V = _matrix_add(
                _matrix_add(
                    _identity(3),
                    _matrix_scalar(
                        omega_hat, (1.0 - math.cos(theta)) / (theta * theta)
                    ),
                ),
                _matrix_scalar(omega_hat_sq, (theta - math.sin(theta)) / (theta**3)),
            )
        else:
            omega_hat = _skew(omega)
            rotation = _matrix_add(_identity(3), omega_hat)
            V = _matrix_add(_identity(3), _matrix_scalar(omega_hat, 0.5))

        translation = [
            sum(V[row][col] * upsilon[col] for col in range(3)) for row in range(3)
        ]
        pose = [row[:] for row in IDENTITY_4X4]
        for row in range(3):
            for col in range(3):
                pose[row][col] = rotation[row][col]
            pose[row][3] = translation[row]
        return pose

    def blur_sigma(self, iteration: int, max_iters: int) -> float:
        if max_iters <= 1:
            return self.config.blur_sigma_end
        ratio = iteration / float(max_iters - 1)
        return self.config.blur_sigma_start + ratio * (
            self.config.blur_sigma_end - self.config.blur_sigma_start
        )

    def apply_gaussian_blur(
        self, image: ImageTensor, iteration: int, max_iters: int
    ) -> ImageTensor:
        sigma = self.blur_sigma(iteration, max_iters)
        kernel = _gaussian_kernel_1d(sigma)
        channels = [_image_channels(image, channel) for channel in range(3)]
        blurred = [_apply_kernel_gray(channel, kernel) for channel in channels]
        return _rebuild_rgb(blurred)

    def compute_pixel_mask(
        self, image: ImageTensor, opacity_map: list[list[float]]
    ) -> list[list[float]]:
        height = len(image)
        width = len(image[0]) if height else 0
        gradients = []
        for y in range(height):
            row = []
            for x in range(width):
                current = sum(image[y][x]) / 3.0
                right = sum(image[y][min(width - 1, x + 1)]) / 3.0
                down = sum(image[min(height - 1, y + 1)][x]) / 3.0
                row.append(abs(current - right) + abs(current - down))
            gradients.append(row)
        values = [value for row in gradients for value in row]
        if not values:
            return []
        sorted_values = sorted(values)
        threshold_index = min(
            len(sorted_values) - 1,
            int(self.config.gradient_quantile * (len(sorted_values) - 1)),
        )
        gradient_threshold = sorted_values[threshold_index]
        mask = _zeros(height, width)
        for y in range(height):
            for x in range(width):
                if (
                    opacity_map[y][x] >= self.config.opacity_threshold
                    and gradients[y][x] >= gradient_threshold
                ):
                    mask[y][x] = 1.0
        if all(value == 0.0 for row in mask for value in row):
            for y in range(height):
                for x in range(width):
                    mask[y][x] = (
                        1.0
                        if opacity_map[y][x] >= self.config.opacity_threshold
                        else 0.0
                    )
        return mask

    def photometric_loss(
        self, query: ImageTensor, rendered: ImageTensor, mask: list[list[float]]
    ) -> float:
        total = 0.0
        count = 0.0
        coords = _masked_sample_coordinates(mask, self.config.max_loss_pixels)
        for y, x, weight in coords:
            total += (
                sum(
                    abs(query[y][x][channel] - rendered[y][x][channel])
                    for channel in range(3)
                )
                * weight
            )
            count += 3.0 * weight
        return total / count if count else float("inf")

    def _tensor_from_image(self, image: ImageTensor):

        return torch.tensor(image, dtype=torch.float32, device="cuda").permute(2, 0, 1)

    def _image_from_tensor(self, image_tensor: "torch.Tensor") -> ImageTensor:
        tensor = image_tensor.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu()
        return [
            [[float(channel) for channel in pixel] for pixel in row]
            for row in tensor.tolist()
        ]

    def _gray_from_tensor(self, image_tensor: "torch.Tensor") -> list[list[float]]:
        tensor = image_tensor.detach().cpu()
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            return [[float(value) for value in row] for row in tensor[0].tolist()]
        if tensor.dim() == 2:
            return [[float(value) for value in row] for row in tensor.tolist()]
        raise ValueError("Expected a single-channel tensor.")

    def exp_map_se3_torch(self, xi: "torch.Tensor") -> "torch.Tensor":
        omega = xi[:3]
        upsilon = xi[3:]
        theta = torch.linalg.norm(omega)
        rotation = torch.eye(3, dtype=xi.dtype, device=xi.device)
        V = torch.eye(3, dtype=xi.dtype, device=xi.device)
        zero = torch.zeros((), dtype=xi.dtype, device=xi.device)
        omega_hat = torch.stack(
            [
                torch.stack([zero, -omega[2], omega[1]]),
                torch.stack([omega[2], zero, -omega[0]]),
                torch.stack([-omega[1], omega[0], zero]),
            ]
        )
        if float(theta.detach().item()) > 1e-9:
            omega_hat_sq = omega_hat @ omega_hat
            sin_term = torch.sin(theta) / theta
            cos_term = (1.0 - torch.cos(theta)) / (theta * theta)
            rotation = rotation + sin_term * omega_hat + cos_term * omega_hat_sq
            V = (
                V
                + ((1.0 - torch.cos(theta)) / (theta * theta)) * omega_hat
                + ((theta - torch.sin(theta)) / (theta**3)) * omega_hat_sq
            )
        else:
            rotation = rotation + omega_hat
            V = V + 0.5 * omega_hat
        translation = V @ upsilon
        pose = torch.eye(4, dtype=xi.dtype, device=xi.device)
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        return pose

    def _apply_gaussian_blur_tensor(
        self,
        image_tensor: "torch.Tensor",
        iteration: int,
        max_iters: int,
    ) -> "torch.Tensor":

        sigma = self.blur_sigma(iteration, max_iters)
        kernel_1d = _gaussian_kernel_1d(sigma)
        kernel = torch.tensor(
            kernel_1d, dtype=image_tensor.dtype, device=image_tensor.device
        )
        kernel_2d = torch.outer(kernel, kernel)
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel_2d = kernel_2d.view(1, 1, kernel_2d.shape[0], kernel_2d.shape[1])
        channels = image_tensor.shape[0]
        weight = kernel_2d.repeat(channels, 1, 1, 1)
        padded = F.pad(
            image_tensor.unsqueeze(0),
            (
                kernel.shape[0] // 2,
                kernel.shape[0] // 2,
                kernel.shape[0] // 2,
                kernel.shape[0] // 2,
            ),
            mode="replicate",
        )
        return F.conv2d(padded, weight, groups=channels).squeeze(0)

    def _compute_pixel_mask_tensor(
        self,
        image_tensor: "torch.Tensor",
        opacity_tensor: "torch.Tensor",
    ) -> "torch.Tensor":
        gray = (
            0.2989 * image_tensor[0]
            + 0.5870 * image_tensor[1]
            + 0.1140 * image_tensor[2]
        )
        current = gray
        right = torch.roll(gray, shifts=-1, dims=1)
        down = torch.roll(gray, shifts=-1, dims=0)
        gradients = (current - right).abs() + (current - down).abs()
        flat = gradients.reshape(-1)
        if flat.numel() == 0:
            return torch.zeros_like(gray)
        threshold_index = min(
            flat.numel() - 1,
            int(self.config.gradient_quantile * max(flat.numel() - 1, 0)),
        )
        gradient_threshold = torch.sort(flat).values[threshold_index]
        mask = (
            (opacity_tensor[0] >= self.config.opacity_threshold)
            & (gradients >= gradient_threshold)
        ).float()
        if float(mask.sum().item()) == 0.0:
            mask = (opacity_tensor[0] >= self.config.opacity_threshold).float()
        return mask

    def _photometric_loss_tensor(
        self,
        query_tensor: "torch.Tensor",
        rendered_tensor: "torch.Tensor",
        mask_tensor: "torch.Tensor",
    ) -> "torch.Tensor":
        diff = (query_tensor - rendered_tensor).abs().sum(dim=0)
        valid_coords = torch.nonzero(mask_tensor > 0.0, as_tuple=False)
        if valid_coords.numel() == 0:
            return torch.tensor(float("inf"), device=query_tensor.device)
        if valid_coords.shape[0] > self.config.max_loss_pixels:
            stride = max(1, valid_coords.shape[0] // self.config.max_loss_pixels)
            valid_coords = valid_coords[::stride][: self.config.max_loss_pixels]
        weights = mask_tensor[valid_coords[:, 0], valid_coords[:, 1]]
        values = diff[valid_coords[:, 0], valid_coords[:, 1]]
        return (values * weights).sum() / torch.clamp(weights.sum() * 3.0, min=1e-6)

    def _loss_for_update(
        self,
        query_image: ImageTensor,
        base_pose: list[list[float]],
        xi: list[float],
        iteration: int,
        max_iters: int,
        blurred_query: ImageTensor | None = None,
        intrinsics=None,
    ) -> tuple[float, dict]:
        update = self.exp_map_se3(xi)
        pose = _compose_pose(update, base_pose)
        target_height = len(query_image)
        target_width = len(query_image[0]) if target_height else 0
        rendered, opacity = self.renderer.render(
            pose,
            target_width=target_width,
            target_height=target_height,
            intrinsics=intrinsics,
        )
        target_height = len(query_image)
        target_width = len(query_image[0]) if target_height else 0
        if target_height and target_width:
            rendered = _resize_rgb(rendered, target_height, target_width)
            opacity = _resize_gray(opacity, target_height, target_width)
        if blurred_query is None:
            blurred_query = self.apply_gaussian_blur(query_image, iteration, max_iters)
        blurred_render = self.apply_gaussian_blur(rendered, iteration, max_iters)
        mask = self.compute_pixel_mask(blurred_query, opacity)
        loss = self.photometric_loss(blurred_query, blurred_render, mask)
        return loss, {
            "rendered": rendered,
            "opacity": opacity,
            "mask": mask,
            "pose": pose,
        }

    def refine_pose(
        self,
        query_image: ImageTensor,
        initial_pose: list[list[float]],
        num_iters: int | None = None,
        debug_dir: Path | None = None,
        intrinsics=None,
    ) -> tuple[PoseEstimate, float | None, dict[str, str]]:

        if self.renderer.gs_backend is None or not hasattr(
            self.renderer, "render_torch"
        ):
            raise RuntimeError(
                "The differentiable Gaussian Splatting backend is required for refinement."
            )
        return self._refine_pose_autograd(
            query_image,
            initial_pose,
            num_iters=num_iters,
            debug_dir=debug_dir,
            intrinsics=intrinsics,
        )

    def _refine_pose_fallback(
        self,
        query_image: ImageTensor,
        initial_pose: list[list[float]],
        num_iters: int | None = None,
        debug_dir: Path | None = None,
    ) -> tuple[PoseEstimate, float | None, dict[str, str]]:
        iterations = num_iters or self.config.num_iters
        xi = [0.0] * 6
        first_blurred_query = self.apply_gaussian_blur(query_image, 0, iterations)
        first_loss, first_debug = self._loss_for_update(
            query_image,
            initial_pose,
            xi,
            0,
            iterations,
            blurred_query=first_blurred_query,
            intrinsics=intrinsics,
        )
        best_loss = first_loss
        best_pose = first_debug["pose"]
        best_debug = first_debug
        m = [0.0] * 6
        v = [0.0] * 6

        for iteration in range(iterations):
            iteration_blurred_query = self.apply_gaussian_blur(
                query_image, iteration, iterations
            )
            gradient = []
            for idx in range(6):
                xi_plus = xi[:]
                xi_minus = xi[:]
                xi_plus[idx] += self.config.finite_difference_eps
                xi_minus[idx] -= self.config.finite_difference_eps
                loss_plus, _ = self._loss_for_update(
                    query_image,
                    initial_pose,
                    xi_plus,
                    iteration,
                    iterations,
                    blurred_query=iteration_blurred_query,
                    intrinsics=intrinsics,
                )
                loss_minus, _ = self._loss_for_update(
                    query_image,
                    initial_pose,
                    xi_minus,
                    iteration,
                    iterations,
                    blurred_query=iteration_blurred_query,
                    intrinsics=intrinsics,
                )
                gradient.append(
                    (loss_plus - loss_minus) / (2.0 * self.config.finite_difference_eps)
                )

            for idx, grad in enumerate(gradient):
                m[idx] = self.config.beta1 * m[idx] + (1.0 - self.config.beta1) * grad
                v[idx] = self.config.beta2 * v[idx] + (1.0 - self.config.beta2) * (
                    grad * grad
                )
                m_hat = m[idx] / (1.0 - self.config.beta1 ** (iteration + 1))
                v_hat = v[idx] / (1.0 - self.config.beta2 ** (iteration + 1))
                xi[idx] -= (
                    self.config.learning_rate
                    * m_hat
                    / (math.sqrt(v_hat) + self.config.epsilon)
                )

            loss, debug = self._loss_for_update(
                query_image,
                initial_pose,
                xi,
                iteration,
                iterations,
                blurred_query=iteration_blurred_query,
                intrinsics=intrinsics,
            )
            if math.isfinite(loss) and loss < best_loss:
                best_loss = loss
                best_pose = debug["pose"]
                best_debug = debug
            if (
                not math.isfinite(loss)
                or loss > first_loss * self.config.divergence_ratio
            ):
                max_stable_loss = first_loss / max(self.config.accept_loss_ratio, 1e-6)
                stable_low_loss = (
                    first_loss <= self.config.absolute_accept_loss
                    and best_loss <= self.config.absolute_accept_loss
                    and best_loss <= max_stable_loss
                )
                if stable_low_loss:
                    return (
                        PoseEstimate(
                            matrix=best_pose,
                            inliers=0,
                            success=True,
                            source="photometric-refinement",
                            metadata={
                                "rotation_delta_deg": _rotation_error_degrees(
                                    _transpose(best_pose[:3])[:3]
                                ),
                                "improvement": first_loss - best_loss,
                                "meaningful_improvement": (first_loss - best_loss)
                                >= max(
                                    1e-6,
                                    first_loss * self.config.min_relative_improvement,
                                ),
                                "acceptable_final_loss": best_loss
                                <= self.config.absolute_accept_loss,
                                "stable_low_loss": True,
                                "max_stable_loss": max_stable_loss,
                                "accepted_after_divergence": True,
                            },
                        ),
                        best_loss,
                        self._write_debug(
                            debug_dir,
                            best_debug["rendered"],
                            best_debug["mask"],
                            best_debug["opacity"],
                        ),
                    )
                return (
                    PoseEstimate(
                        matrix=initial_pose,
                        inliers=0,
                        success=False,
                        source="refinement-fallback",
                        metadata={"reason": "diverged"},
                    ),
                    first_loss,
                    self._write_debug(
                        debug_dir,
                        first_debug["rendered"],
                        first_debug["mask"],
                        first_debug["opacity"],
                    ),
                )

        improvement = first_loss - best_loss
        meaningful_improvement = improvement >= max(
            1e-6, first_loss * self.config.min_relative_improvement
        )
        acceptable_final_loss = (
            best_loss <= first_loss * self.config.accept_loss_ratio
            and best_loss <= self.config.absolute_accept_loss
        )
        max_stable_loss = first_loss / max(self.config.accept_loss_ratio, 1e-6)
        stable_low_loss = (
            first_loss <= self.config.absolute_accept_loss
            and best_loss <= self.config.absolute_accept_loss
            and best_loss <= max_stable_loss
        )
        success = acceptable_final_loss or stable_low_loss
        result_pose = best_pose if success else initial_pose
        artifacts = self._write_debug(
            debug_dir, best_debug["rendered"], best_debug["mask"], best_debug["opacity"]
        )
        return (
            PoseEstimate(
                matrix=result_pose,
                inliers=0,
                success=success,
                source="photometric-refinement",
                metadata={
                    "rotation_delta_deg": (
                        _rotation_error_degrees(_transpose(result_pose[:3])[:3])
                        if success
                        else 0.0
                    ),
                    "improvement": improvement,
                    "meaningful_improvement": meaningful_improvement,
                    "acceptable_final_loss": acceptable_final_loss,
                    "stable_low_loss": stable_low_loss,
                    "max_stable_loss": max_stable_loss,
                },
            ),
            best_loss if success else first_loss,
            artifacts,
        )

    def _refine_pose_autograd(
        self,
        query_image: ImageTensor,
        initial_pose: list[list[float]],
        num_iters: int | None = None,
        debug_dir: Path | None = None,
        intrinsics=None,
    ) -> tuple[PoseEstimate, float | None, dict[str, str]]:

        iterations = num_iters or self.config.num_iters
        query_tensor = self._tensor_from_image(query_image)
        base_pose = torch.tensor(initial_pose, dtype=torch.float32, device="cuda")
        xi = torch.zeros(6, dtype=torch.float32, device="cuda", requires_grad=True)
        optimizer = torch.optim.Adam([xi], lr=self.config.learning_rate)

        first_loss_value: float | None = None
        best_loss = float("inf")
        best_pose = initial_pose
        best_render: ImageTensor = query_image
        best_mask: list[list[float]] = []
        best_opacity: list[list[float]] = []

        for iteration in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            pose_update = self.exp_map_se3_torch(xi)
            pose_tensor = pose_update @ base_pose
            blurred_query = self._apply_gaussian_blur_tensor(
                query_tensor, iteration, iterations
            )
            rendered_tensor, opacity_tensor = self.renderer.render_torch(
                pose_tensor,
                target_width=query_tensor.shape[2],
                target_height=query_tensor.shape[1],
                intrinsics=intrinsics,
            )
            blurred_render = self._apply_gaussian_blur_tensor(
                rendered_tensor, iteration, iterations
            )
            mask_tensor = self._compute_pixel_mask_tensor(blurred_query, opacity_tensor)
            loss = self._photometric_loss_tensor(
                blurred_query, blurred_render, mask_tensor
            )
            if not torch.isfinite(loss):
                if (
                    first_loss_value is not None
                    and best_loss <= self.config.absolute_accept_loss
                    and first_loss_value <= self.config.absolute_accept_loss
                    and best_loss
                    <= first_loss_value / max(self.config.accept_loss_ratio, 1e-6)
                ):
                    return (
                        PoseEstimate(
                            matrix=best_pose,
                            inliers=0,
                            success=True,
                            source="photometric-refinement-autograd",
                            metadata={
                                "rotation_delta_deg": _rotation_error_degrees(
                                    _transpose(best_pose[:3])[:3]
                                ),
                                "improvement": first_loss_value - best_loss,
                                "meaningful_improvement": (first_loss_value - best_loss)
                                >= max(
                                    1e-6,
                                    first_loss_value
                                    * self.config.min_relative_improvement,
                                ),
                                "acceptable_final_loss": best_loss
                                <= self.config.absolute_accept_loss,
                                "stable_low_loss": True,
                                "max_stable_loss": first_loss_value
                                / max(self.config.accept_loss_ratio, 1e-6),
                                "accepted_after_non_finite": True,
                            },
                        ),
                        best_loss,
                        self._write_debug(
                            debug_dir, best_render, best_mask, best_opacity
                        ),
                    )
                return (
                    PoseEstimate(
                        matrix=initial_pose,
                        inliers=0,
                        success=False,
                        source="refinement-fallback",
                        metadata={"reason": "non_finite_loss"},
                    ),
                    first_loss_value,
                    {},
                )
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().item())
            if first_loss_value is None:
                first_loss_value = loss_value
            if loss_value < best_loss:
                best_loss = loss_value
                best_pose = pose_tensor.detach().cpu().tolist()
                best_render = self._image_from_tensor(rendered_tensor)
                best_mask = self._gray_from_tensor(mask_tensor.unsqueeze(0))
                best_opacity = self._gray_from_tensor(opacity_tensor)
            if (
                first_loss_value is not None
                and loss_value > first_loss_value * self.config.divergence_ratio
            ):
                max_stable_loss = first_loss_value / max(
                    self.config.accept_loss_ratio, 1e-6
                )
                stable_low_loss = (
                    first_loss_value <= self.config.absolute_accept_loss
                    and best_loss <= self.config.absolute_accept_loss
                    and best_loss <= max_stable_loss
                )
                if stable_low_loss:
                    return (
                        PoseEstimate(
                            matrix=best_pose,
                            inliers=0,
                            success=True,
                            source="photometric-refinement-autograd",
                            metadata={
                                "rotation_delta_deg": _rotation_error_degrees(
                                    _transpose(best_pose[:3])[:3]
                                ),
                                "improvement": first_loss_value - best_loss,
                                "meaningful_improvement": (first_loss_value - best_loss)
                                >= max(
                                    1e-6,
                                    first_loss_value
                                    * self.config.min_relative_improvement,
                                ),
                                "acceptable_final_loss": best_loss
                                <= self.config.absolute_accept_loss,
                                "stable_low_loss": True,
                                "max_stable_loss": max_stable_loss,
                                "accepted_after_divergence": True,
                            },
                        ),
                        best_loss,
                        self._write_debug(
                            debug_dir, best_render, best_mask, best_opacity
                        ),
                    )
                return (
                    PoseEstimate(
                        matrix=initial_pose,
                        inliers=0,
                        success=False,
                        source="refinement-fallback",
                        metadata={"reason": "diverged"},
                    ),
                    first_loss_value,
                    self._write_debug(debug_dir, best_render, best_mask, best_opacity),
                )

        if first_loss_value is None:
            return (
                PoseEstimate(
                    matrix=initial_pose,
                    inliers=0,
                    success=False,
                    source="refinement-fallback",
                    metadata={"reason": "no_iterations"},
                ),
                None,
                {},
            )
        improvement = first_loss_value - best_loss
        meaningful_improvement = improvement >= max(
            1e-6, first_loss_value * self.config.min_relative_improvement
        )
        acceptable_final_loss = (
            best_loss <= first_loss_value * self.config.accept_loss_ratio
            and best_loss <= self.config.absolute_accept_loss
        )
        max_stable_loss = first_loss_value / max(self.config.accept_loss_ratio, 1e-6)
        stable_low_loss = (
            first_loss_value <= self.config.absolute_accept_loss
            and best_loss <= self.config.absolute_accept_loss
            and best_loss <= max_stable_loss
        )
        success = acceptable_final_loss or stable_low_loss
        artifacts = self._write_debug(debug_dir, best_render, best_mask, best_opacity)
        result_pose = best_pose if success else initial_pose
        return (
            PoseEstimate(
                matrix=result_pose,
                inliers=0,
                success=success,
                source="photometric-refinement-autograd",
                metadata={
                    "rotation_delta_deg": (
                        _rotation_error_degrees(_transpose(result_pose[:3])[:3])
                        if success
                        else 0.0
                    ),
                    "improvement": improvement,
                    "meaningful_improvement": meaningful_improvement,
                    "acceptable_final_loss": acceptable_final_loss,
                    "stable_low_loss": stable_low_loss,
                    "max_stable_loss": max_stable_loss,
                },
            ),
            best_loss if success else first_loss_value,
            artifacts,
        )

    def _write_debug(
        self,
        debug_dir: Path | None,
        rendered: ImageTensor,
        mask: list[list[float]],
        opacity: list[list[float]],
    ) -> dict[str, str]:
        if debug_dir is None:
            return {}
        debug_dir.mkdir(parents=True, exist_ok=True)
        render_path = debug_dir / "render.json"
        mask_path = debug_dir / "mask.json"
        opacity_path = debug_dir / "opacity.json"
        render_path.write_text(json.dumps({"image": rendered}))
        mask_path.write_text(json.dumps({"mask": mask}))
        opacity_path.write_text(json.dumps({"opacity": opacity}))
        return {
            "render": str(render_path),
            "mask": str(mask_path),
            "opacity": str(opacity_path),
        }
