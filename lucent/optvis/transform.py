# Copyright 2020 The Lucent Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import, division, print_function

import math
import warnings
from typing import Callable, List, Optional, Sequence, Tuple

import kornia
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize

KORNIA_VERSION = kornia.__version__


def jitter(d: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Randomly jitter the image.

    Args:
        d: How much the image will be jittered in both directions.

    Returns:
        A function that takes in an image and returns a jittered version of it.
    """
    assert d > 0, "Jitter parameter d must be more than 0, currently {}".format(d)

    def inner(image_t: torch.Tensor) -> torch.Tensor:
        w, h = image_t.shape[-2:]
        sx = np.random.choice([-1, 1])
        sy = np.random.choice([-1, 1])
        dx = np.random.choice(d + 1)
        dy = np.random.choice(d + 1)
        if sx > 0:
            image_t = image_t[..., dx:, :].contiguous()
        else:
            image_t = image_t[..., : w - dx, :].contiguous()

        if sy > 0:
            image_t = image_t[..., dy:, :].contiguous()
        else:
            image_t = image_t[..., : h - dy, :].contiguous()

        return image_t

    return inner


def pad(
    w: int, mode: str = "reflect", constant_value: float = 0.5
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Pad the image.

    Args:
        w: How much the image will be padded in both directions.
        mode: Padding mode. One of "constant", "reflect", "replicate", or "circular".
        constant_value: Value to pad with if mode is "constant".

    Returns:
        A function that takes in an image and returns a padded version of it.
    """
    if mode != "constant":
        constant_value = 0

    def inner(image_t: torch.Tensor) -> torch.Tensor:
        return F.pad(
            image_t,
            [w] * 4,
            mode=mode,
            value=constant_value,
        )

    return inner


def random_scale(
    scales: Sequence[float], pad_constant_value: float = 0.5
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Randomly scale the image.

    Args:
        scales: A list of scales to randomly choose from.
        pad_constant_value: Value to pad with.

    Returns:
        A function that takes in an image and returns a scaled version of it.
    """
    def inner(image_t: torch.Tensor) -> torch.Tensor:
        scale = np.random.choice(scales)
        shp = image_t.shape[2:]
        scale_shape = tuple([_roundup(scale * d) for d in shp])
        pad_x = max(0, _roundup((shp[1] - scale_shape[1]) / 2))
        pad_y = max(0, _roundup((shp[0] - scale_shape[0]) / 2))
        upsample = torch.nn.Upsample(
            size=scale_shape, mode="bilinear", align_corners=True
        )
        return F.pad(upsample(image_t), [pad_y, pad_x] * 2, value=pad_constant_value)

    return inner


def random_rotate(
    angles: Sequence[float], units: str = "degrees"
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Randomly rotate the image.

    Args:
        angles: A list of angles to randomly choose from.
        units: Units of the angles. One of "degrees" or "radians".

    Returns:
        A function that takes in an image and returns a rotated version of it.
    """
    def inner(image_t: torch.Tensor) -> torch.Tensor:
        b, _, h, w = image_t.shape
        # kornia takes degrees
        alpha = _rads2angle(np.random.choice(angles), units)
        angle = torch.ones(b) * alpha
        if KORNIA_VERSION < "0.4.0":
            scale = torch.ones(b)
        else:
            scale = torch.ones(b, 2)
        center = torch.ones(b, 2)
        center[..., 0] = (image_t.shape[3] - 1) / 2
        center[..., 1] = (image_t.shape[2] - 1) / 2
        M = kornia.geometry.transform.get_rotation_matrix2d(center, angle, scale).to(
            image_t.device
        )
        rotated_image = kornia.geometry.transform.warp_affine(
            image_t.float(), M, dsize=(h, w)
        )
        return rotated_image

    return inner


def compose(
    transforms: Sequence[Callable[[torch.Tensor], torch.Tensor]]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Compose a list of transforms.

    Args:
        transforms: A list of transforms to compose.

    Returns:
        A function that takes in an image and returns a transformed version of it.
    """
    def inner(x: torch.Tensor) -> torch.Tensor:
        for transform in transforms:
            x = transform(x)
        return x

    return inner


def _roundup(value: float) -> int:
    return int(math.ceil(value))


def _rads2angle(angle: float, units: str) -> float:
    if units.lower() == "degrees":
        return angle
    if units.lower() in ["radians", "rads", "rad"]:
        angle = angle * 180.0 / np.pi
    return angle


def normalize(
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Normalize the image; default values correspond to standard ImageNet mean/std.

    Args:
        mean: mean values for each channel.
        std: standard deviation values for each channel.

    Returns:
        A function that normalizes the image.
    """
    # ImageNet normalization for torchvision models
    # see https://pytorch.org/docs/stable/torchvision/models.html
    normal = Normalize(mean=mean, std=std)

    def inner(image_t):
        return torch.stack([normal(t) for t in image_t])

    return inner


def center_crop(h: int, w: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Center crop the image to at most the given height and width.

    If the image is smaller than the given height and width, then the image is
    returned as is.

    Args:
        h: height of the crop.
        w: width of the crop.

    Returns:
        A function that center crops the image.
    """

    def inner(x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] >= h and x.shape[3] >= w:
            oy = (x.shape[2] - h) // 2
            ox = (x.shape[3] - w) // 2

            return x[:, :, oy : oy + h, ox : ox + w]
        elif x.shape[2] < h and x.shape[3] < w:
            return x
        else:
            raise ValueError(
                "Either both width and height must be smaller than the "
                "image, or both must be larger."
            )

    return inner


def resize(
    h: int, w: int, interpolation: str = "bilinear"
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Resize the image to at least the given height and width.

    Does not change the image if it is larger than the given height and width.

    Args:
        h: The height to resize to.
        w: The width to resize to.
        interpolation: The interpolation method to use. Must be one of
            "nearest", "bilinear", or "bicubic".

    Returns:
        A function that resizes the image.
    """

    def inner(x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] >= h and x.shape[3] >= w:
            return x
        return F.interpolate(x, (h, w), mode=interpolation, align_corners=True)

    return inner


def random_resized_crop(
    resize_size: Tuple[int, int],
    crop_center_std: float = 0.15,
    crop_size_mean: float = 0.35,
    crop_size_std: float = 0.05,
    crop_center_mean: float = 0.5,
    square_crops: bool = True,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Randomly crop and resize the image.

    Args:
        resize_size: The size to resize the image to.
        crop_center_std: The standard deviation of the distribution to sample
            the center crop location from.
        crop_size_mean: The mean of the distribution to sample the crop size
            from.
        crop_size_std: The standard deviation of the distribution to sample the
            crop size from.
        crop_center_mean: The mean of the distribution to sample the crop center
            from.
        square_crops: If True, then the crop will be square. Otherwise, the
            crop will be rectangular.

    Returns:
        A function that randomly crops and resizes the image.
    """

    def inner(image_t: torch.Tensor) -> torch.Tensor:
        b, c, w, h = image_t.shape
        dst_boxes = torch.tensor(
            [
                [0, 0],
                [resize_size[0] - 1, 0],
                [resize_size[0] - 1, resize_size[1] - 1],
                [0, resize_size[1] - 1],
            ],
            device=image_t.device,
            dtype=image_t.dtype,
        )
        dst_boxes = torch.tile(dst_boxes.unsqueeze(0), (b, 1, 1))

        src_centers = torch.clamp(
            torch.randn((2, b), device=image_t.device) * crop_center_std
            + crop_center_mean,
            0,
            1,
        )
        src_sizes = (
            torch.clamp(
                torch.randn((2, b), device=image_t.device) * crop_size_std
                + crop_size_mean,
                0.05,
                1.0,
            )
            / 2
        )

        if square_crops:
            src_sizes[1] = src_sizes[0]

        src_boxes_x = torch.stack(
            [
                src_centers[0] - src_sizes[0],
                src_centers[0] + src_sizes[0],
                src_centers[0] + src_sizes[0],
                src_centers[0] - src_sizes[0],
            ],
            1,
        )
        src_boxes_y = torch.stack(
            [
                src_centers[1] - src_sizes[1],
                src_centers[1] - src_sizes[1],
                src_centers[1] + src_sizes[1],
                src_centers[1] + src_sizes[1],
            ],
            1,
        )
        src_boxes_x = torch.clamp(src_boxes_x * w, 0, w - 1)
        src_boxes_y = torch.clamp(src_boxes_y * h, 0, h - 1)
        src_boxes = torch.stack([src_boxes_x, src_boxes_y], 2)

        return kornia.geometry.transform.crop_by_boxes(image_t, src_boxes, dst_boxes)

    return inner


def gaussian_noise(std: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Add Gaussian noise to the image.

    Args:
        std: The standard deviation of the Gaussian noise.

    Returns:
        A function that adds Gaussian noise to the image.
    """
    assert std > 0, "Standard deviation must be positive, currently {}".format(std)

    def inner(image_t: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(image_t) * std
        image_t = image_t + noise
        image_t = torch.clamp(image_t, 0, 1)
        return image_t

    return inner


def uniform_noise(level: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Add uniform noise to the image.

    Args:
        level: The level of the uniform noise. The noise will be in the range
            [-level, level].

    Returns:
        A function that adds uniform noise to the image.
    """

    assert level > 0, "Noise level level must be positive, currently {}".format(level)

    def inner(image_t: torch.Tensor) -> torch.Tensor:
        noise = torch.rand_like(image_t) * 2 * level - level
        image_t = image_t + noise
        image_t = torch.clamp(image_t, 0, 1)
        return image_t

    return inner


def preprocess_inceptionv1() -> Callable[[torch.Tensor], torch.Tensor]:
    # Original Tensorflow's InceptionV1 model
    # takes in [-117, 138]
    # See https://github.com/tensorflow/lucid/blob/master/lucid/modelzoo/other_models/InceptionV1.py#L56
    # Thanks to ProGamerGov for this!
    return lambda x: x * 255 - 117


def get_standard_transforms(
    source_shape: Tuple[int, int], target_shape: Optional[Tuple[int, int]] = None
) -> List[Callable[[torch.Tensor], torch.Tensor]]:
    """Get a list of standard augmentations.

    Args:
        source_shape: The shape of the source image.
        target_shape: The shape of the target image. If None, then no resize
            will be performed.

    Returns:
        A list of standard augmentations.
    """
    unit = max(source_shape) // 32

    augmentations = [
        pad(3 * unit, mode="constant", constant_value=0.5),
        jitter(2 * unit),
        random_scale([1 + (i - 5) / 50.0 for i in range(11)]),
        random_rotate(list(range(-10, 11)) + 5 * [0]),
        jitter(unit),
    ]

    if target_shape:
        augmentations.append(center_crop(*target_shape))
        augmentations.append(resize(*target_shape))
    else:
        warnings.warn("No target shape provided, so no resize will be performed.")

    return augmentations
