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

"""High-level wrapper for parameterizing images."""

from __future__ import absolute_import, division, print_function

from typing import Literal

from lucent.optvis.param.color import to_valid_rgb
from lucent.optvis.param.spatial import fft_image, fft_maco_image, pixel_image


def image(
    w,
    h=None,
    batch=None,
    decorrelate=True,
    mode: Literal["pixel", "fft", "maco_fft"] = "fft",
    channels=None,
    **inner_kwargs,
):
    h = h or w
    batch = batch or 1
    ch = channels or 3
    shape = [batch, ch, h, w]
    if mode == "fft_maco":
        params, image_f = fft_maco_image(shape, **inner_kwargs)
    elif mode == "fft":
        params, image_f = fft_image(shape, **inner_kwargs)
    elif mode == "pixel":
        params, image_f = pixel_image(shape, **inner_kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}.")

    if channels:
        output = to_valid_rgb(image_f, decorrelate=False)
    else:
        output = to_valid_rgb(image_f, decorrelate=decorrelate)
    return params, output
