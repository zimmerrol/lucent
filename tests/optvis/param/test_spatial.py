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

import pytest
import torch

from lucent.optvis import param
from lucent.optvis.param.spatial import fft_spectrum_magnitude


def test_pixel():
    shape = (1, 1)
    params, image_f = param.pixel_image(shape)
    assert params[0].shape == shape
    assert image_f().shape == shape


def test_fft():
    shape = (1, 1, 1, 1)
    params, image_f = param.fft_image(shape)
    assert params[0].shape == (1, 1, 1, 1, 2)
    assert image_f().shape == shape


def test_fft_maco():
    shape = (1, 3, 32, 32)
    spectrum_magnitude = torch.rand(shape[1], shape[2], shape[3] // 2 + 1)
    params, image_f = param.fft_maco_image(
        shape, spectrum_magnitude=spectrum_magnitude)
    assert params[0].shape == (1, 3, shape[2], shape[3] // 2 + 1), params[0].shape
    assert image_f().shape == shape
    inferred_spectrum_magnitude = fft_spectrum_magnitude(image_f())[0]
    # We need to ignore the first and last columns of the spectrum magnitude
    # because of the way torch.fft.rfftn works. This should not be a relevant problem
    # because the first and last columns of the spectrum are usually very small,
    # especially for large images.
    inferred_spectrum_magnitude[..., 0] = spectrum_magnitude[..., 0]
    inferred_spectrum_magnitude[..., -1] = spectrum_magnitude[..., -1]
    torch.testing.assert_close(inferred_spectrum_magnitude, spectrum_magnitude)
