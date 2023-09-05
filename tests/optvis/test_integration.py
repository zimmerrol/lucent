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
from lucent.optvis import objectives, param, render, transform
from lucent.modelzoo import inceptionv1


@pytest.fixture
def inceptionv1_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = inceptionv1().to(device).eval()
    return model


@pytest.mark.parametrize("decorrelate", [True, False])
@pytest.mark.parametrize("mode", ["pixel", "fft", "fft_maco"])
def test_integration(inceptionv1_model, decorrelate, mode):
    obj = "mixed3a_1x1_pre_relu_conv:0"
    if mode == "fft_maco":
        inner_kwargs = {"spectrum_magnitude": torch.fft.rfftn(
            torch.randn(3, 224, 224)).abs()}
    else:
        inner_kwargs = {}
    params_f = lambda: param.image(224, decorrelate=decorrelate, mode=mode,
                                  **inner_kwargs)
    optimizer_f = lambda params: torch.optim.Adam(params, lr=0.1)

    images = render.render_vis(
        inceptionv1_model,
        obj,
        (224, 224),
        preprocess="inceptionv1",
        params_f=params_f,
        optimizer_f=optimizer_f,
        thresholds=(1, 2),
        verbose=True,
        show_inline=True,
    )
    start_image, end_image = images[0], images[-1]

    assert (start_image != end_image).any()
