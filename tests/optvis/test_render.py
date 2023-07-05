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

from lucent.modelzoo import inceptionv1
from lucent.optvis import param, render


@pytest.fixture(params=[True, False])
def inceptionv1_model(request):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = inceptionv1().to(device).eval()
    if request.param:
        model = torch.nn.DataParallel(model)
    return model


@pytest.fixture(params=[True, False])
def gelu_dummy_model(request):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class DummyModel(torch.nn.Sequential):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.mixed3b = torch.nn.Conv2d(3, 3, 3)
            self.mixed3b_act = torch.nn.GELU()
            self.mixed4a = torch.nn.Conv2d(3, 3, 3)
            self.mixed4a_act = torch.nn.GELU()

        def forward(self, input):
            x = self.mixed3b(input)
            x = self.mixed3b_act(x)
            x = self.mixed4a(x)
            x = self.mixed4a_act(x)
            return x

    model = DummyModel().to(device).eval()
    if request.param:
        model = torch.nn.DataParallel(model)
    return model


def test_render_vis(inceptionv1_model):
    thresholds = (1, 2)
    imgs = render.render_vis(
        inceptionv1_model, "mixed4a:0", thresholds=thresholds, show_image=False
    )
    assert len(imgs) == len(thresholds)
    assert imgs[0].shape == (1, 128, 128, 3)


@pytest.mark.parametrize(
    "model",
    [pytest.lazy_fixture("gelu_dummy_model"), pytest.lazy_fixture("inceptionv1_model")],
)
@pytest.mark.parametrize("redirected_activation_warmup", [0, 16])
def test_redirect_activations(model, redirected_activation_warmup):
    thresholds = (1, 2)
    imgs = render.render_vis(
        model,
        "mixed4a:0",
        thresholds=thresholds,
        show_image=False,
        redirected_activation_warmup=redirected_activation_warmup,
    )
    assert len(imgs) == len(thresholds)
    assert imgs[0].shape == (1, 128, 128, 3)


def test_interrupt_render_vis(inceptionv1_model, capfd):
    def iteration_callback(*args):
        raise render.RenderInterrupt()

    thresholds = (10,)
    imgs = render.render_vis(
        inceptionv1_model,
        "mixed4a:0",
        thresholds=thresholds,
        show_image=False,
        iteration_callback=iteration_callback,
    )
    assert len(imgs) == 1
    assert imgs[0].shape == (1, 128, 128, 3)

    assert "Interrupted optimization at step" in capfd.readouterr().out


def test_modelhook(inceptionv1_model):
    _, image_f = param.image(224)
    with render.ModelHook(inceptionv1_model, image_f) as hook:
        inceptionv1_model(image_f())
        assert hook("input").shape == (1, 3, 224, 224)
        assert hook("labels").shape == (1, 1008)


def test_partial_modelhook(inceptionv1_model):
    _, image_f = param.image(224)
    with render.ModelHook(inceptionv1_model, image_f, layer_names=["mixed4a"]) as hook:
        inceptionv1_model(image_f())
        assert hook("input").shape == (1, 3, 224, 224)
        assert hook("labels").shape == (1, 1008)
        assert hook("mixed4a").shape == (1, 508, 14, 14)
        with pytest.raises(AssertionError):
            print(hook("mixed4b").shape)
