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

import lucent.optvis.objectives
from lucent.modelzoo import inceptionv1
from lucent.optvis import objectives, param, render
from lucent.util import set_seed
from torchvision import models

set_seed(137)


NUM_STEPS = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def inceptionv1_model():
    return inceptionv1().to(device).eval()


@pytest.fixture
def inceptionv1_model_and_channel_mode():
    model = inceptionv1().to(device).eval()
    return model, "first"


@pytest.fixture
def channel_last_conv_linear_model_and_channel_mode():
    class FCLinearModel(torch.nn.Module):
        def __init__(self):
            super(FCLinearModel, self).__init__()
            self.mixed3a_1x1_pre_relu_conv = torch.nn.Linear(3, 64, bias=False)
            self.mixed4a_pool_reduce_pre_relu_conv = torch.nn.Linear(
                64, 64, bias=False
            )
            self.mixed4c_pool_reduce_pre_relu_conv = torch.nn.Linear(
                64, 512, bias=False
            )
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = torch.permute(x, (0, 2, 3, 1))
            x = self.mixed3a_1x1_pre_relu_conv(x)
            x = self.relu(x)
            x = self.mixed4a_pool_reduce_pre_relu_conv(x)
            x = self.relu(x)
            x = self.mixed4c_pool_reduce_pre_relu_conv(x)
            return torch.permute(x, (0, 3, 1, 2))

    return FCLinearModel().to(device).eval(), "last"


@pytest.fixture
def vit_b_16_model():
    model = models.vit_b_16().to(device).eval()
    return model


def assert_gradient_descent(
    objective: lucent.optvis.objectives.ObjectiveT,
    model: torch.nn.Module,
    input_size: int = 224,
    n_attempts: int = 3,
):
    for _ in range(n_attempts):
        params, image = param.image(input_size, batch=2)
        optimizer = torch.optim.Adam(params, lr=0.05)
        with render.ModelHook(model, image) as T:
            objective_f = objectives.as_objective(objective)
            model(image())
            start_value = objective_f(T)
            for _ in range(NUM_STEPS):
                optimizer.zero_grad()
                model(image())
                loss = objective_f(T)
                loss.backward()
                optimizer.step()
            end_value = objective_f(T)
        if start_value > end_value:
            return

    raise AssertionError(
        "Gradient descent did not improve objective value: {0} -> {1}".format(
            start_value, end_value
        )
    )


@pytest.mark.parametrize(
    "model_and_channel_mode",
    [
        pytest.lazy_fixture("inceptionv1_model_and_channel_mode"),
        pytest.lazy_fixture("channel_last_conv_linear_model_and_channel_mode"),
    ],
)
def test_neuron(model_and_channel_mode):
    model, channel_mode = model_and_channel_mode
    objective = objectives.neuron(
        "mixed3a_1x1_pre_relu_conv", 0, channel_mode=channel_mode
    )
    assert_gradient_descent(objective, model)


@pytest.mark.parametrize(
    "model_and_channel_mode",
    [
        pytest.lazy_fixture("inceptionv1_model_and_channel_mode"),
        pytest.lazy_fixture("channel_last_conv_linear_model_and_channel_mode"),
    ],
)
def test_channel(model_and_channel_mode):
    model, channel_mode = model_and_channel_mode
    objective = objectives.channel(
        "mixed3a_1x1_pre_relu_conv", 0, channel_mode=channel_mode
    )
    assert_gradient_descent(objective, model)


@pytest.mark.parametrize(
    "model",
    [
        pytest.lazy_fixture("vit_b_16_model"),
    ],
)
def test_vit_channel(model):
    objective = objectives.vit_channel("encoder_layers_encoder_layer_2_mlp_3", 0)
    assert_gradient_descent(objective, model)


@pytest.mark.parametrize(
    "model_and_channel_mode",
    [
        pytest.lazy_fixture("inceptionv1_model_and_channel_mode"),
        pytest.lazy_fixture("channel_last_conv_linear_model_and_channel_mode"),
    ],
)
def test_neuron_weight(model_and_channel_mode):
    model, channel_mode = model_and_channel_mode
    weight = torch.randn(64).to(device)  # 64 is the channel number of that layer
    objective = objectives.neuron_weight(
        "mixed3a_1x1_pre_relu_conv", weight, channel_mode=channel_mode
    )
    assert_gradient_descent(objective, model)


@pytest.mark.parametrize(
    "model_and_channel_mode",
    [
        pytest.lazy_fixture("inceptionv1_model_and_channel_mode"),
        pytest.lazy_fixture("channel_last_conv_linear_model_and_channel_mode"),
    ],
)
def test_localgroup_weight(model_and_channel_mode):
    model, channel_mode = model_and_channel_mode
    weight = torch.randn(64).to(device)  # 64 is the channel number of that layer
    objective = objectives.localgroup_weight(
        "mixed3a_1x1_pre_relu_conv",
        weight,
        x=10,
        y=10,
        wx=3,
        wy=3,
        channel_mode=channel_mode,
    )  # a 3 by 3 square start from (10, 10)
    assert_gradient_descent(objective, model)


@pytest.mark.parametrize(
    "model_and_channel_mode",
    [
        pytest.lazy_fixture("inceptionv1_model_and_channel_mode"),
        pytest.lazy_fixture("channel_last_conv_linear_model_and_channel_mode"),
    ],
)
def test_channel_weight(model_and_channel_mode):
    model, channel_mode = model_and_channel_mode
    weight = torch.randn(64).to(device)  # 64 is the channel number of that layer
    objective = objectives.channel_weight(
        "mixed3a_1x1_pre_relu_conv", weight, channel_mode=channel_mode
    )
    assert_gradient_descent(objective, model)


def test_sum(inceptionv1_model):
    channel = lambda n: objectives.channel("mixed4a_pool_reduce_pre_relu_conv", n)
    objective = objectives.Objective.sum([channel(21), channel(32)])
    assert_gradient_descent(objective, inceptionv1_model)


def test_linear_transform(inceptionv1_model):
    objective = (
        1 + 1 * -objectives.channel("mixed4a_pool_reduce_pre_relu_conv", 0) / 1 - 1
    )
    assert_gradient_descent(objective, inceptionv1_model)


def test_mul(inceptionv1_model):
    channel = lambda n: objectives.channel("mixed4a_pool_reduce_pre_relu_conv", n)
    objective = channel(21) * channel(32)
    assert_gradient_descent(objective, inceptionv1_model)


def test_div(inceptionv1_model):
    channel = lambda n: objectives.channel("mixed4a_pool_reduce_pre_relu_conv", n)
    objective = channel(21) / channel(32)
    assert_gradient_descent(objective, inceptionv1_model)


def test_sub_objectives(inceptionv1_model):
    objective_values = [
        objectives.channel("mixed4a", 0),
        objectives.channel("mixed5a", 1),
    ]
    objective_a = objectives.Objective.sum(objective_values)
    objective_b = -1 * objective_values[0] + 2 * objective_values[1]
    assert objective_a.sub_objectives == objective_values
    assert len(objective_b.sub_objectives) == 2
    assert objective_b.sub_objectives[0].sub_objectives == [objective_values[0]]
    assert objective_b.sub_objectives[1].sub_objectives == [objective_values[1]]

    params, image = param.image(224, batch=2)
    with render.ModelHook(inceptionv1_model, image) as T:
        inceptionv1_model(image())
        objective_a_values = objective_a(T, return_sub_objectives=True)
        objective_b_values = objective_b(T, return_sub_objectives=True)
        assert len(objective_a_values) == 2
        assert len(objective_b_values) == 2
        assert len(objective_a_values[1]) == 2
        assert len(objective_b_values[1]) == 2
        assert sum(objective_a_values[1]) == objective_a_values[0]
        assert sum(objective_b_values[1]) == objective_b_values[0]


def test_blur_input_each_step(inceptionv1_model):
    objective = objectives.blur_input_each_step()
    assert_gradient_descent(objective, inceptionv1_model)


@pytest.mark.parametrize(
    "model_and_channel_mode",
    [
        pytest.lazy_fixture("inceptionv1_model_and_channel_mode"),
        pytest.lazy_fixture("channel_last_conv_linear_model_and_channel_mode"),
    ],
)
def test_channel_interpolate(model_and_channel_mode):
    model, channel_mode = model_and_channel_mode
    objective = objectives.channel_interpolate(
        "mixed4a_pool_reduce_pre_relu_conv",
        32,
        "mixed4a_pool_reduce_pre_relu_conv",
        33,
        channel_mode=channel_mode,
    )
    assert_gradient_descent(objective, model)


def test_alignment(inceptionv1_model):
    objective = objectives.alignment("mixed4a")
    assert_gradient_descent(objective, inceptionv1_model)


@pytest.mark.parametrize(
    "model_and_channel_mode",
    [
        pytest.lazy_fixture("inceptionv1_model_and_channel_mode"),
        pytest.lazy_fixture("channel_last_conv_linear_model_and_channel_mode"),
    ],
)
def test_diversity(model_and_channel_mode):
    model, channel_mode = model_and_channel_mode
    objective = objectives.channel(
        "mixed4a_pool_reduce_pre_relu_conv", 0, channel_mode=channel_mode
    ) - 100 * objectives.diversity(
        "mixed4a_pool_reduce_pre_relu_conv", channel_mode=channel_mode
    )
    assert_gradient_descent(objective, model)


@pytest.mark.parametrize(
    "model_and_channel_mode",
    [
        pytest.lazy_fixture("inceptionv1_model_and_channel_mode"),
        pytest.lazy_fixture("channel_last_conv_linear_model_and_channel_mode"),
    ],
)
def test_direction(model_and_channel_mode):
    model, channel_mode = model_and_channel_mode
    direction = torch.rand(64) * 1000
    objective = objectives.direction(
        layer="mixed4c_pool_reduce_pre_relu_conv",
        direction=direction,
        channel_mode=channel_mode,
    )
    assert_gradient_descent(objective, model)


@pytest.mark.parametrize(
    "model_and_channel_mode",
    [
        pytest.lazy_fixture("inceptionv1_model_and_channel_mode"),
        pytest.lazy_fixture("channel_last_conv_linear_model_and_channel_mode"),
    ],
)
def test_direction_neuron(model_and_channel_mode):
    model, channel_mode = model_and_channel_mode
    direction = torch.rand(64) * 1000
    objective = objectives.direction_neuron(
        layer="mixed4c_pool_reduce_pre_relu_conv",
        direction=direction,
        channel_mode=channel_mode,
    )
    assert_gradient_descent(objective, model)
