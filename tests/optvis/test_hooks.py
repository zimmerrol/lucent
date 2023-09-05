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

import torch

from lucent.optvis import hooks


def test_module_hook():
    class DummyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x, x**2, x**3

    simple_layer = torch.nn.Linear(3, 3)
    complex_layer = DummyLayer()

    x = torch.ones(1, 3) * 2

    simple_hook = hooks.ModuleHook(simple_layer)
    simple_output = simple_layer(x)

    complex_hook = hooks.ModuleHook(complex_layer)
    complex_output = complex_layer(x)

    torch.testing.assert_allclose(simple_hook.features, simple_output)
    assert len(complex_hook.features) == len(complex_output) == 3
    torch.testing.assert_allclose(
        torch.cat(complex_hook.features), torch.cat(complex_output)
    )
