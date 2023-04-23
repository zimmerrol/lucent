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

import typing
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    List,
    Protocol,
)

import numpy as np
import torch
import torch.nn.functional as F
from decorator import decorator
from torch import nn

from lucent.optvis.objectives_util import (
    _extract_act_pos,
    _make_arg_str,
    _T_handle_batch,
)

ObjectiveReturnT = Union[torch.Tensor, Tuple[torch.Tensor, Sequence[torch.Tensor]]]
ObjectiveT = Callable[[nn.Module, bool], ObjectiveReturnT]
WrapObjectiveInnerT = Tuple[Callable[[nn.Module], torch.Tensor], List[str]]


class WrapObjectiveOuterT(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> WrapObjectiveInnerT:
        ...

    @property
    def __name__(self) -> str:
        ...


class Objective:
    def __init__(
        self,
        objective_func: ObjectiveT,
        name: str = "",
        description: str = "",
        relevant_layers: Optional[List[str]] = None,
        sub_objectives: Optional[Sequence["Objective"]] = None,
    ):
        self.objective_func = objective_func
        self.name = name
        self.description = description
        if sub_objectives is None:
            sub_objectives = []
        self.sub_objectives = sub_objectives
        if relevant_layers is None:
            relevant_layers = []
        self._relevant_layers = relevant_layers

    def __call__(
        self, model: torch.nn.Module, return_sub_objectives: bool = False
    ) -> ObjectiveReturnT:
        return self.objective_func(model, return_sub_objectives)

    @property
    def relevant_layers(self) -> List[str]:
        return self._relevant_layers + [
            rl for so in self.sub_objectives for rl in so.relevant_layers
        ]

    def __add__(self, other):
        if isinstance(other, (int, float)):

            def objective_func(
                model: torch.nn.Module, return_sub_objectives: bool = False
            ) -> ObjectiveReturnT:
                inner = self(model, return_sub_objectives)

                if not return_sub_objectives:
                    assert isinstance(inner, torch.Tensor)
                    return inner + other
                else:
                    assert isinstance(inner, (tuple, list))
                    assert isinstance(inner[0], torch.Tensor)
                    assert isinstance(inner[1], (tuple, list))
                    return inner[0] + other, inner[1]

            name = self.name
            description = self.description
            sub_objectives = self.sub_objectives
        else:

            def objective_func(
                model: torch.nn.Module, return_sub_objectives: bool = False
            ) -> ObjectiveReturnT:
                inner_left = self(model, return_sub_objectives)
                inner_right = other(model, return_sub_objectives)
                if not return_sub_objectives:
                    return inner_left + inner_right
                else:
                    return (
                        inner_left[0] + inner_right[0],
                        inner_left[1]
                        + inner_right[1]
                        + [inner_left[0], inner_right[0]],
                    )

            name = ", ".join([self.name, other.name])
            description = (
                "Sum(" + " +\n".join([self.description, other.description]) + ")"
            )
            sub_objectives = [self, other]

        return Objective(
            objective_func,
            name=name,
            description=description,
            sub_objectives=sub_objectives,
        )

    @staticmethod
    def sum(objs: Sequence["Objective"]):
        def objective_func(
            model: torch.nn.Module, return_sub_objectives: bool = False
        ) -> ObjectiveReturnT:
            inners = [obj(model, return_sub_objectives) for obj in objs]
            if not return_sub_objectives:
                assert all(isinstance(inner, torch.Tensor) for inner in inners)
                return sum(inners)  # type: ignore
            else:
                assert all(isinstance(inner, (tuple, list)) for inner in inners)
                assert all(isinstance(inner[0], torch.Tensor) for inner in inners)
                assert all(isinstance(inner[1], (tuple, list)) for inner in inners)
                return sum(inner[0] for inner in inners), [
                    it for inner in inners for it in inner[1]
                ] + [
                    inner[0] for inner in inners
                ]  # type: ignore

        descriptions = [obj.description for obj in objs]
        description = "Sum(" + " +\n".join(descriptions) + ")"
        sub_objectives = objs
        names = [obj.name for obj in objs]
        name = ", ".join(names)

        return Objective(
            objective_func,
            name=name,
            description=description,
            sub_objectives=sub_objectives,
        )

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        return self + (-1 * other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):

            def objective_func(
                model: torch.nn.Module, return_sub_objectives: bool = False
            ) -> ObjectiveReturnT:
                inner = self(model, return_sub_objectives)
                if not return_sub_objectives:
                    return inner * other
                else:
                    assert isinstance(inner, (tuple, list))
                    assert isinstance(inner[0], torch.Tensor)
                    return inner[0] * other, inner[1]

            return Objective(
                objective_func,
                name=self.name,
                description=self.description,
                sub_objectives=[self],
            )
        elif isinstance(other, Objective):

            def objective_func(
                model: torch.nn.Module, return_sub_objectives: bool = False
            ) -> ObjectiveReturnT:
                inner_left = self(model, return_sub_objectives)
                inner_right = other(model, return_sub_objectives)
                if not return_sub_objectives:
                    return inner_left * inner_right
                else:
                    return (
                        inner_left[0] * inner_right[0],
                        inner_left[1]
                        + inner_right[1]
                        + [inner_left[0], inner_right[0]],
                    )

            description = (
                "Mult(" + " +\n".join([self.description, other.description]) + ")"
            )

            return Objective(
                objective_func,
                name=self.name,
                description=description,
                sub_objectives=[self, other],
            )
        else:
            # Note: In original Lucid library, objectives can be multiplied with non-numbers
            # Removing for now until we find a good use case
            raise TypeError(
                "Can only multiply by int, float or Objective. "
                "Received type " + str(type(other))
            )

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self.__mul__(1 / other)
        elif isinstance(other, Objective):

            def objective_func(
                model: torch.nn.Module, return_sub_objectives: bool = False
            ) -> ObjectiveReturnT:
                inner_left = self(model, return_sub_objectives)
                inner_right = other(model, return_sub_objectives)
                if not return_sub_objectives:
                    return inner_left / inner_right
                else:
                    return (
                        inner_left[0] / inner_right[0],
                        inner_left[1]
                        + inner_right[1]
                        + [inner_left[0], inner_right[0]],
                    )

            description = (
                "Div(" + " +\n".join([self.description, other.description]) + ")"
            )

            return Objective(
                objective_func,
                name=self.name,
                description=description,
                sub_objectives=[self, other],
            )
        else:
            raise TypeError(
                "Can only divide by int, float or Objective. "
                "Received type " + str(type(other))
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)


def wrap_objective():
    @decorator
    def inner(func: WrapObjectiveOuterT, *args, **kwds) -> Objective:
        inner_func, relevant_layers = func(*args, **kwds)

        def objective_func(
            model: torch.nn.Module, return_sub_objectives: bool = False
        ) -> ObjectiveReturnT:
            # For atomic objectives there are no sub objectives, thus, we return an
            # empty list.
            if return_sub_objectives:
                return inner_func(model), []
            else:
                return inner_func(model)

        objective_name = func.__name__
        args_str = " [" + ", ".join([_make_arg_str(arg) for arg in args]) + "]"
        description = objective_name.title() + args_str

        return Objective(objective_func, objective_name, description, relevant_layers)

    return inner


def handle_batch(batch=None):
    return lambda f: lambda model: f(_T_handle_batch(model, batch=batch))


@wrap_objective()
def neuron(
    layer: str,
    n_channel,
    x=None,
    y=None,
    channel_mode: Union[Literal["first"], Literal["last"]] = "first",
    batch=None,
) -> WrapObjectiveInnerT:
    """Visualize a single neuron of a single channel.

    Defaults to the center neuron. When width and height are even numbers, we
    choose the neuron in the bottom right of the center 2x2 neurons.

    Odd width & height:               Even width & height:

    +---+---+---+                     +---+---+---+---+
    |   |   |   |                     |   |   |   |   |
    +---+---+---+                     +---+---+---+---+
    |   | X |   |                     |   |   |   |   |
    +---+---+---+                     +---+---+---+---+
    |   |   |   |                     |   |   | X |   |
    +---+---+---+                     +---+---+---+---+
                                      |   |   |   |   |
                                      +---+---+---+---+

    """

    if channel_mode not in ("first", "last"):
        raise ValueError("channel_mode must be 'first' or 'last.")

    @handle_batch(batch)
    def inner(model: nn.Module):
        layer_t = model(layer)
        layer_t = _extract_act_pos(layer_t, x, y, channel_mode)
        return -layer_t[:, n_channel].mean()

    return inner, [layer]


@wrap_objective()
def channel(
    layer: str,
    n_channel,
    channel_mode: Union[Literal["first"], Literal["last"]] = "first",
    batch=None,
) -> WrapObjectiveInnerT:
    """Visualize a single channel"""

    if channel_mode not in ("first", "last"):
        raise ValueError("channel_mode must be 'first' or 'last.")

    @handle_batch(batch)
    def inner(model: nn.Module) -> torch.Tensor:
        if channel_mode == "first":
            return -model(layer)[:, n_channel].mean()
        else:
            return -model(layer)[..., n_channel].mean()

    return inner, [layer]


@wrap_objective()
def neuron_weight(
    layer: str,
    weight,
    x=None,
    y=None,
    channel_mode: Union[Literal["first"], Literal["last"]] = "first",
    batch=None,
) -> WrapObjectiveInnerT:
    """Linearly weighted channel activation at one location as objective
    weight: a torch Tensor vector same length as channel.
    """

    @handle_batch(batch)
    def inner(model: nn.Module):
        layer_t = model(layer)
        layer_t = _extract_act_pos(layer_t, x, y, channel_mode)
        if weight is None:
            return -layer_t.mean()
        else:
            return -(layer_t.squeeze() * weight).mean()

    return inner, [layer]


@wrap_objective()
def channel_weight(
    layer: str,
    weight,
    batch=None,
    channel_mode: Union[Literal["first"], Literal["last"]] = "first",
) -> WrapObjectiveInnerT:
    """Linearly weighted channel activation as objective
    weight: a torch Tensor vector same length as channel."""
    if channel_mode == "first":
        weight = weight.view((1, -1, 1, 1))
    elif channel_mode == "last":
        weight = weight.view((1, 1, 1, -1))
    else:
        raise ValueError("channel_mode must be 'first' or 'last.")

    @handle_batch(batch)
    def inner(model: nn.Module):
        layer_t = model(layer)
        return -(layer_t * weight).mean()

    return inner, [layer]


@wrap_objective()
def localgroup_weight(
    layer: str,
    weight=None,
    x=None,
    y=None,
    wx=1,
    wy=1,
    channel_mode: Union[Literal["first"], Literal["last"]] = "first",
    batch=None,
) -> WrapObjectiveInnerT:
    """Linearly weighted channel activation around some spot as objective
    weight: a torch Tensor vector same length as channel."""

    if channel_mode not in ("first", "last"):
        raise ValueError("channel_mode must be 'first' or 'last.")

    @handle_batch(batch)
    def inner(model: nn.Module):
        layer_t = model(layer)

        if channel_mode == "last":
            layer_t = layer_t.permute(0, 3, 1, 2)

        if weight is None:
            return -(layer_t[:, :, y : y + wy, x : x + wx]).mean()
        else:
            return -(
                layer_t[:, :, y : y + wy, x : x + wx] * weight.view(1, -1, 1, 1)
            ).mean()

    return inner, [layer]


@wrap_objective()
def direction(
    layer: str,
    direction: torch.Tensor,
    channel_mode: Union[Literal["first"], Literal["last"]] = "first",
    batch: Optional[int] = None,
) -> WrapObjectiveInnerT:
    """Visualize a direction

    InceptionV1 example:
    > direction = torch.rand(512, device=device)
    > obj = objectives.direction(layer='mixed4c', direction=direction)

    Args:
        layer: Name of layer in model (string)
        direction: Direction to visualize. torch.Tensor of shape (num_channels,)
        batch: Batch number (int)

    Returns:
        Objective

    """

    if channel_mode == "last":
        direction = direction.view(1, 1, 1, -1)
    elif channel_mode == "first":
        direction = direction.view(1, -1, 1, 1)
    else:
        raise ValueError("channel_mode must be 'first' or 'last.")

    @handle_batch(batch)
    def inner(model):
        return -torch.nn.CosineSimilarity(dim=1)(direction, model(layer)).mean()

    return inner, [layer]


@wrap_objective()
def direction_neuron(
    layer: str,
    direction: torch.Tensor,
    x=None,
    y=None,
    channel_mode: Union[Literal["first"], Literal["last"]] = "first",
    batch=None,
) -> WrapObjectiveInnerT:
    """Visualize a single (x, y) position along the given direction

    Similar to the neuron objective, defaults to the center neuron.

    InceptionV1 example:
    > direction = torch.rand(512, device=device)
    > obj = objectives.direction_neuron(layer='mixed4c', direction=direction)

    Args:
        layer: Name of layer in model (string)
        direction: Direction to visualize. torch.Tensor of shape (num_channels,)
        batch: Batch number (int)

    Returns:
        Objective

    """

    if channel_mode == "last":
        direction = direction.view(1, 1, 1, -1)
    elif channel_mode == "first":
        direction = direction.view(1, -1, 1, 1)
    else:
        raise ValueError("channel_mode must be 'first' or 'last.")

    @handle_batch(batch)
    def inner(model: nn.Module):
        # breakpoint()
        layer_t = model(layer)
        layer_t = _extract_act_pos(layer_t, x, y, channel_mode)
        return -torch.nn.CosineSimilarity(dim=1)(direction, layer_t).mean()

    return inner, [layer]


def _torch_blur(tensor: torch.Tensor, out_c: int = 3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    depth = tensor.shape[1]
    weight = np.zeros([depth, depth, out_c, out_c])
    for ch in range(depth):
        weight_ch = weight[ch, ch, :, :]
        weight_ch[:, :] = 0.5
        weight_ch[1:-1, 1:-1] = 1.0
    weight_t = torch.tensor(weight).float().to(device)
    conv_f = lambda t: F.conv2d(t, weight_t, None, 1, 1)
    return conv_f(tensor) / conv_f(torch.ones_like(tensor))


@wrap_objective()
def blur_input_each_step():
    """Minimizing this objective is equivelant to blurring input each step.
    Optimizing (-k)*blur_input_each_step() is equivelant to:
    input <- (1-k)*input + k*blur(input)
    An operation that was used in early feature visualization work.
    See Nguyen, et al., 2015.
    """

    def inner(T):
        t_input = T("input")
        with torch.no_grad():
            t_input_blurred = _torch_blur(t_input)
        return -0.5 * torch.sum((t_input - t_input_blurred) ** 2)

    return inner, []


@wrap_objective()
def channel_interpolate(
    layer1: str,
    n_channel1: int,
    layer2: str,
    n_channel2: int,
    channel_mode: Union[Literal["first"], Literal["last"]] = "first",
) -> WrapObjectiveInnerT:
    """Interpolate between layer1, n_channel1 and layer2, n_channel2.
    Optimize for a convex combination of layer1, n_channel1 and
    layer2, n_channel2, transitioning across the batch.
    Args:
        layer1: layer to optimize 100% at batch=0.
        n_channel1: neuron index to optimize 100% at batch=0.
        layer2: layer to optimize 100% at batch=N.
        n_channel2: neuron index to optimize 100% at batch=N.
    Returns:
        Objective
    """

    if channel_mode not in ("first", "last"):
        raise ValueError("channel_mode must be 'first' or 'last.")

    def inner(model: nn.Module) -> torch.Tensor:
        batch_n = list(model(layer1).shape)[0]

        layer1_t = model(layer1)
        layer2_t = model(layer2)

        if channel_mode == "last":
            layer1_t = layer1_t.permute(0, 3, 1, 2)
            layer2_t = layer2_t.permute(0, 3, 1, 2)

        arr1 = layer1_t[:, n_channel1]
        arr2 = layer2_t[:, n_channel2]
        weights = np.arange(batch_n) / (batch_n - 1)
        losses = []
        for n in range(batch_n):
            loss = (1 - weights[n]) * arr1[n].mean()
            loss -= weights[n] * arr2[n].mean()
            losses.append(loss)
        sum_loss = sum(losses)
        return sum_loss

    return inner, [layer1, layer2]


@wrap_objective()
def alignment(
    layer: str, decay_ratio: float = 2
) -> Tuple[Callable[[nn.Module], torch.Tensor], List[str]]:
    """Encourage neighboring images to be similar.
    When visualizing the interpolation between two objectives, it's often
    desirable to encourage analogous objects to be drawn in the same position,
    to make them more comparable.
    This term penalizes L2 distance between neighboring images, as evaluated at
    layer.
    In general, we find this most effective if used with a parameterization that
    shares across the batch. (In fact, that works quite well by itself, so this
    function may just be obsolete.)
    Args:
        layer: layer to penalize at.
        decay_ratio: how much to decay penalty as images move apart in batch.
    Returns:
        Objective.
    """

    def inner(model: nn.Module):
        batch_n = list(model(layer).shape)[0]
        layer_t = model(layer)
        accum = 0
        for d in [1, 2, 3, 4]:
            for i in range(batch_n - d):
                a, b = i, i + d
                arr_a, arr_b = layer_t[a], layer_t[b]
                accum += ((arr_a - arr_b) ** 2).mean() / decay_ratio ** float(d)
        return accum

    return inner, [layer]


@wrap_objective()
def diversity(
    layer: str, channel_mode: Union[Literal["first"], Literal["last"]] = "first"
) -> WrapObjectiveInnerT:
    """Encourage diversity between each batch element.

    A neural net feature often responds to multiple things, but naive feature
    visualization often only shows us one. If you optimize a batch of images,
    this objective will encourage them all to be different.

    In particular, it calculates the correlation matrix of activations at layer
    for each image, and then penalizes cosine similarity between them. This is
    very similar to ideas in style transfer, except we're *penalizing* style
    similarity instead of encouraging it.

    Args:
        layer: layer to evaluate activation correlations on.

    Returns:
        Objective.
    """

    def inner(model: nn.Module):
        layer_t = model(layer)

        if channel_mode == "last":
            layer_t = layer_t.permute(0, 3, 1, 2)

        batch, channels, _, _ = layer_t.shape
        flattened = layer_t.view(batch, channels, -1)
        grams = torch.matmul(flattened, torch.transpose(flattened, 1, 2))
        grams = F.normalize(grams, p=2, dim=(1, 2))  # type: ignore
        return (
            -sum(
                [
                    sum([(grams[i] * grams[j]).sum() for j in range(batch) if j != i])
                    for i in range(batch)
                ]
            )
            / batch
        )

    return inner, [layer]


def as_objective(
    obj: Union[str, Objective, Callable[[nn.Module], torch.Tensor]]
) -> Objective:
    """Convert obj into Objective class.

    Strings of the form "layer:n" become the Objective channel(layer, n).
    Objectives are returned unchanged.

    Args:
        obj: string or Objective.

    Returns:
        Objective
    """
    if isinstance(obj, Objective):
        return obj
    if callable(obj):
        obj = typing.cast(Callable[[nn.Module], torch.Tensor], obj)
        return wrap_objective()(func=lambda *args, **kwargs: (obj, ["all"]))
    if isinstance(obj, str):
        layer, chn_s = obj.split(":")
        layer, chn = layer.strip(), int(chn_s)
        return channel(layer, chn)
