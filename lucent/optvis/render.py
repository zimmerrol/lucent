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

import contextlib
import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm

from lucent.misc.io import show
from lucent.optvis import objectives, param, redirections, transform
from lucent.optvis.hooks import ModelHook

ObjectiveT = Union[str, Callable[[torch.Tensor], torch.Tensor]]
ParamsT = Callable[[], Tuple[Sequence[torch.Tensor], Callable[[], torch.Tensor]]]
OptimizerT = Callable[[Sequence[torch.Tensor]], torch.optim.Optimizer]


class RenderInterrupt(Exception):
    pass


def render_vis(
    model: nn.Module,
    objective_f: ObjectiveT,
    params_f: Optional[ParamsT] = None,
    optimizer_f: Optional[OptimizerT] = None,
    transforms: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
    thresholds: Sequence[int] = (512,),
    verbose: bool = False,
    preprocess: bool = True,
    progress: bool = True,
    show_image: bool = True,
    save_image: bool = False,
    image_name: Optional[str] = None,
    show_inline: bool = False,
    fixed_image_size: Optional[int] = None,
    redirected_activation_warmup: int = 16,
    iteration_callback: Optional[
        Callable[
            [
                ModelHook,
                Tuple[torch.Tensor, Sequence[torch.Tensor]],
                torch.Tensor,
                Sequence[torch.Tensor],
            ],
            None,
        ]
    ] = None,
    additional_layers_of_interest: Optional[List[str]] = None,
    return_gradient_masks: bool = False,
    apply_gradient_masks: bool = False,
) -> List[np.ndarray]:
    if params_f is None:
        params_f = lambda: param.image(128)
    # params_f is a function that should return two things
    # params - parameters to update, which we pass to the optimizer
    # image_f - a function that returns an image as a tensor
    params, image_f = params_f()

    image_shape = image_f().shape[-2:]

    if optimizer_f is None:
        optimizer_f = lambda params: torch.optim.Adam(params, lr=5e-2)
    optimizer = optimizer_f(params)

    if transforms is None:
        transforms = transform.get_standard_transforms(max(image_shape))
    transforms = transforms.copy()

    if preprocess:
        if model._get_name() == "InceptionV1":
            # Original Tensorflow InceptionV1 takes input range [-117, 138]
            transforms.append(transform.preprocess_inceptionv1())
        else:
            # Assume we use normalization for torchvision.models
            # See https://pytorch.org/docs/stable/torchvision/models.html
            transforms.append(transform.normalize())

    # Upsample images smaller than 224
    image_shape = image_f().shape
    if fixed_image_size is not None:
        new_size = fixed_image_size
    elif image_shape[2] < 224 or image_shape[3] < 224:
        new_size = 224
    else:
        new_size = None
    if new_size:
        transforms.append(
            torch.nn.Upsample(size=new_size, mode="bilinear", align_corners=True)
        )

    transform_f = transform.compose(transforms)

    objective_f = objectives.as_objective(objective_f)

    if additional_layers_of_interest is None:
        additional_layers_of_interest = []

    with ModelHook(
        model, image_f, objective_f.relevant_layers + additional_layers_of_interest
    ) as hook, contextlib.ExitStack() as stack:
        if redirected_activation_warmup:
            # We use an ExitStack to make sure that that replacement of the activation
            # functions in torch with our redirect ones is undone when we exit
            # the context.
            stack.enter_context(redirections.redirect_relu())
            stack.enter_context(redirections.redirect_gelu())

            warnings.warn(
                "Using redirected activations at the beginning of "
                "optimization. This should not be used at the same time "
                "as the legacy RedirectedReLU mechanism."
            )

        if verbose:
            model(transform_f(image_f()))
            print("Initial loss: {:.3f}".format(objective_f(hook)))

        @torch.no_grad()
        def _get_potentially_masked_image_and_mask(image: torch.Tensor) -> Tuple[
            np.ndarray, np.ndarray]:
            image_np = tensor_to_img_array(image)
            if apply_gradient_masks or return_gradient_masks:
                image_gradient_mask = torch.mean(
                    torch.abs(torch.stack(image_gradients)), dim=0)
                # image_gradient_mask = torch.softmax(
                #    image_gradient_mask.view(
                #        image_gradient_mask.shape[0], -1), dim=-1).view(
                #    image_gradient_mask.shape)
                image_gradient_mask -= torch.min(image_gradient_mask)
                image_gradient_mask /= torch.max(image_gradient_mask)
                image_gradient_mask = image_gradient_mask.numpy().transpose(
                    (0, 2, 3, 1))
                if apply_gradient_masks:
                    image_np = image_np * image_gradient_mask
            else:
                image_gradient_mask = None

            return image_np, image_gradient_mask

        images = []
        image_gradient_masks = []
        image_gradients = []
        try:
            for i in tqdm(range(1, max(thresholds) + 1), disable=(not progress)):
                optimizer.zero_grad()
                try:
                    image = image_f()
                    model(transform_f(image))
                except RuntimeError as ex:
                    if i == 1:
                        # Only display the warning message
                        # on the first iteration, no need to do that
                        # every iteration
                        warnings.warn(
                            "Some layers could not be computed because the size of the "
                            "image is not big enough. It is fine, as long as the non"
                            "computed layers are not used in the objective function"
                            f"(exception details: '{ex}')"
                        )
                    raise ex
                loss, sublosses = objective_f(hook, True)
                gradients = torch.autograd.grad(loss, params + [image])
                for p, g in zip(params, gradients[:-1]):
                    p.grad = g
                image_gradients.append(gradients[-1].detach().cpu())
                optimizer.step()

                if i in thresholds:
                    potentially_masked_image, image_gradient_mask = (
                        _get_potentially_masked_image_and_mask(image))
                    if return_gradient_masks:
                        image_gradient_masks.append(image_gradient_mask)
                    if verbose:
                        print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
                        if show_inline:
                            show(potentially_masked_image)
                    images.append(potentially_masked_image)

                if iteration_callback:
                    try:
                        iteration_callback(hook, (loss, sublosses), image, params)
                    except RenderInterrupt:
                        # This is a special exception that allows to stop the rendering
                        # process from the callback.
                        # This is useful, e.g., to stop the rendering process
                        # when the loss is below a certain threshold.
                        print("Interrupted optimization at step {:d}.".format(i))
                        if verbose:
                            print(
                                "Loss at step {}: {:.3f}".format(i, objective_f(hook))
                            )
                        images.append(_get_potentially_masked_image_and_mask(image)[0])
                        break

                if i == redirected_activation_warmup:
                    # Stop using redirected versions of activation functions after
                    # redirected_activation_warmup iterations
                    # (for redirected_activation_warmup = 16, this is a heuristic
                    # from lucid).
                    stack.close()

        except KeyboardInterrupt:
            print("Interrupted optimization at step {:d}.".format(i))
            if verbose:
                print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
            potentially_masked_image, image_gradient_mask = (
                _get_potentially_masked_image_and_mask(image_f()))
            images.append(potentially_masked_image)
            if return_gradient_masks:
                image_gradient_masks.append(image_gradient_mask)

    if save_image:
        export(image_f(), image_name)
    if show_inline:
        show(tensor_to_img_array(image_f()))
    elif show_image:
        view(image_f())
    if return_gradient_masks:
        return images, image_gradient_masks
    else:
        return images


def tensor_to_img_array(tensor: torch.Tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image


def view(tensor: torch.Tensor):
    image = tensor_to_img_array(tensor)
    assert len(image.shape) in [
        3,
        4,
    ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
    # Change dtype for PIL.Image
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    Image.fromarray(image).show()


def export(tensor: torch.Tensor, image_name: Optional[str] = None):
    image_name = image_name or "image.jpg"
    image = tensor_to_img_array(tensor)
    assert len(image.shape) in [
        3,
        4,
    ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
    # Change dtype for PIL.Image
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    Image.fromarray(image).save(image_name)
