import collections
import typing
from types import TracebackType
from typing import Any, Callable, Optional, OrderedDict, Sequence, Tuple, Type, Union

import torch
from torch import nn


class ModuleHook:
    def __init__(self, module: nn.Module):
        def hook_fn(m: nn.Module, args: Any, output: torch.Tensor):
            def add_to_features(t: torch.Tensor, idx: Optional[int] = None):
                device = t.device
                if idx is None:
                    self._features[str(device)] = t
                else:
                    self._features[f"{idx}_{str(device)}"] = t

            if torch.is_tensor(output):
                add_to_features(output)
            elif isinstance(output, (tuple, list)):
                for idx, out in enumerate(output):
                    if torch.is_tensor(out):
                        add_to_features(out, idx)

        self.hook = module.register_forward_hook(hook_fn)
        self._features: OrderedDict[str, torch.Tensor] = collections.OrderedDict()

    @property
    def features(self) -> Union[None, torch.Tensor, Tuple[torch.Tensor, ...]]:
        keys = list(sorted(self._features.keys()))
        if len(keys) == 0:
            return None
        elif len(keys) == 1:
            return self._features[keys[0]]
        else:
            if "_" in keys[0]:
                # Multiple tensors per device, i.e., a layer with multiple outputs.
                tuple_idxs = list(set([k.split("_")[0] for k in keys]))
                devices = list(set([k.split("_")[1] for k in keys]))

                if len(devices) == 1:
                    # All tensors on the same device, so we can just return a tuple.
                    return tuple([self._features[k] for k in keys])
                else:
                    return tuple(
                        [
                            typing.cast(
                                torch.Tensor,
                                torch.nn.parallel.gather(
                                    [
                                        self._features[k]
                                        for k in keys
                                        if k.startswith(str(idx))
                                    ],
                                    target_device=torch.device(keys[0].split("_")[1]),
                                    dim=0,
                                ),
                            )
                            for idx in tuple_idxs
                        ]
                    )
            else:
                return torch.nn.parallel.gather(
                    [self._features[k] for k in keys],
                    target_device=torch.device(keys[0]),
                )

    def close(self):
        self.hook.remove()


class ModelHook:
    def __init__(
        self,
        model: nn.Module,
        image_f: Optional[Callable[[], torch.Tensor]] = None,
        layer_names: Optional[Sequence[str]] = None,
    ):
        self.model = model
        self.image_f = image_f
        self.features: OrderedDict[str, ModuleHook] = collections.OrderedDict()
        self.layer_names = layer_names

    def __enter__(self):
        hook_all_layers = self.layer_names is not None and "all" in self.layer_names

        # recursive hooking function
        def hook_layers(net, prefix=[]):
            if hasattr(net, "_modules"):
                layers = list(net._modules.items())
                for i, (name, layer) in enumerate(layers):
                    effective_name = "_".join(prefix + [name])
                    if layer is None:
                        # e.g. GoogLeNet's aux1 and aux2 layers
                        continue

                    if self.layer_names is not None and i < len(layers) - 1:
                        # only save activations for chosen layers
                        if (
                            effective_name not in self.layer_names
                            and not hook_all_layers
                        ):
                            # Don't save activations for this layer but check if it
                            # has any layers we want to save.
                            hook_layers(layer, prefix=prefix + [name])
                            continue

                    self.features[effective_name] = ModuleHook(layer)
                    hook_layers(layer, prefix=prefix + [name])

        if isinstance(self.model, torch.nn.DataParallel):
            hook_layers(self.model.module)
        else:
            hook_layers(self.model)

        def hook(layer):
            if layer == "input":
                out = self.image_f()
            elif layer == "labels":
                out = list(self.features.values())[-1].features
            else:
                assert layer in self.features, (
                    f"Invalid layer {layer}. Retrieve the list of layers with "
                    "`lucent.modelzoo.util.get_model_layers(model)`."
                )
                out = self.features[layer].features
            assert out is not None, (
                "There are no saved feature maps. Make sure to put the model in eval "
                "mode, like so: `model.to(device).eval()`. See README for example."
            )
            return out

        return hook

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ):
        for k in self.features.copy():
            self.features[k].close()
            del self.features[k]
