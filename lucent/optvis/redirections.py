import numpy as np
import torch

__all__ = ["redirect_relu", "redirect_gelu"]


class ReLURedirectionGenerator:
    """Redirects the torch.nn.functional.relu function to our custom one."""

    def __enter__(self):
        setattr(torch.nn.functional, "unredirected_relu", torch.nn.functional.relu)
        torch.nn.functional.relu = _redirected_relu

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.nn.functional.relu = getattr(torch.nn.functional, "unredirected_relu")
        delattr(torch.nn.functional, "unredirected_relu")


class GELURedirectionGenerator:
    """Redirects the torch.nn.functional.gelu function to our custom one."""

    def __enter__(self):
        setattr(torch.nn.functional, "unredirected_gelu", torch.nn.functional.gelu)
        torch.nn.functional.gelu = _redirected_gelu

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.nn.functional.gelu = getattr(torch.nn.functional, "unredirected_gelu")
        delattr(torch.nn.functional, "unredirected_gelu")


def redirect_relu() -> ReLURedirectionGenerator:
    """Redirects the torch.nn.functional.relu function to our custom one."""
    return ReLURedirectionGenerator()


def redirect_gelu() -> GELURedirectionGenerator:
    """Redirects the torch.nn.functional.gelu function to our custom one."""
    return GELURedirectionGenerator()


class RedirectedReLUFunction(torch.autograd.Function):
    """A workaround when there is no gradient flow because of gradient clipping of relu.
    This is a reimplementation of the approach in
    lucid (see lucid.misc.redirected_relu_grad).
    Note: this means that the gradient is technically "wrong".
    """

    @staticmethod
    def forward(ctx, input, inplace: bool = False):  # type: ignore
        ctx.save_for_backward(input)
        if not hasattr(torch.nn.functional, "unredirected_relu"):
            raise NotImplementedError(
                "RedirectedReLUFunction requires redirect_relu must be called first."
            )
        return getattr(torch.nn.functional, "unredirected_relu")(input, inplace)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = None

        if ctx.needs_input_grad[0]:
            # Actual gradient of relu.
            grad_input = grad_output * (input > 0).float()
            # If the input is negative (and the actual gradient, thus, would be zero)
            # we redirect the gradient to the input if the gradient before points
            # in positive direction.
            redirected_grad_output = torch.where(
                (input < 0) & (grad_output > 0), torch.zeros_like(grad_output), grad_output
            )

            # Only use redirected gradient where nothing got through original gradient.
            grad_input_reshaped = grad_input.reshape(grad_input.size(0), -1)
            grad_mag = torch.norm(grad_input_reshaped, dim=1)
            grad_mag = grad_mag.view(grad_mag.size(0), *([1] * (grad_input.dim() - 1)))

            grad_input = torch.where(grad_mag > 0, grad_input, redirected_grad_output)

        # Gradient wrt. inplace variable is always None.
        return grad_input, None


class RedirectedGELUFunction(torch.autograd.Function):
    """A workaround when there is no gradient flow because of gradient clipping of gelu.
    Note: this means that the gradient is technically "wrong".
    """

    @staticmethod
    def forward(ctx, input, approximate: str = "none"):  # type: ignore
        ctx.save_for_backward(input)
        if approximate != "none":
            raise NotImplementedError("approximate mode not implemented for gelu.")
        if not hasattr(torch.nn.functional, "unredirected_gelu"):
            raise NotImplementedError(
                "RedirectedGELUFunction requires redirect_gelu must be called first."
            )
        return getattr(torch.nn.functional, "unredirected_gelu")(
            input, approximate=approximate
        )

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = None

        if ctx.needs_input_grad[0]:
            # Actual gradient of gelu.
            gelu_grad = 1 / 2 * (torch.erf(input / np.sqrt(2)) + 1) + (
                input * torch.exp(-(input**2) / 2)
            ) / np.sqrt(2 * np.pi)
            grad_input = grad_output * gelu_grad
            # If the input is negative (and the actual gradient, thus, would be zero)
            # we redirect the gradient to the input if the gradient before points
            # in positive direction.
            redirected_grad_output = torch.where(
                (input < 0.0) & (grad_output > 0),
                torch.zeros_like(grad_output),
                grad_output,
            )

            # Only use redirected gradient where nothing got through original gradient.
            grad_input_reshaped = grad_input.reshape(grad_input.size(0), -1)
            grad_mag = torch.norm(grad_input_reshaped, dim=1)
            grad_mag = grad_mag.view(grad_mag.size(0), *([1] * (grad_input.dim() - 1)))
            grad_input = torch.where(grad_mag > 0, grad_input, redirected_grad_output)

        # Gradient wrt. approximate variable is always None.
        return grad_input, None


def _redirected_relu(x, inplace: bool = False):
    return RedirectedReLUFunction.apply(x, inplace)


def _redirected_gelu(x, approximate: str = "none"):
    return RedirectedGELUFunction.apply(x, approximate)
