import contextlib

import torch

__all__ = ["redirect_relu"]


@contextlib.contextmanager
def redirect_relu():
    """Redirects the torch.nn.functional.relu function to our custom one."""
    torch.nn.functional.unredirected_relu = torch.nn.functional.relu
    torch.relu = _redirected_relu_fn
    yield
    torch.nn.functional.relu = torch.nn.functional.unredirected_relu


class RedirectedReLUFunction(torch.autograd.Function):
    """A workaround when there is no gradient flow because of gradient clipping of relu.
    This is a reimplementation of the approach in
    lucid (see lucid.misc.redirected_relu_grad).
    Note: this means that the gradient is technically "wrong".
    """

    @staticmethod
    def forward(ctx, input, inplace: bool = False):
        ctx.save_for_backward(input)
        if inplace:
            output = torch.relu_(input)
        else:
            output = torch.relu(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = None

        if ctx.needs_input_grad[0]:
            # Actual gradient of relu.
            grad_input = (grad_output > 0).float()
            # If the input is negative (and the actual gradient, thus, would be zero)
            # we redirect the gradient to the input if the gradient before points
            # in positive direction.
            redirected_grad_input = torch.where(
                input < 0 | grad_input > 0, torch.zeros_like(grad_input), grad_input
            )

            # Only use redirected gradient where nothing got through original gradient.
            grad_input_reshaped = grad_input.view(grad_input.size(0), -1)
            grad_mag = torch.norm(grad_input_reshaped, dim=1)
            grad_input = torch.where(grad_mag > 0, grad_input, redirected_grad_input)

        # Gradient wrt. inplace variable is always None.
        return grad_input, None


_redirected_relu_fn = RedirectedReLUFunction.apply
