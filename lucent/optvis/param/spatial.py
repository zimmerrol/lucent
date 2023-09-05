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

import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TORCH_VERSION = torch.__version__


def pixel_image(shape, sd=None):
    sd = sd or 0.01
    tensor = (torch.randn(*shape) * sd).to(device).requires_grad_(True)
    return [tensor], lambda: tensor


# From https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py
def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""
    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)


def fft_image(shape, sd=None, decay_power=1):
    batch, channels, h, w = shape
    freqs = rfft2d_freqs(h, w)
    init_val_size = (
        (batch, channels) + freqs.shape + (2,)
    )  # 2 for imaginary and real components
    sd = sd or 0.01

    spectrum_real_imag_t = (
        (torch.randn(*init_val_size) * sd).to(device).requires_grad_(True)
    )

    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    scale = torch.tensor(scale).float()[None, None, ..., None].to(device)

    def inner():
        scaled_spectrum_t = scale * spectrum_real_imag_t
        if TORCH_VERSION >= "1.7.0":
            import torch.fft

            if type(scaled_spectrum_t) is not torch.complex64:
                scaled_spectrum_t = torch.view_as_complex(scaled_spectrum_t)
            image = torch.fft.irfftn(scaled_spectrum_t, s=(h, w), norm="ortho")
        else:
            import torch

            image = torch.irfft(
                scaled_spectrum_t, 2, normalized=True, signal_sizes=(h, w)
            )
        image = image[:batch, :channels, :h, :w]
        # Magic constant from Lucid library; increasing this seems to reduce saturation.
        magic = 4.0
        image = image / magic
        return image

    return [spectrum_real_imag_t], inner


def fft_maco_image(shape, spectrum_magnitude: torch.Tensor):
    batch, channels, h, w = shape
    freqs = rfft2d_freqs(h, w)

    init_phase_shape = (batch, channels) + freqs.shape

    assert spectrum_magnitude.ndim == 3
    spectrum_magnitude = spectrum_magnitude[None].repeat(batch, 1, 1, 1)

    assert spectrum_magnitude.shape == init_phase_shape, (
        f"Shape of magnitude {spectrum_magnitude.shape} does not "
        f"match phase {init_phase_shape}."
    )

    spectrum_phase = (torch.rand(init_phase_shape) * np.pi).to(device)

    spectrum_phase = spectrum_phase.requires_grad_(True)

    def inner():
        if TORCH_VERSION <= "1.7.0":
            raise NotImplementedError("MACO FFT requires torch version >= 1.7.0")

        import torch.fft

        spectrum = torch.polar(spectrum_magnitude, spectrum_phase)
        image = torch.fft.irfftn(spectrum, s=(h, w), norm="ortho")
        image = image[:batch, :channels, :h, :w]

        return image

    return [spectrum_phase], inner
