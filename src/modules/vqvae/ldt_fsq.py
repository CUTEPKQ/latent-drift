"""
Finite Scalar Quantization with LDT encoding
Adapted from finite_scalar_quantization.py with LDT-style CNN encoding
"""

from __future__ import annotations
from functools import wraps, partial
from contextlib import nullcontext
from typing import List, Tuple
from collections import namedtuple

import torch
import torch.nn as nn
from torch.nn import Module
from torch import tensor, Tensor, int32
from torch.amp import autocast

from einops import rearrange, pack, unpack

LossBreakdown = namedtuple('LossBreakdown', ['commitment'])


# helper functions

def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


# tensor helpers

def round_ste(z):
    """round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


def floor_ste(z):
    """floor with straight through gradients."""
    zhat = z.floor()
    return z + (zhat - z).detach()


# main class

class LDT_FSQ(Module):
    def __init__(
        self,
        levels: list[int] | tuple[int, ...] = None,
        dim: int | None = None,
        num_codebooks=1,
        keep_num_codebooks_dim: bool | None = None,
        scale: float | None = None,
        allowed_dtypes: tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        channel_first=False,
        projection_has_bias=True,
        return_indices=True,
        force_quantization_f32=True,
        preserve_symmetry=False,
        noise_dropout=0.,
        patch_size=8,
        image_size=[112, 96],
    ):
        super().__init__()

        if levels is None:
            levels = [8, 8, 8, 5, 5, 5]  # Default levels 2^16
            
        if isinstance(levels, tuple):
            levels = list(levels)

        _levels = tensor(levels, dtype=int32)
        self.register_buffer('_levels', _levels, persistent=False)

        _basis = torch.cumprod(tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer('_basis', _basis, persistent=False)

        self.scale = scale

        self.preserve_symmetry = preserve_symmetry
        self.noise_dropout = noise_dropout

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        self.channel_first = channel_first

        # LDT-style projections
        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim, bias=projection_has_bias) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim, bias=projection_has_bias) if has_projections else nn.Identity()

        # LDT-style CNN encoder
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(in_channels=effective_codebook_dim, out_channels=effective_codebook_dim,
                     kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=effective_codebook_dim, out_channels=effective_codebook_dim,
                     kernel_size=3, stride=1, padding=0),
        )

        self.patch_size = patch_size
        self.image_size = image_size

        self.has_projections = has_projections

        self.return_indices = return_indices

        if return_indices:
            self.codebook_size = self._levels.prod().item()
            implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
            self.register_buffer('implicit_codebook', implicit_codebook, persistent=False)

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32
        
        self.register_buffer('zero', torch.tensor(0.), persistent=False)

    def bound(self, z, eps=1e-3):
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        bounded_z = (z + shift).tanh() * half_l - offset
        half_width = self._levels // 2
        return round_ste(bounded_z) / half_width

    def symmetry_preserving_bound(self, z):
        """QL(x) = 2 / (L - 1) * [(L - 1) * (tanh(x) + 1) / 2 + 0.5] - 1"""
        levels_minus_1 = (self._levels - 1)
        scale = 2. / levels_minus_1
        bracket = (levels_minus_1 * (z.tanh() + 1) / 2.) + 0.5
        bracket = floor_ste(bracket)
        return scale * bracket - 1.

    def quantize(self, z):
        """Quantizes z, returns quantized zhat, same shape as z."""
        shape, device, noise_dropout, preserve_symmetry = z.shape[0], z.device, self.noise_dropout, self.preserve_symmetry
        bound_fn = self.symmetry_preserving_bound if preserve_symmetry else self.bound

        bounded_z = bound_fn(z)

        if not self.training or noise_dropout == 0.:
            return bounded_z

        offset_mask = torch.bernoulli(torch.full_like(bounded_z, noise_dropout)).bool()
        offset = torch.rand_like(bounded_z) - 0.5
        bounded_z = torch.where(offset_mask, bounded_z + offset, bounded_z)

        return bounded_z

    def _scale_and_shift(self, zhat_normalized):
        if self.preserve_symmetry:
            return (zhat_normalized + 1.) / (2. / (self._levels - 1))

        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        if self.preserve_symmetry:
            return zhat * (2. / (self._levels - 1)) - 1.

        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def indices_to_level_indices(self, indices):
        """Converts indices to indices at each level"""
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def codes_to_indices(self, zhat):
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).round().to(int32)

    def indices_to_codes(self, indices):
        """Inverse of `codes_to_indices`."""
        assert exists(indices)

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        codes = self.project_out(codes)

        if is_img_or_video or self.channel_first:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def encode(self, input_data, batch_size):
        """LDT-style encoding with CNN"""
        # Project input
        input_data = self.project_in(input_data)  # b * seq * dim
        
        # Permute to channel-first
        input_data = input_data.permute(0, 2, 1).contiguous()  # b * dim * seq
        
        # Reshape to 4D for CNN
        h = int(self.image_size[0] / self.patch_size)
        w = int(self.image_size[1] / self.patch_size)
        input_data = input_data.reshape(batch_size, self.effective_codebook_dim, h, w)
        
        # Apply CNN encoder
        input_data = self.cnn_encoder(input_data)  # Output: b * dim * 1 * 1
        
        # Reshape back
        input_data = input_data.reshape(batch_size, self.effective_codebook_dim, -1)  # b * dim * (h*w)
        input_data = input_data.permute(0, 2, 1).contiguous()  # b * (h*w) * dim
        input_data = input_data.reshape(-1, self.effective_codebook_dim)
        
        return input_data

    def decode2forward(self, quantized_input, batch_size):
        """Decode quantized features back to original dimension"""

        quantized_input = quantized_input.reshape(batch_size, self.effective_codebook_dim, -1)  # b * dim * seq
        quantized_input = quantized_input.permute(0, 2, 1).contiguous()  # b * seq * dim
        
        quantized_input = self.project_out(quantized_input)
        return quantized_input

    def indices2quantized(self, indices, batch_size):
        """Convert indices to quantized output
        
        Parameters
        ----------
        indices : Tensor[B x L]
            Codebook indices
        batch_size : int
            Batch size
            
        Returns
        -------
        Tensor[B x L x D]
            Quantized output projected back to original dimension
        """
        # Convert indices to codes using the implicit 
   
        codes = self._indices_to_codes(indices)
        
        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')
        
        # Reshape for decode2forward: flatten batch and sequence
        codes = codes.reshape(-1, self.effective_codebook_dim)
        
        # Decode back to original dimension
        quantized_input = self.decode2forward(codes, batch_size)
        return quantized_input

    def forward(
        self,
        input_data_first,
        input_data_last,
        return_loss_breakdown=False,
        mask=None,
        return_loss=True,
    ):
        """
        Forward pass with LDT-style encoding
        
        Parameters
        ----------
        input_data_first : Tensor[B x N x D]
            First frame input
        input_data_last : Tensor[B x N x D]
            Last frame input
        return_loss_breakdown : bool
            Whether to return detailed loss breakdown
        mask : Tensor, optional
            Mask for loss computation
        return_loss : bool
            Whether to compute losses
            
        Returns
        -------
        Tensor[B x N x D]
            Quantized output
        Tensor
            Indices
        Tensor
            Aux loss (zero for FSQ)
        """
        batch_size = input_data_first.shape[0]

        # LDT-style encoding
        input_data_first = input_data_first.contiguous()

        #!pre_latent 
        input_data_first = self.encode(input_data_first, batch_size)

        input_data_last = self.encode(input_data_last, batch_size)

        # Compute difference
        z = input_data_last - input_data_first

        # Split out number of codebooks
        z = rearrange(z, 'b (c d) -> b c d', c=self.num_codebooks)

        # Force quantization to f32 if needed
        force_f32 = self.force_quantization_f32
        quantization_context = partial(autocast, 'cuda', enabled=False) if force_f32 else nullcontext

        with quantization_context():
            orig_dtype = z.dtype

            if force_f32 and orig_dtype not in self.allowed_dtypes:
                z = z.float()

            codes = self.quantize(z)

            # Calculate indices
            indices = None
            if self.return_indices:
                indices = self.codes_to_indices(codes)

            codes = rearrange(codes, 'b c d -> b (c d)')
            codes = codes.to(orig_dtype)

        # Decode back to original dimension

        #! target 
        out = self.decode2forward(codes, batch_size)

        # FSQ doesn't have auxiliary loss
        aux_loss = self.zero
        commit_loss = self.zero

        if not self.keep_num_codebooks_dim and self.return_indices:
            indices = maybe(rearrange)(indices, '... 1 -> ...')
            indices = rearrange(indices, '(b l) -> b l', b=batch_size)

        ret = (out, aux_loss, indices)

        if return_loss_breakdown:
            return ret, LossBreakdown(commit_loss)
        else:
            return ret


if __name__ == "__main__":
    quantizer = LDT_FSQ(
        dim=1024,
        num_codebooks=1,
        patch_size=8,
        image_size=[112, 96],
    )

    pre = torch.rand(184, 168, 1024)
    post = torch.rand(184, 168, 1024)


    z_q, aux_loss, indices = quantizer(pre, post)
   
    print(f"Quantized output shape: {z_q.shape}")
    print(f"Indices shape: {indices.shape if indices is not None else None}")
    print(f"Aux loss: {aux_loss}")
