

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision.models as tvm
from typing import Tuple, Optional
# from saffron.mamba.mamba import Mamba,MambaConfig
from zeta.nn import SSM
from einops import rearrange, repeat
import math


class MambaEncoder(nn.Module):
    """A encoder model with self attention mechanism."""

    def __init__(
        self,
        n_layers: int,
        n_head: int,
        d_k: int,
        d_v: int,
        d_model: int,
        d_inner: int,
        dropout: float,
        activation_attn: str = "softmax",
        activation_ff: str = "relu",
        prenorm: bool = False,
    ) -> None:

        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        # self.layer_stack = Mamba(
        #     # This module uses roughly 3 * expand * d_model^2 parameters
        #     d_model=d_model, # Model dimension d_model
        #     d_state=16,  # SSM state expansion factor
        #     d_conv=4,    # Local convolution width
        #     expand=2,    # Block expansion factor
        # )
        self.dt_rank = math.ceil(d_model / 16)
        self.layer_stack = nn.ModuleList(
            [
                VisionEncoderMambaBlock(
                    dim=d_model,
                    dt_rank=self.dt_rank,
                    dim_inner=d_inner,
                    d_state=16,

                )
                for _ in range(n_layers)
            ]
        )
        # self.layer_stack = Mamba(mambaconfig)
        self.prenorm = prenorm
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        enc_output = self.dropout(x + pos_enc)
        # if not self.prenorm:  # post-norm
        enc_output = self.layer_norm(enc_output)

        # expand batch as 1

        enc_output = enc_output.unsqueeze(0)

        # print(enc_output.shape)
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output)

        # enc_output = self.layer_stack(enc_output)

        # if self.prenorm:
        #     enc_output = self.layer_norm(enc_output)  # pre-norm

        enc_output = enc_output.squeeze(0)


        return enc_output


class VisionEncoderMambaBlock(nn.Module):
    """
    VisionMambaBlock is a module that implements the Mamba block from the paper
    Vision Mamba: Efficient Visual Representation Learning with Bidirectional
    State Space Model

    Args:
        dim (int): The input dimension of the input tensor.
        heads (int): The number of heads in the multi-head attention mechanism.
        dt_rank (int): The rank of the state space model.
        dim_inner (int): The dimension of the inner layer of the
            multi-head attention.
        d_state (int): The dimension of the state space model.


    Example:
    >>> block = VisionMambaBlock(dim=256, heads=8, dt_rank=32,
            dim_inner=512, d_state=256)
    >>> x = torch.randn(1, 32, 256)
    >>> out = block(x)
    >>> out.shape
    torch.Size([1, 32, 256])
    """

    def __init__(
        self,
        dim: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
    ):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        # self.forward_conv1d = nn.Conv1d(
        #     in_channels=dim, out_channels=dim, kernel_size=1
        # )
        self.forward_conv1d = nn.Conv1d(
            in_channels=dim_inner, out_channels=dim_inner, kernel_size=3,padding=(3 - 1) // 2,groups=dim_inner
            # in_channels=dim_inner, out_channels=dim_inner, kernel_size=1,groups=dim_inner
        )
        # self.forward_conv2d = nn.Conv2d(
        #     in_channels=dim_inner,
        #     out_channels=dim_inner,
        #     groups=dim_inner,
        #     bias=True,
        #     kernel_size=3,
        #     padding=(3 - 1) // 2,
        # )

        # self.backward_conv1d = nn.Conv1d(
        #     in_channels=dim, out_channels=dim, kernel_size=1
        # )
        self.outnorm = nn.LayerNorm(dim_inner)
        self.innorm = nn.LayerNorm(dim)
        self.activation = nn.SiLU()
        # self.ssm = SSM(dim, dt_rank, dim_inner, d_state)
        self.ssm = SSM(dim_inner, dt_rank, dim_inner, d_state)

        # Linear layer for z and x
        # self.proj = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim_inner)
        self.in_proj = nn.Linear(dim, 2 * dim_inner)

        # Softplus
        self.softplus = nn.Softplus()

        self.out_proj = nn.Linear(dim_inner, dim)

    def forward(self, x: torch.Tensor):
        # x is of shape [batch_size, seq_len, dim]
        b, s, d = x.shape

        # Skip connection
        skip = x

        # Normalization
        x = self.innorm(x)

        # # # Split x into x1 and x2 with linears
        # z1 = self.proj(x)
        # x1 = self.proj(x)

        xz = self.in_proj(x)  # (B, L, 2*ED)
        x1, z1 = xz.chunk(2, dim=-1)  # (B, L, ED), (B, L, ED)

        # forward con1d
        x1_rearranged = rearrange(x1, "b s d -> b d s")
        # x1_rearranged = rearrange(x1, "b s d -> d b s")
        forward_conv_output = self.forward_conv1d(x1_rearranged)
        # forward_conv_output = self.forward_conv2d(x1_rearranged)
        forward_conv_output = rearrange(forward_conv_output, "b d s -> b s d")
        # forward_conv_output = rearrange(forward_conv_output, "d b s -> b s d")

        x1_ssm = self.ssm(forward_conv_output)
        x1_ssm = self.outnorm(x1_ssm)

        # # backward conv x2
        # x2_rearranged = rearrange(x1, "b s d -> b d s")
        # x2 = self.backward_conv1d(x2_rearranged)
        # x2 = rearrange(x2, "b d s -> b s d")
        #
        # # Backward ssm
        # x2 = self.ssm(x2)

        # Activation
        z = self.activation(z1)

        # # matmul with z + backward ssm
        # x2 = x2 @ z

        # Matmul with z and x1
        # x1 = x1_ssm @ z
        x1 = x1_ssm* z

        # # Add both matmuls
        # x = x1 + x2
        output = self.out_proj(x1)

        # Add skip connection
        return output + skip
        # return output