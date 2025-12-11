from typing import Optional
import torch
from fairseq.utils import get_activation_fn
from fairseq.modules import LayerNorm

class ConvolutionalSpatialGatingUnit(torch.nn.Module):
    """Convolutional Spatial Gating Unit (CSGU)"""

    def __init__(
        self,
        input_feat: int,
        depthwise_conv_kernel_size: int = 31,
        dropout: float = 0.0,
        use_linear_after_conv: bool = False,
        activation_fn: str = "identity",
    ):
        super().__init__()

        self.input_feat = input_feat
        self.use_linear_after_conv = use_linear_after_conv

        if self.input_feat % 2 != 0:
            raise ValueError("Input size must be divisible by 2!")
        channels = input_feat // 2
        if activation_fn == "identity":
            self.activation = torch.nn.Identity()
        else:
            self.activation = get_activation_fn(activation_fn)(channels)
        self.norm = LayerNorm(channels)

        self.conv = torch.nn.Conv1d(
            channels,
            channels,
            depthwise_conv_kernel_size,
            1,
            (depthwise_conv_kernel_size - 1) // 2,
            groups=channels,
            )
        if self.use_linear_after_conv:
            self.linear = torch.nn.Linear(channels, channels)
            torch.nn.init.normal_(self.linear.weight, std=1e-6)
            torch.nn.init.ones_(self.linear.bias)
        torch.nn.init.normal_(self.conv.weight, std=1e-6)
        torch.nn.init.ones_(self.conv.bias)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):

        x1, x2 = x.chunk(2, dim=-1)

        x2 = self.norm(x2)
        x2 = x2.transpose(1, 2)        # (B, T, C) → (B, C, T)
        x2 = self.conv(x2)
        x2 = x2.transpose(1, 2)        # (B, C, T) → (B, T, C)
        if self.use_linear_after_conv:
            x2 = self.linear(x2)
        x2 = self.activation(x2)

        return self.dropout(x2 * x1)

class ConvolutionModule(torch.nn.Module):
    """Convolution block used in the branchformer block"""

    def __init__(
        self,
        input_feat: int,
        embed_dim: int = 3072,
        depthwise_conv_kernel_size: int = 31,
        dropout: float = 0.0,
        activation_fn: str = "gelu",
        gate_activation_fn: str = "identity",
        use_linear_after_conv: bool = False,
    ):
        super().__init__()

        self.pre_channel_proj = torch.nn.Linear(input_feat, embed_dim)
        self.post_channel_proj = torch.nn.Linear(embed_dim // 2, input_feat)
        try:
            self.activation_fn = get_activation_fn(activation_fn)(embed_dim)
        except AttributeError:
            self.activation_fn = get_activation_fn(activation_fn)
        self.csgu = ConvolutionalSpatialGatingUnit(
            input_feat=embed_dim,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            use_linear_after_conv=use_linear_after_conv,
            activation_fn=gate_activation_fn,
        )

    def forward(self, x):

        x = self.activation_fn(self.pre_channel_proj(x))  # (B, T, D)
        x = self.csgu(x)  # (B, T, D//2)
        x = self.post_channel_proj(x)  # (B, T, D)

        return x