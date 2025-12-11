# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional

import torch

from fairseq.modules import (
    ESPNETMultiHeadedAttention,
    LayerNorm,
    MultiheadAttention,
    RelPositionMultiHeadedAttention,
    RotaryPositionMultiHeadedAttention,
)
from fairseq.utils import get_activation_fn
from .branchformer_conv_modules import ConvolutionalSpatialGatingUnit, ConvolutionModule

class FeedForwardModule(torch.nn.Module):
    """Positionwise feed forward layer used in e-branchformer"""

    def __init__(
        self,
        input_feat,
        hidden_units,
        dropout,
        activation_fn="swish",
        bias=True,
    ):
        """
        Args:
            input_feat: Input feature dimension
            hidden_units: Hidden unit dimension
            dropout: dropout value
            activation_fn: Name of activation function
            bias: If linear layers should have bias
        """

        super(FeedForwardModule, self).__init__()
        self.w_1 = torch.nn.Linear(input_feat, hidden_units, bias=bias)
        self.w_2 = torch.nn.Linear(hidden_units, input_feat, bias=bias)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        try:
            self.activation_fn = get_activation_fn(activation_fn)(hidden_units)
        except AttributeError:
            self.activation_fn = get_activation_fn(activation_fn)

    def forward(self, x):
        """
        Args:
            x: Input Tensor of shape  T X B X C
        Returns:
            Tensor of shape T X B X C
        """
        x = self.w_1(x)
        x = self.activation_fn(x)
        x = self.dropout1(x)
        x = self.w_2(x)
        return self.dropout2(x)

class EBranchformerEncoderLayer(torch.nn.Module):
    """Branchformer block based on “Branchformer: Parallel MLP-Attention Architectures to Capture
    Local and Global Context for Speech Recognition and Understanding,”"""

    def __init__(
        self,
        embed_dim,
        attention_heads,
        dropout,
        use_fp16,
        depthwise_conv_kernel_size=31,
        depthwise_merge_conv_kernel_size=3,
        activation_fn="gelu",
        attn_type="espnet",
        pos_enc_type="rel_pos",
        gate_activation_fn="identity",
        use_linear_after_conv=False,
        csgu_linear_units=3072,
        use_macaron_ffn=True,
        ffn_embed_dim=3072,
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            attention_heads: Number of attention heads in MHA
            dropout: dropout value
            depthwise_conv_kernel_size: Size of kernel in depthwise conv layer in convolution module
            activation_fn: Activation function name to use in convulation block and feed forward block
            attn_type: MHA implementation from ESPNET vs fairseq
            pos_enc_type: Positional encoding type - abs, rope, rel_pos
        """
        self.pos_enc_type = pos_enc_type
        self.use_macaron_ffn = use_macaron_ffn
        super(EBranchformerEncoderLayer, self).__init__()

        self.ff_scale = 1.0
        if self.use_macaron_ffn:
            self.ff_scale = 0.5
            self.norm_ff_macaron = LayerNorm(embed_dim)
            self.ffn1 = FeedForwardModule(
                input_feat=embed_dim,
                hidden_units=ffn_embed_dim,
                dropout=dropout,
                activation_fn="swish",
            )

        self.self_attn_layer_norm = LayerNorm(embed_dim, export=False)
        self.dropout = torch.nn.Dropout(dropout)
        if attn_type == "espnet":
            if self.pos_enc_type == "rel_pos":
                self.self_attn = RelPositionMultiHeadedAttention(
                    embed_dim,
                    attention_heads,
                    dropout=dropout,
                )
            elif self.pos_enc_type == "rope":
                self.self_attn = RotaryPositionMultiHeadedAttention(
                    embed_dim, attention_heads, dropout=dropout, precision=use_fp16
                )
            elif self.pos_enc_type == "abs":
                self.self_attn = ESPNETMultiHeadedAttention(
                    embed_dim,
                    attention_heads,
                    dropout=dropout,
                )
            else:
                raise Exception(f"Unsupported attention type {self.pos_enc_type}")
        else:
            # Default to fairseq MHA
            self.self_attn = MultiheadAttention(
                embed_dim,
                attention_heads,
                dropout=dropout,
            )

        self.conv_module = ConvolutionModule(
            input_feat=embed_dim,
            embed_dim=csgu_linear_units,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            activation_fn=activation_fn,
            gate_activation_fn=gate_activation_fn,
            use_linear_after_conv=use_linear_after_conv,
        )

        self.depthwise_conv_fusion = torch.nn.Conv1d(
            embed_dim + embed_dim,
            embed_dim + embed_dim,
            kernel_size=depthwise_merge_conv_kernel_size,
            stride=1,
            padding=(depthwise_merge_conv_kernel_size - 1) // 2,
            groups=embed_dim + embed_dim,
            bias=True,
        )
        torch.nn.init.normal_(self.depthwise_conv_fusion.weight, std=1e-6)
        # torch.nn.init.ones_(self.depthwise_conv_fusion.bias)
        # torch.nn.init.xavier_uniform_(self.depthwise_conv_fusion.weight)
        torch.nn.init.zeros_(self.depthwise_conv_fusion.bias)

        self.merge_proj = torch.nn.Linear(embed_dim * 2, embed_dim)
        self.conv_layer_norm = LayerNorm(embed_dim)
        
        # if self.use_macaron_ffn:
        self.norm_ff = LayerNorm(embed_dim)
        self.ffn2 = FeedForwardModule(
            input_feat=embed_dim,
            hidden_units=ffn_embed_dim,
            dropout=dropout,
            activation_fn="swish",
        )

        self.final_layer_norm = LayerNorm(embed_dim, export=False)

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[torch.Tensor],
        position_emb: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: Tensor of shape T X B X C
            encoder_padding_mask: Optional mask tensor
            positions:
        Returns:
            Tensor of shape T X B X C
        """

        if self.use_macaron_ffn:
            residual = x
            x = self.norm_ff_macaron(x)
            x = self.ffn1(x)
            x = x * self.ff_scale + residual

        x1 = x
        x2 = x
    
        x1 = self.self_attn_layer_norm(x1)
        if self.pos_enc_type == "rel_pos":
            x1, attn = self.self_attn(
                query=x1,
                key=x1,
                value=x1,
                key_padding_mask=encoder_padding_mask,
                pos_emb=position_emb,
                need_weights=False,
            )
        else:
            x1, attn = self.self_attn(
                query=x1,
                key=x1,
                value=x1,
                key_padding_mask=encoder_padding_mask,
                need_weights=False,
            )
        x1 = self.dropout(x1)

        x2 = self.conv_layer_norm(x2)
        # TBC to BTC
        x2 = x2.transpose(0, 1)
        x2 = self.conv_module(x2)
        # BTC to TBC
        x2 = x2.transpose(0, 1)
        x2 = self.dropout(x2)


        # concat_x = torch.cat([x1, x2], dim=-1)  # Shape: T x B x (2 * embed_dim)
    
        # Apply depthwise_conv_fusion
        # concat_x_btc = concat_x.transpose(0, 1)  # TBC to BTC: B x T x (2 * embed_dim)
        # concat_x_btc = concat_x_btc.transpose(1, 2)  # B x T x (2 * embed_dim) -> B x (2 * embed_dim) x T
        # fused_x = self.depthwise_conv_fusion(concat_x_btc)  # Apply depthwise conv
        # fused_x = fused_x.transpose(1, 2).transpose(0, 1)  # B x (2 * embed_dim) x T -> T x B x (2 * embed_dim)
        # x = x + self.dropout(self.merge_proj(x_concat + x_temp))
        
        x_concat = torch.cat([x1, x2], dim=-1)

        # x_temp = x_concat.transpose(1, 2)
        # x_temp = self.depthwise_conv_fusion(x_temp)
        # x_temp = x_temp.transpose(1, 2)


        x_temp = x_concat.transpose(0, 1).transpose(1, 2)
        x_temp = self.depthwise_conv_fusion(x_temp)
        x_temp = x_temp.transpose(1, 2).transpose(0, 1)

        x = x + self.dropout(self.merge_proj(x_concat + x_temp))
        
        # x = x + self.dropout(self.merge_proj(torch.cat([x1, x2], dim=-1)))


        layer_result = x
    
        # if self.use_macaron_ffn:
        residual = x
        x = self.norm_ff(x)
        x = self.ffn2(x)
        x = x * self.ff_scale + residual
 
        x = self.final_layer_norm(x)
        return x, (attn, layer_result)


class EBranchformerWav2Vec2EncoderLayer(EBranchformerEncoderLayer):
    """Encoder layer for Wav2vec2 encoder"""

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
        position_emb=None,
    ):
        return super().forward(x, self_attn_padding_mask, position_emb)
