import numpy as np
import torch
import torch.nn.functional as F

from flambe.nn import Module


class DotProductAttention(Module):

    def __init__(self, dropout: float = 0):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                key_padding_mask: torch.Tensor):

        query = query.transpose(0, 1).contiguous()
        key = key.transpose(0, 1).contiguous()
        value = value.transpose(0, 1).contiguous()

        scaling = float(query.size(-1)) ** -0.5
        query = query * scaling

        attn = torch.bmm(query, key.transpose(1, 2).contiguous())

        if key_padding_mask is not None:
            attn.masked_fill_(key_padding_mask.unsqueeze(1), -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.bmm(attn, value)
        output = output.transpose(0, 1).contiguous()

        return output, attn
