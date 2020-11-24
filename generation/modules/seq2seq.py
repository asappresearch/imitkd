from typing import Optional, Tuple, Union, Dict

import torch
from torch import nn
from torch import Tensor
from torch.nn import Embedding

from flambe.nn.module import Module
from flambe.nn.rnn import RNNEncoder

from generation.modules.rnn import RNNDecoder


class Seq2Seq(Module):

    def __init__(self,
                 src_embedding: Module,
                 tgt_embedding: Module,
                 encoder: Module,
                 decoder: Module,
                 output_layer: Module,
                 dropout: float = 0,
                 embedding_dropout: float = 0,
                 src_padding_idx: int = 0,
                 tgt_padding_idx: int = 0,
                 weight_tying: Dict[str, str] = {}):

        super().__init__()
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.output_layer = output_layer

        self.dropout = nn.Dropout(dropout)
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        self.src_padding_idx = src_padding_idx
        self.tgt_padding_idx = tgt_padding_idx

        if weight_tying is not None:
            for key_path, val_path in weight_tying.items():
                key_module = self
                val_module = self
                for attr in key_path.split('.'):
                    key_module = getattr(key_module, attr)
                for attr in val_path.split('.'):
                    val_module = getattr(val_module, attr)
                assert hasattr(key_module, 'weight')
                assert hasattr(val_module, 'weight')
                key_module.weight = val_module.weight

        self.reset_parameters()

    def forward(self,
                src: Tensor,
                tgt_context: Tensor,
                tgt_words: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:

        memory, src_padding_mask = self.encode(src)
        output = self.decode(tgt_context, memory, src_padding_mask)

        if isinstance(output, tuple):
            tgt_encoding = output[0]
        else:
            tgt_encoding = output

        if tgt_words is not None:
            mask = (tgt_words != self.tgt_padding_idx)
            # Flatten to compute loss across batch and sequence
            flat_mask = mask.view(-1)

            flat_tgt_encoding = tgt_encoding.contiguous().view(-1, tgt_encoding.size(2))[flat_mask]
            flat_tgt_preds = self.apply_output_layer(flat_tgt_encoding)
            flat_tgt_words = tgt_words.contiguous().view(-1)[flat_mask]

            return flat_tgt_preds, flat_tgt_words
        else:
            tgt_preds = self.apply_output_layer(tgt_encoding)
            return tgt_preds

    def encode(self, src: Tensor):
        src_embedded = self.src_embedding(src)
        src_embedded = self.embedding_dropout(src_embedded)
        src_padding_mask = (src != self.src_padding_idx)
        memory = self.encoder(src_embedded,
                              padding_mask=src_padding_mask)
        return memory, src_padding_mask

    def decode(self,
               tgt_context: Tensor,
               memory: Union[Tensor, Tuple[Tensor, Tensor]],
               src_padding_mask: Tensor):
        tgt_embedded = self.tgt_embedding(tgt_context)
        tgt_embedded = self.embedding_dropout(tgt_embedded)
        tgt_padding_mask = (tgt_context != self.tgt_padding_idx)
        output = self.decoder(tgt_embedded, memory=memory,
                              padding_mask=tgt_padding_mask,
                              memory_key_padding_mask=src_padding_mask)
        return output

    def apply_output_layer(self, tgt_encoding: Tensor):
        tgt_encoding = self.dropout(tgt_encoding)
        tgt_preds = self.output_layer(tgt_encoding)
        return tgt_preds

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'src_embedding' in name:
                dim = param.size(1)
                param.data.normal_(mean=0, std=dim**-0.5)
                param.data[self.src_padding_idx] = 0
            if 'tgt_embedding' in name:
                dim = param.size(1)
                param.data.normal_(mean=0, std=dim**-0.5)
                param.data[self.tgt_padding_idx] = 0
