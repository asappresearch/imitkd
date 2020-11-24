from typing import Optional, Tuple, Union, Dict, Any
from abc import abstractmethod

import torch
from torch import nn
from torch import Tensor

from flambe.compile import State
from flambe.nn.module import Module
from flambe.nn.rnn import RNNEncoder
from flambe.learn.utils import select_device

from generation.modules.seq2seq import Seq2Seq
from generation.modules.rnn import RNNDecoder


class Translator(Module):

    def __init__(self,
                 model: Optional[Seq2Seq] = None,
                 device: Optional[str] = None,
                 tgt_sos_idx: int = 2,
                 tgt_eos_idx: int = 3,
                 max_seq_len: int = 50,
                 max_pad_len: Optional[int] = None):

        super().__init__()
        self.device = select_device(device)

        self.tgt_sos_idx = tgt_sos_idx
        self.tgt_eos_idx = tgt_eos_idx

        self.max_seq_len = max_seq_len
        self.max_pad_len = max_pad_len

        self.model_passed_in = model is not None
        if self.model_passed_in:
            self.initialize(model)

    def initialize(self, model):
        self.model = model.eval()
        self.tgt_pad_idx = self.model.tgt_padding_idx
        self.rnn_decode = isinstance(self.model.decoder, RNNDecoder)

    @abstractmethod
    def forward(self, src: Tensor, tgt: Optional[Tensor] = None):
        pass

    def _standardize_seq_len(self, tgt: Tensor, len: int):
        if tgt.size(1) < len:
            tgt_pad_idx = torch.tensor(self.tgt_pad_idx)
            extra_len = len - tgt.size(1)
            extra = tgt_pad_idx.repeat((tgt.size(0), extra_len))
            extra = extra.to(self.device)
            tgt = torch.cat([tgt, extra], dim=1)
        return tgt

    def _state(self,
               state_dict: State,
               prefix: str,
               local_metadata: Dict[str, Any]) -> State:
        if not self.model_passed_in and (prefix+'model') in state_dict:
            del state_dict[prefix + 'model']
        return state_dict
