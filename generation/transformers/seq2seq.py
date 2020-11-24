from typing import Optional, Union, List, Dict, Any, Tuple

import torch
from torch import Tensor
# from transformers import BartForConditionalGeneration

from flambe.nn import Module
from flambe.learn.utils import select_device

from generation.utils import pad_to_len


class PretrainedBartTranslator(Module):

    def __init__(self,
                 beam_size: int,
                 max_seq_len: int,
                 device: Optional[str] = None) -> None:
        super().__init__()

        self.beam_size = beam_size
        self.max_seq_len = max_seq_len

        self._model = BartForConditionalGeneration.from_pretrained('bart-large-cnn')

        self.tgt_sos_idx = self._model.config.bos_token_id
        self.tgt_eos_idx = self._model.config.eos_token_id
        self.tgt_pad_idx = self._model.config.pad_token_id

        self.device = select_device(device)

    def forward(self,
                src: Tensor,
                tgt: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:

        pred = self._model.generate(src, num_beams=self.beam_size, max_length=self.max_seq_len, early_stopping=True)

        if tgt is not None:
            tgt = pad_to_len(tgt.cpu(), len=self.max_seq_len, pad_idx=self.tgt_pad_idx, dim=1)

            return pred, tgt
        else:
            return pred
