from typing import Optional, Union, Tuple

import torch
from torch import Tensor

from generation.translation.beam_search import BeamSearchTranslator
from generation.metric import Bleu
from generation.modules.seq2seq import Seq2Seq
from generation.utils import topk_2d

class SeqInterTranslator(BeamSearchTranslator):

    def __init__(self,
                 metric: Bleu,
                 model: Optional[Seq2Seq] = None,
                 device: Optional[str] = None,
                 tgt_sos_idx: int = 2,
                 tgt_eos_idx: int = 3,
                 max_seq_len: int = 50,
                 max_pad_len: Optional[int] = None,
                 beam_size: int = 5,
                 len_norm: bool = False,
                 len_pen: float = 1.,
                 block_ngram_repeats: Optional[int] = None):

        super().__init__(
            model, device, tgt_sos_idx, tgt_eos_idx, max_seq_len, max_pad_len, beam_size, len_norm, len_pen, block_ngram_repeats
        )
        self.metric = metric

    def forward(self, src: Tensor, tgt: Tensor):

        with torch.no_grad():
            # Encode source
            memory, src_padding_mask = self.model.encode(src)

            # Search over decoder for beams
            if self.rnn_decode:
                beams, log_probs = self._rnn_search(memory,
                                                    src_padding_mask)
            else:
                beams, log_probs = self._search(memory,
                                                src_padding_mask)

            # Compute Bleu score between all candidate-tgt pairs
            B, K = log_probs.size()
            scores = torch.zeros((B, K))
            for b in range(B):
                for k in range(K):
                    scores[b, k] = self.metric.compute(beams[b, k].view(1, -1), tgt[b].view(1, -1))

            _, idx = scores.max(dim=-1)
            pred = beams[torch.arange(B), idx]

            return pred
