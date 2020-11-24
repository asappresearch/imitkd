from typing import Optional

import torch
from torch import Tensor
from torch.distributions import Categorical

from generation.translation.translator import Translator
from generation.modules.seq2seq import Seq2Seq


class GreedyTranslator(Translator):

    def __init__(self,
                 model: Optional[Seq2Seq] = None,
                 device: Optional[str] = None,
                 tgt_sos_idx: int = 2,
                 tgt_eos_idx: int = 3,
                 max_seq_len: int = 50,
                 max_pad_len: Optional[int] = None,
                 sample: bool = False,
                 sample_top_k: Optional[int] = None):

        self.sample = sample
        self.sample_top_k = sample_top_k

        super().__init__(model, device, tgt_sos_idx, tgt_eos_idx, max_seq_len, max_pad_len)

    def forward(self, src: Tensor, tgt: Optional[Tensor] = None):
        with torch.no_grad():
            batch_size = src.size(0)
            start = torch.tensor(self.tgt_sos_idx).repeat((batch_size, 1))
            start = start.to(self.device)
            beam = [start]
            beam_finished = (start != start) # default false

            memory, src_padding_mask = self.model.encode(src)
            tgt_context = start

            for _ in range(self.max_seq_len - 1):
                output = self.model.decode(tgt_context, memory,
                                           src_padding_mask)

                if self.rnn_decode:
                    tgt_encoding, state = output
                    memory = (memory[0], state)
                    tgt_logits = self.model.apply_output_layer(tgt_encoding)
                    if self.sample:
                        k = self.sample_top_k
                        if k is not None:
                            topk, indices = torch.topk(tgt_logits,
                                                       k, dim=-1)
                            topk = torch.softmax(topk, dim=-1)
                            tgt_probs = torch.zeros_like(tgt_logits)
                            tgt_probs.scatter_(-1, indices, topk)
                            tokens = Categorical(probs=tgt_probs).sample()
                        else:
                            tokens = Categorical(logits=tgt_logits).sample()
                        # print('I am sampling!')
                    else:
                        _, tokens = tgt_logits.max(dim=2)
                    tgt_context = tokens

                else:
                    if isinstance(output, tuple) or isinstance(output, list):
                        tgt_encoding, state = output[:2]
                        tgt_encoding = tgt_encoding[:, -1:, :]
                    else:
                        tgt_encoding = output[:, -1:, :]
                    tgt_logits = self.model.apply_output_layer(tgt_encoding)
                    if self.sample:
                        k = self.sample_top_k
                        if k is not None:
                            topk, indices = torch.topk(tgt_logits,
                                                       k, dim=-1)
                            topk = torch.softmax(topk, dim=-1)
                            tgt_probs = torch.zeros_like(tgt_logits)
                            tgt_probs.scatter_(-1, indices, topk)
                            tokens = Categorical(probs=tgt_probs).sample()
                        else:
                            tokens = Categorical(logits=tgt_logits).sample()
                        # print('I am sampling!')
                    else:
                        _, tokens = tgt_logits.max(dim=2)
                    tgt_context = torch.cat(beam + [tokens], dim=1)

                tokens[beam_finished] = self.tgt_pad_idx
                beam.append(tokens)
                beam_finished |= (tokens == self.tgt_eos_idx)

                if torch.all(beam_finished).item():
                    pad = torch.tensor(self.tgt_pad_idx)
                    diff = self.max_seq_len - len(beam)
                    extra = pad.repeat((batch_size, 1)).to(self.device)
                    beam = beam + [extra] * diff
                    break

            pred = torch.cat(beam, dim=1)
            if tgt is not None:
                tgt = self._standardize_seq_len(tgt, self.max_seq_len)
                return pred, tgt
            else:
                return pred


class GreedyTranslatorWithSeed(Translator):

    def __init__(self,
                 model: Optional[Seq2Seq] = None,
                 device: Optional[str] = None,
                 tgt_sos_idx: int = 2,
                 tgt_eos_idx: int = 3,
                 max_seq_len: int = 50):

        super().__init__(model, device, tgt_sos_idx, tgt_eos_idx, max_seq_len)

    def forward(self,
                src: Tensor,
                seed: Tensor,
                tgt: Optional[Tensor] = None):
        with torch.no_grad():
            batch_size = src.size(0)
            start = torch.tensor(self.tgt_sos_idx).repeat((batch_size, 1))
            start = start.to(self.device)
            beam = [seed]
            beam_finished = torch.tensor([[self.tgt_eos_idx in x] for x in seed])

            memory, src_padding_mask = self.model.encode(src)
            tgt_context = seed

            for _ in range(self.max_seq_len - seed.size(1)):
                output = self.model.decode(tgt_context, memory,
                                           src_padding_mask)

                if isinstance(output, tuple) or isinstance(output, list):
                    tgt_encoding, state = output[:2]
                    tgt_encoding = tgt_encoding[:, -1:, :]
                else:
                    tgt_encoding = output[:, -1:, :]
                tgt_logits = self.model.apply_output_layer(tgt_encoding)
                _, tokens = tgt_logits.max(dim=2)
                tgt_context = torch.cat(beam + [tokens], dim=1)

                tokens[beam_finished] = self.tgt_pad_idx
                beam.append(tokens)
                beam_finished |= (tokens == self.tgt_eos_idx)

                if torch.all(beam_finished).item():
                    pad = torch.tensor(self.tgt_pad_idx)
                    diff = self.max_seq_len - len(beam) - seed.size(1) + 1
                    extra = pad.repeat((batch_size, 1)).to(self.device)
                    beam = beam + [extra] * diff
                    break

            pred = torch.cat(beam, dim=1)
            if tgt is not None:
                tgt = self._standardize_seq_len(tgt, self.max_seq_len)
                return pred, tgt
            else:
                return pred
