from typing import Optional, Union, Tuple

import torch
from torch import Tensor

from generation.translation.translator import Translator
from generation.modules.seq2seq import Seq2Seq
from generation.utils import topk_2d

class BeamSearchTranslator(Translator):

    def __init__(self,
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

        super().__init__(model, device, tgt_sos_idx, tgt_eos_idx, max_seq_len, max_pad_len)

        self.beam_size = beam_size
        self.len_norm = len_norm
        self.len_pen = len_pen
        self.block_ngram_repeats = block_ngram_repeats

        if block_ngram_repeats is not None:
            assert block_ngram_repeats >= 2

    def forward(self, src: Tensor, tgt: Optional[Tensor] = None):

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

        if self.len_norm:
            beam_lengths = (beams != self.tgt_pad_idx).sum(dim=-1)
            scores = log_probs / (beam_lengths ** self.len_pen)
            _, idx = scores.max(dim=-1)
            pred = beams[torch.arange(beams.size(0)), idx]
        else:
            # Just take the max log likelihood
            pred = beams[:, 0, :]

        if self.max_pad_len is not None:
            self._standardize_seq_len(tgt, self.max_pad_len)

        if tgt is not None:
            if self.max_pad_len is None:
                tgt = self._standardize_seq_len(tgt, self.max_seq_len)
            else:
                tgt = self._standardize_seq_len(tgt, self.max_pad_len)
            return pred, tgt
        else:
            return pred

    def _search(self,
                memory: Union[Tensor, Tuple[Tensor]],
                src_padding_mask: Tensor):

        # Define dimensions
        batch_size = src_padding_mask.size(0)
        B, K, T = batch_size, self.beam_size, self.max_seq_len

        # Maintain n gram list for blocking
        N = self.block_ngram_repeats
        ngrams = torch.LongTensor([]).to(self.device)

        # Take first step in beam search
        start = torch.tensor(self.tgt_sos_idx).repeat((B, 1))
        start = start.to(self.device)

        tgt_encoding = self.model.decode(start, memory,
                                      src_padding_mask)
        tgt_encoding = tgt_encoding[0] if isinstance(tgt_encoding, tuple) else tgt_encoding
        tgt_logits = self.model.apply_output_layer(tgt_encoding)
        log_probs = torch.log_softmax(tgt_logits, dim=-1).squeeze(1)
        V = log_probs.size(-1)
        log_probs, tokens = torch.topk(log_probs, k=K,
                                       sorted=True, dim=-1) # B x K
        start = torch.tensor(self.tgt_sos_idx).repeat((B, K))
        start = start.to(self.device)
        beams = torch.stack([start, tokens], dim=2) # B x K x 2
        beams_finished = (tokens == self.tgt_eos_idx) # B x K

        if isinstance(memory, tuple):
            memory_ = tuple([m.repeat_interleave(K, dim=0) for m in memory])
        else:
            memory_ = memory.repeat_interleave(K, dim=0)

        src_padding_mask_ = src_padding_mask.repeat_interleave(K, dim=0)

        pad_log_probs = torch.tensor(-float('inf')).repeat(V)
        pad_log_probs[self.tgt_pad_idx] = 0
        pad_log_probs = pad_log_probs.to(self.device)

        for _ in range(T - 2):

            # Batch computation across candidate beams
            beams_ = beams.view(B*K, -1)
            tgt_encoding_ = self.model.decode(beams_, memory_,
                                              src_padding_mask_)
            tgt_encoding_ = tgt_encoding_[0] if isinstance(tgt_encoding_, tuple) else tgt_encoding_
            tgt_encoding_ = tgt_encoding_[:, -1:, :] # (B*K) x 1 x V
            tgt_logits_ = self.model.apply_output_layer(tgt_encoding_)

            tgt_logits_ = tgt_logits_.view(B, K, -1) # B x K x V
            next_log_probs = torch.log_softmax(tgt_logits_, dim=-1)

            # Block n grams
            if len(ngrams) != 0:
                ngrams_prefix = ngrams[:, :, :, -N:-1] # B x K x (T-N) x (N-1)
                ngrams_last = ngrams[:, :, :, -1] # B x K x (T-N)

                curr_prefix = beams[:, :, -(N-1):].unsqueeze(2)
                ngrams_match = (curr_prefix == ngrams_prefix).all(dim=3)
                B_idx, K_idx, T_idx = torch.where(ngrams_match)
                block_idx = ngrams_last[B_idx, K_idx, T_idx]
                next_log_probs[B_idx, K_idx, block_idx] = -float('inf')

            # Enforce pad if beam has finished
            next_log_probs[beams_finished] = pad_log_probs
            log_probs = log_probs.unsqueeze(2) + next_log_probs

            log_probs, indices1, indices2 = topk_2d(log_probs, K) # B x K
            rows = torch.arange(B).unsqueeze(1)
            beams = beams[rows, indices1] # B x K x (T-1)
            tokens = indices2 # B x K
            beams = torch.cat([beams, tokens.unsqueeze(2)], dim=-1)
            # B x K x T
            beams_finished = beams_finished[rows, indices1] | (tokens == self.tgt_eos_idx)

            # Update ngrams
            if N is not None and beams.size(2) >= N:
                # Add new ngrams
                new_ngrams = beams[:, :, -N:].unsqueeze(2) # B x K x 1 x N
                if len(ngrams) != 0:
                    ngrams = ngrams[rows, indices1]
                ngrams = torch.cat([ngrams, new_ngrams], dim=2) # B x K x (T-N) x N

            if torch.all(beams_finished).item():
                pad = torch.tensor(self.tgt_pad_idx)
                diff = self.max_seq_len - beams.size(-1)
                extra = pad.repeat((B, K, diff)).to(self.device)
                beams = torch.cat([beams, extra], dim=-1)
                break

        return beams, log_probs

    def _rnn_search(self,
                    memory: Tuple[Tensor],
                    src_padding_mask: Tensor):

        # Define dimensions
        batch_size = src_padding_mask.size(0)
        B, K, T = batch_size, self.beam_size, self.max_seq_len

        # Maintain n gram list for blocking
        N = self.block_ngram_repeats
        ngrams = torch.LongTensor([]).to(self.device)

        # Take first step in beam search
        start = torch.tensor(self.tgt_sos_idx).repeat((B, 1))
        start = start.to(self.device)

        tgt_encoding, state = self.model.decode(start, memory,
                                                src_padding_mask)
        tgt_logits = self.model.apply_output_layer(tgt_encoding)
            # B x 1 x V
        log_probs = torch.log_softmax(tgt_logits, dim=-1).squeeze(1)
            # B x V
        V = log_probs.size(-1)
        log_probs, tokens = torch.topk(log_probs, k=K,
                                       sorted=True, dim=-1) # B x K

        state = state.transpose(0, 1).unsqueeze(1)
            # B x 1 x L x H
        states = state.repeat_interleave(K, dim=1)
            # B x K x L x H
        _, _, L, H = states.size()
        start = torch.tensor(self.tgt_sos_idx).repeat((B, K))
        start = start.to(self.device)
        beams = torch.stack([start, tokens], dim=2) # B x K x 2
        beams_finished = (tokens == self.tgt_eos_idx)

        src_encoding, _ = memory
        src_encoding_ = src_encoding.repeat_interleave(K, dim=0)
        src_padding_mask_ = src_padding_mask.repeat_interleave(K, dim=0)

        pad_log_probs = torch.tensor(-float('inf')).repeat(V)
        pad_log_probs[self.tgt_pad_idx] = 0
        pad_log_probs = pad_log_probs.to(self.device)

        for _ in range(T - 2):

            # Batch computation across candidate beams
            tokens_ = tokens.view(B*K, 1)
            states_ = states.view(B*K, L, H)
            states_ = states_.transpose(0, 1).contiguous() # L x (B*K) x H
            memory_ = (src_encoding_, states_)
            out_ = self.model.decode(tokens_, memory_,
                                     src_padding_mask_)
            tgt_encoding_, states_ = out_
            tgt_logits_ = self.model.apply_output_layer(tgt_encoding_)

            tgt_logits_ = tgt_logits_.view(B, K, -1) # B x K x V
            states_ = states_.transpose(0, 1)
            states = states_.squeeze(0).view(B, K, L, H)

            next_log_probs = torch.log_softmax(tgt_logits_, dim=-1)

            # Block n grams
            if len(ngrams) != 0:
                ngrams_prefix = ngrams[:, :, :, -N:-1] # B x K x (T-N) x (N-1)
                ngrams_last = ngrams[:, :, :, -1] # B x K x (T-N)

                curr_prefix = beams[:, :, -(N-1):].unsqueeze(2)
                ngrams_match = (curr_prefix == ngrams_prefix).all(dim=3)
                B_idx, K_idx, T_idx = torch.where(ngrams_match)
                block_idx = ngrams_last[B_idx, K_idx, T_idx]
                next_log_probs[B_idx, K_idx, block_idx] = -float('inf')

            # Enforce pad if beam has finished
            next_log_probs[beams_finished] = pad_log_probs
            log_probs = log_probs.unsqueeze(2) + next_log_probs

            log_probs, indices1, indices2 = topk_2d(log_probs, K) # B x K
            rows = torch.arange(B).unsqueeze(1)
            beams = beams[rows, indices1] # B x K x (T-1)
            states = states[rows, indices1]
            tokens = indices2 # B x K
            beams = torch.cat([beams, tokens.unsqueeze(2)], dim=-1)

            beams_finished = beams_finished[rows, indices1] | (tokens == self.tgt_eos_idx)

            # Update ngrams
            if N is not None and beams.size(2) >= N:
                # Add new n_grams
                new_ngrams = beams[:, :, -N:].unsqueeze(2) # B x K x 1 x N
                if len(ngrams) != 0:
                    ngrams = ngrams[rows, indices1]
                ngrams = torch.cat([ngrams, new_ngrams], dim=2) # B x K x (T-N) x N

            if torch.all(beams_finished).item():
                pad = torch.tensor(self.tgt_pad_idx)
                diff = self.max_seq_len - beams.size(-1)
                extra = pad.repeat((B, K, diff)).to(self.device)
                beams = torch.cat([beams, extra], dim=-1)
                break

        return beams, log_probs
