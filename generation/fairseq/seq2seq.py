from typing import Optional, Tuple, Union, Any, Dict
import copy

import torch
from torch import Tensor
import fairseq

from flambe.nn import Module
from flambe.learn.utils import select_device

from generation.utils import pad_and_cat, pad_to_len


class PretrainedFairseqSeq2Seq(Module):

    def __init__(self,
                 alias: str,
                 **kwargs) -> None:
        super().__init__()

        try:
            interface = torch.hub.load('pytorch/fairseq', alias,
                                        **kwargs)
        except:
            interface = torch.hub.load('pytorch/fairseq', alias,
                                        **kwargs)

        self._model = interface.models[0]
        self.src_padding_idx = self._model.encoder.padding_idx
        self.tgt_padding_idx = self._model.decoder.padding_idx

    def forward(self,
                src: Tensor,
                tgt_context: Tensor,
                tgt_words: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        preds, _ = self._model(src_tokens=src, src_lengths=None,
                               prev_output_tokens=tgt_context)
        if tgt_words is not None:
            mask = (tgt_words != self.tgt_padding_idx)
            return preds[mask], tgt_words[mask]
        else:
            return preds


class PretrainedFairseqSeq2SeqTranslator(Module):

    def __init__(self,
                 alias: str,
                 beam_size: int,
                 max_seq_len: int,
                 tgt_sos_idx: int = 2,
                 tgt_eos_idx: int = 3,
                 device: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__()

        interface = torch.hub.load('pytorch/fairseq', alias, **kwargs)

        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.tgt_sos_idx = tgt_sos_idx
        self.tgt_eos_idx = tgt_eos_idx

        self.interface = interface
        args = copy.copy(self.interface.args)
        args.beam = beam_size
        args.max_len_b = max_seq_len - 1 # one extra token for <SOS>
        self.generator = self.interface.task.build_generator(args)

        self.src_pad_idx = interface.models[0].encoder.padding_idx
        self.tgt_pad_idx = interface.models[0].decoder.padding_idx

        self.device = select_device(device)

    def forward(self,
                src: Tensor,
                tgt: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        src_sample = self._build_fairseq_sample(src)
        out = self.interface.task.inference_step(self.generator,
                                                 self.interface.models,
                                                 src_sample)
        preds = []
        for x in out:
            preds.append(x[0]['tokens'].unsqueeze(0).cpu())
        pred = pad_and_cat(preds, pad_idx=self.tgt_pad_idx, dim=0)

        sos = torch.tensor([[self.tgt_sos_idx]] * pred.size(0))
        sos = sos
        pred = torch.cat([sos, pred], dim=1)
        pred = pad_to_len(pred, len=self.max_seq_len, pad_idx=self.tgt_pad_idx, dim=1)

        if tgt is not None:
            tgt = pad_to_len(tgt.cpu(), len=self.max_seq_len, pad_idx=self.tgt_pad_idx, dim=1)

            return pred, tgt
        else:
            return pred

    def _build_fairseq_sample(self, src: Tensor) -> Dict[str, Any]:
        lengths = []
        for x in src:
            lengths.append(torch.sum(x != self.tgt_pad_idx).item())
        lengths = torch.tensor(lengths)

        batch_size = src.size(0)
        sample = {
            'id': torch.arange(batch_size),
            'nsentences': batch_size,
            'ntokens': lengths.sum().item(),
            'net_input': {
                'src_tokens': src.to(self.device),
                'src_lengths': lengths.to(self.device)
            },
            'target': None
        }
        return sample
