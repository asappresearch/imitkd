from typing import Optional, Tuple

import torch
import fairseq

from flambe.field import Field


class PretrainedFairseqTextField(Field):

    def __init__(self,
                 alias: str,
                 use_tgt_dict: bool = False,
                 tokenizer: Optional[str] = None,
                 bpe: Optional[str] = None,
                 **kwargs) -> None:
        try:
            interface = torch.hub.load('pytorch/fairseq', alias,
                                        tokenizer=tokenizer, bpe=bpe,
                                        **kwargs)
        except:
            print('An exception occurred.')
            interface = torch.hub.load('pytorch/fairseq', alias,
                                        tokenizer=tokenizer, bpe=bpe,
                                        force_reload=True, **kwargs)

        self.tokenizer = interface.tokenizer
        self.bpe = interface.bpe

        if use_tgt_dict:
            self.dictionary = interface.tgt_dict
        else:
            self.dictionary = interface.src_dict

        self.pad = self.dictionary.pad_word
        self.unk = self.dictionary.unk_word
        self.eos = self.dictionary.eos_word

        self.pad_idx = self.dictionary.pad_index
        self.eos_idx = self.dictionary.eos_index
        self.unk_idx = self.dictionary.unk_index
        if use_tgt_dict:
            self.sos = self.eos # Fairseq sets sos = eos
            self.sos_idx = self.eos_idx
        else:
            self.sos = None
            self.sos_idx = None

        self.vocab = self.dictionary.indices

    def process(self, example: str, tokenize_normal: bool = False) -> torch.Tensor:
        if self.tokenizer is not None:
            example = self.tokenizer.encode(example)
        if self.bpe is not None:
            example = self.bpe.encode(example)
        if self.sos is not None:
            example = self.sos + ' ' + example

        if tokenize_normal:
            example = self.dictionary.encode_line(example,
                                                  add_if_not_exist=False)
            example = example.long()
            return example
        else:
            # Numericalize
            numericals = []
            for token in example.split(' '):
                if token not in self.vocab:
                    if self.unk is None or self.unk not in self.vocab:
                        raise ValueError("Encounterd out-of-vocabulary token \
                                          but the unk_token is either missing \
                                          or not defined in the vocabulary.")
                    else:
                        token = self.unk

                numerical = self.vocab[token]  # type: ignore
                numericals.append(numerical)

            # Append eos
            numericals.append(self.eos_idx)

            ret = torch.tensor(numericals, device='cpu').long()
            return ret


    @property
    def vocab_size(self) -> int:
        unique_ids = set(v for k, v in self.vocab.items())
        return len(unique_ids)


class PretrainedFairseqLMField(PretrainedFairseqTextField):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def process(self, example: str) -> Tuple[torch.Tensor, ...]:  # type: ignore00
        ret = super().process(example)
        return ret[:-1], ret[1:]
