from typing import Optional, Union, List, Dict, Any, Tuple

import torch
from transformers import BertTokenizer

from flambe.field import Field


class PretrainedBartTextField(Field):
    """Field intergation of the transformers library.

    Instantiate this object using any alias available in the
    `transformers` library. More information can be found here:

    https://huggingface.co/transformers/

    """

    def __init__(self, max_seq_len: Optional[int] = None) -> None:
        """Initialize a pretrained tokenizer.

        Parameters
        ----------
        alias: str
            Alias of a pretrained tokenizer.
        cache_dir: str, optional
            A directory where to cache the downloaded vocabularies.
        max_len_truncate: int, default = 500
            Truncates the length of the tokenized sequence.
            Because several pretrained models crash when this is
            > 500, it defaults to 500
        add_special_tokens: bool, optional
            Add the special tokens to the inputs. Default ``True``.

        """
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab = self._tokenizer.get_vocab()
        self.max_seq_len = max_seq_len

        self.pad = self._tokenizer.pad_token
        self.unk = self._tokenizer.unk_token
        self.eos = self._tokenizer.eos_token
        self.sos = self._tokenizer.bos_token

        self.pad_idx = self._tokenizer.pad_token_id
        self.eos_idx = self._tokenizer.eos_token_id
        self.unk_idx = self._tokenizer.unk_token_id
        self.sos_idx = self._tokenizer.bos_token_id

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary length.

        Returns
        -------
        int
            The length of the vocabulary

        """
        return len(self._tokenizer)

    def process(self, example: str) -> List[torch.Tensor]:
        """Process an example, and create a Tensor.

        Parameters
        ----------
        example: str
            The example to process, as a single string

        Returns
        -------
        torch.Tensor
            The processed example, tokenized and numericalized

        """
        return self._tokenizer.batch_encode_plus([example], return_tensors="pt", output_past=True, max_length=self.max_seq_len)['input_ids'][0]


class PretrainedBartLMField(PretrainedBartTextField):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def process(self, example: str) -> Tuple[torch.Tensor, ...]:  # type: ignore00
        """Process an example and create 2 Tensors.

        Parameters
        ----------
        example: str
            The example to process, as a single string

        Returns
        -------
        Tuple[torch.Tensor, ...]
            The processed example, tokenized and numericalized

        """
        ret = super().process(example)
        return ret[:-1], ret[1:]


class PretrainedBertTextField(Field):

    def __init__(self, max_seq_len: Optional[int] = None) -> None:
        """Initialize a pretrained tokenizer.

        Parameters
        ----------
        alias: str
            Alias of a pretrained tokenizer.
        cache_dir: str, optional
            A directory where to cache the downloaded vocabularies.
        max_len_truncate: int, default = 500
            Truncates the length of the tokenized sequence.
            Because several pretrained models crash when this is
            > 500, it defaults to 500
        add_special_tokens: bool, optional
            Add the special tokens to the inputs. Default ``True``.

        """
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab = self._tokenizer.get_vocab()
        self.max_seq_len = max_seq_len

        self.pad = self._tokenizer.pad_token
        self.unk = self._tokenizer.unk_token
        self.eos = self._tokenizer.sep_token
        self.sos = self._tokenizer.cls_token

        self.pad_idx = self._tokenizer.pad_token_id
        self.eos_idx = self._tokenizer.sep_token_id
        self.unk_idx = self._tokenizer.unk_token_id
        self.sos_idx = self._tokenizer.cls_token_id

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary length.

        Returns
        -------
        int
            The length of the vocabulary

        """
        return len(self._tokenizer)

    def process(self, example: str) -> List[torch.Tensor]:
        """Process an example, and create a Tensor.

        Parameters
        ----------
        example: str
            The example to process, as a single string

        Returns
        -------
        torch.Tensor
            The processed example, tokenized and numericalized

        """
        return torch.tensor(self._tokenizer.encode(example, max_length=self.max_seq_len))


class PretrainedBertLMField(PretrainedBertTextField):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def process(self, example: str) -> Tuple[torch.Tensor, ...]:  # type: ignore00
        """Process an example and create 2 Tensors.

        Parameters
        ----------
        example: str
            The example to process, as a single string

        Returns
        -------
        Tuple[torch.Tensor, ...]
            The processed example, tokenized and numericalized

        """
        ret = super().process(example)
        return ret[:-1], ret[1:]
