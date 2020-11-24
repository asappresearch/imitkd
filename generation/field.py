from typing import Tuple, Optional
from collections import OrderedDict as odict
import torch
import numpy as np

from flambe import field
from flambe.tokenizer import Tokenizer, WordTokenizer

from flambe.compile.registrable import registrable_factory


class TextField(field.TextField):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def setup(self, *data: np.ndarray) -> None:
        pass

    @registrable_factory
    @classmethod
    def from_textfield(cls,
                       textfield: field.TextField,
                       **kwargs) -> field.TextField:
        instance = cls(**kwargs)
        instance.load_state(textfield.get_state())
        return instance


class LMField(TextField):

    def __init__(self,
                 **kwargs) -> None:
        super().__init__(**kwargs)

    def process(self, example: str) -> Tuple[torch.Tensor, ...]:  # type: ignore00
        ret = super().process(example)
        return ret[:-1], ret[1:]
