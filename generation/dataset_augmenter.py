import math
from typing import Dict, List, Optional, Any, Tuple, Iterator
import dill

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_

from generation.translation.translator import Translator
from generation.modules.seq2seq import Seq2Seq
from flambe.compile import Schema, State, Component, Link
from flambe.learn.utils import select_device
from flambe.nn import Module  # type: ignore[attr-defined]
from flambe.sampler import Sampler
from flambe.metric import Metric
from flambe.logging import log

from flambe.dataset import Dataset
from flambe.sampler import Sampler

from generation.datasets import TensorDataset
from generation.utils import tensor_to_list


class SeqKDRunner(Component):

    def __init__(self,
                 dataset: Dataset,
                 sampler: Sampler,
                 translator: Translator,
                 base_dir: str,
                 train_file: str,
                 val_file: str,
                 test_file: str,
                 others_file: str,
                 split: str = 'train',
                 use_seqinter: bool = False,
                 batch_saving: Optional[int] = None) -> None:

        self.dataset = dataset
        self.sampler = sampler
        self.device = select_device(None)
        self.translator = translator.to(self.device)

        self.base_dir = base_dir
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.others_file = others_file
        self.split = split
        self.use_seqinter = use_seqinter
        self.batch_saving = batch_saving

        self.register_attrs('base_dir', 'train_file', 'val_file', 'test_file', 'others_file')


    def _batch_to_device(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        batch = tuple(t.to(self.device) for t in batch)
        return batch

    def run(self) -> bool:

        if self.batch_saving is not None:
            batch_id = 0

        data = getattr(self.dataset, self.split)
        train_iterator = self.sampler.sample(data)
        train_data = []
        for i, batch in enumerate(train_iterator):
            print(i)
            src, tgt_context, tgt_words = self._batch_to_device(batch)
            with torch.no_grad():
                if self.use_seqinter:
                    new_tgt = self.translator(src, tgt_words)
                else:
                    new_tgt = self.translator(src)
            src = src.cpu()
            new_tgt = new_tgt.cpu()

            tgt_pad = self.translator.tgt_pad_idx

            try:
                src_pad = self.translator.model.src_padding_idx
            except:
                src_pad = self.translator.src_pad_idx

            src_lst = tensor_to_list(src, src_pad)
            new_tgt_lst = tensor_to_list(new_tgt, tgt_pad)
            tgt_context_lst = [x[:-1] for x in new_tgt_lst]
            tgt_words_lst = [x[1:] for x in new_tgt_lst]
            new_data = zip(src_lst, tgt_context_lst, tgt_words_lst)
            train_data = train_data + list(new_data)

            if self.batch_saving is not None:
                if i % self.batch_saving == self.batch_saving - 1:
                    with open(self.base_dir + '/' + str(batch_id) + self.train_file, 'wb') as f:
                        torch.save(train_data, f, dill, 2)
                    batch_id += 1
                    train_data = []
                    print('SAVED FILE')

        seqKD_train = train_data
        seqKD_val = [x for x in self.dataset.val]
        seqKD_test = [x for x in self.dataset.test]

        src_vocab_size = self.dataset.src.vocab_size
        tgt_vocab_size = self.dataset.tgt.vocab_size
        tgt_vocab = self.dataset.tgt.vocab

        print('CREATED ALL OBJECTS')

        if self.batch_saving is not None:
            with open(self.base_dir + '/' + str(batch_id) + self.train_file, 'wb') as f:
                torch.save(seqKD_train, f, dill, 2)
        else:
            with open(self.base_dir + '/' + self.train_file, 'wb') as f:
                torch.save(seqKD_train, f, dill, 2)

        with open(self.base_dir + '/' + self.val_file, 'wb') as f:
            torch.save(seqKD_val, f, dill, 2)

        with open(self.base_dir + '/' + self.test_file, 'wb') as f:
            torch.save(seqKD_test, f, dill, 2)

        with open(self.base_dir + '/' + self.others_file, 'wb') as f:
            items = (src_vocab_size, tgt_vocab_size, tgt_vocab)
            torch.save(items, f, dill, 2)

        return False
