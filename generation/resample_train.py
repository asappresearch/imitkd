import math
from typing import Dict, List, Optional, Any, Tuple, Iterator
import numpy as np

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_
from torch.utils.data import TensorDataset

from flambe.compile import Schema, State, Component, Link
from flambe.learn.utils import select_device
from flambe.nn import Module  # type: ignore[attr-defined]
from flambe.sampler import Sampler
from flambe.metric import Metric
from flambe.logging import log
from flambe.sampler.base import collate_fn

from generation.translation.translator import Translator
from generation.modules.seq2seq import Seq2Seq
from generation.train import Seq2SeqTrainer
from generation.sampler import TensorSampler, DataLoader
from generation.utils import nan_mean


class Seq2SeqResampleTrainer(Seq2SeqTrainer):

    def __init__(self,
                 milestones: List[int],
                 original_data: List[List[torch.Tensor]],
                 max_seq_len: int,
                 teacher: Optional[Seq2Seq] = None,
                 teacher_translator: Optional[Translator] = None,
                 aggregate: bool = False,
                 merge_original: bool = False,
                 use_conf: bool = False,
                 merge_prob: Optional[List[float]] = None,
                 **kwargs):

        self.milestones = set(milestones)
        self.aggregate = aggregate
        self.merge_original = merge_original
        self.merge_prob = None
        if self.merge_original and self.merge_prob is not None:
            self.counter = 0
        self.use_conf = use_conf
        if teacher is not None:
            device = select_device(None)
            self.teacher = teacher.to(device)
            self.teacher = teacher.eval()

        batch_size = kwargs['train_sampler'].batch_size
        drop_last = kwargs['train_sampler'].drop_last
        shuffle = kwargs['train_sampler'].shuffle

        original_data = filter(lambda lst: all([len(x) <= max_seq_len for x in lst]), original_data)
        original_data = collate_fn(original_data, pad=0)

        self.teacher_translator = teacher_translator

        self.original = original_data
        self.train_sets = [original_data]

        train_sampler = TensorSampler(self.train_sets,
                                           probs=[1.0],
                                           batch_size=batch_size,
                                           drop_last=drop_last,
                                           shuffle=shuffle)
        kwargs['train_sampler'] = train_sampler
        super().__init__(**kwargs)


    def _eval_step(self) -> None:
        super()._eval_step()

        if self._step == 0 and self.teacher_translator is not None:
            self.teacher_translator.initialize(self.teacher)

            batch_size = self.train_sampler.batch_size
            drop_last = self.train_sampler.drop_last
            shuffle = self.train_sampler.shuffle

            dataset = TensorDataset(*self.original)
            teacher_loader = DataLoader(dataset=dataset,
                                        shuffle=False,
                                        batch_size=batch_size,
                                        drop_last=False)
            srcs = []
            new_contexts = []
            new_words = []

            for batch in teacher_loader:
                batch = (t.to(self.device) for t in batch)
                src, tgt_context, tgt_words = batch

                with torch.no_grad():
                    new_tgt = self.teacher_translator(src)
                new_tgt_context = new_tgt[:, :-1]
                new_tgt_words = new_tgt[:, 1:]

                srcs.append(src.cpu())
                new_contexts.append(new_tgt_context.cpu())
                new_words.append(new_tgt_words.cpu())

            srcs = torch.cat(srcs, dim=0)
            new_contexts = torch.cat(new_contexts, dim=0)
            new_words = torch.cat(new_words, dim=0)

            original_data = (srcs, new_contexts, new_words)

            self.original = original_data
            self.train_sets = [original_data]

            self.train_sampler = TensorSampler(self.train_sets,
                                               probs=[1.0],
                                               batch_size=batch_size,
                                               drop_last=drop_last,
                                               shuffle=shuffle)
            self._create_train_iterator()

        if self._step in self.milestones:

            print(self._step)
            batch_size = self.train_sampler.batch_size
            drop_last = self.train_sampler.drop_last
            shuffle = self.train_sampler.shuffle

            loader = DataLoader(dataset=TensorDataset(*self.original),
                       shuffle=False,
                       batch_size=batch_size,
                       drop_last=False)
            srcs = []
            new_contexts = []
            new_words = []

            if self.use_conf:
                num_confs = 0

            for batch in loader:
                batch = self._batch_to_device(batch)
                src, tgt_context, tgt_words = batch

                with torch.no_grad():
                    new_tgt_context = self.translator(src)
                    new_tgt_logits = self.teacher(src, new_tgt_context)
                    if self.use_conf:
                        new_tgt_probs = torch.softmax(new_tgt_logits,
                                                      dim=-1)
                        new_tgt_confs, new_tgt_words = new_tgt_probs.max(dim=-1)

                    else:
                        _, new_tgt_words = new_tgt_logits.max(dim=-1)

                mask = (new_tgt_context == self.translator.tgt_eos_idx)
                mask |= (new_tgt_context == self.translator.tgt_pad_idx)
                new_tgt_words[mask] = self.translator.tgt_pad_idx
                if self.use_conf:
                    new_tgt_confs[mask] = np.nan
                    new_tgt_confs_means = nan_mean(new_tgt_confs, dim=1)
                    idx_mask = (new_tgt_confs_means > 0.65).unsqueeze(1) # using 0.5 cutoff
                    new_tgt_context = torch.where(idx_mask,
                                                  new_tgt_context,
                                                  tgt_context)
                    new_tgt_words = torch.where(idx_mask,
                                                new_tgt_words,
                                                tgt_words)
                    num_confs += idx_mask.sum().item()

                srcs.append(src.cpu())
                new_contexts.append(new_tgt_context.cpu())
                new_words.append(new_tgt_words.cpu())

            srcs = torch.cat(srcs, dim=0)
            new_contexts = torch.cat(new_contexts, dim=0)
            new_words = torch.cat(new_words, dim=0)

            if self.use_conf:
                frac = num_confs / len(srcs)
                print('Fraction above threshold: %.2f' % frac)

            if self.aggregate:
                self.train_sets.append((srcs, new_contexts, new_words))
            elif self.merge_original:
                new_set = (srcs, new_contexts, new_words)
                if len(self.train_sets) == 1:
                    self.train_sets.append(new_set)
                else:
                    self.train_sets[1] = new_set
            else:
                self.train_sets = [(srcs, new_contexts, new_words)]
            # self.train_sets.append((srcs, new_contexts, new_words))
            if self.merge_original and self.merge_prob is not None:
                p = self.merge_prob[self.counter]
                self.counter += 1
                self.train_sampler = TensorSampler(self.train_sets,
                                                   probs=[1-p, p],
                                                   batch_size=batch_size,
                                                   drop_last=drop_last,
                                                   shuffle=shuffle)
            else:
                N = len(self.train_sets)
                self.train_sampler = TensorSampler(self.train_sets,
                                                   probs=[1/N]*N,
                                                   batch_size=batch_size,
                                                   drop_last=drop_last,
                                                   shuffle=shuffle)
            self._create_train_iterator()
