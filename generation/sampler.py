import math
import torch
from typing import List
from torch.utils.data import TensorDataset
from torch.distributions import Categorical

from flambe.sampler.base import *
from flambe.learn.utils import select_device
from flambe.logging import log

from generation.translation.greedy import GreedyTranslator
from generation.utils import pad_to_len


class BaseSamplerWithFilter(BaseSampler):

    def __init__(self, max_seq_len, **kwargs):
        self.max_seq_len = max_seq_len
        super().__init__(**kwargs)

    def sample(self,
               data: Sequence[Sequence[torch.Tensor]],
               n_epochs: int = 1) -> Iterator[Tuple[torch.Tensor, ...]]:
        if len(data) == 0:
            raise ValueError("No examples provided")

        if self.downsample:
            if not (0 < self.downsample <= 1):
                raise ValueError("Downsample value should be in the range (0, 1]")
            if self.downsample_seed:
                downsample_generator = np.random.RandomState(self.downsample_seed)
            else:
                downsample_generator = np.random
            random_indices = downsample_generator.permutation(len(data))
            data = [data[i] for i in random_indices[:int(self.downsample * len(data))]]

        collate_fn_p = partial(collate_fn, pad=self.pad)
        def collate_with_filter(batch):
            batch = filter(lambda lst: all([len(x) <= self.max_seq_len for x in lst]), batch)
            return collate_fn_p(batch)
        loader = DataLoader(dataset=data,  # type: ignore
                            shuffle=self.shuffle,
                            batch_size=self.batch_size,
                            collate_fn=collate_with_filter,
                            num_workers=self.n_workers,
                            pin_memory=self.pin_memory,
                            drop_last=self.drop_last)

        if n_epochs == -1:
            while True:
                yield from loader
        else:
            for _ in range(n_epochs):
                yield from loader


class TensorSampler(BaseSampler):

    def __init__(self,
                 data: List[List[torch.Tensor]] = [],
                 probs: List[float] = [],
                 max_seq_len: int = 50, **kwargs):

        assert len(data) == len(probs)

        gold_source = None
        contexts = []
        targets = []
        for i, (source, context, target) in enumerate(data):
            if i == 0:
                gold_source = source
            contexts.append(context)
            targets.append(target)

            assert gold_source[0].tolist() == source[0].tolist()

        contexts = torch.stack(contexts, dim=1)
        targets = torch.stack(targets, dim=1)

        self.dataset = (gold_source, contexts, targets)

        self.dist = Categorical(torch.tensor(probs))

        super().__init__(**kwargs)

    def sample(self,
               data: Sequence[Sequence[torch.Tensor]],
               n_epochs: int = 1) -> Iterator[Tuple[torch.Tensor, ...]]:
        if len(data) == 0:
            raise ValueError("No examples provided")

        dataset = TensorDataset(*self.dataset)

        loader = DataLoader(dataset=dataset,
                            shuffle=self.shuffle,
                            batch_size=self.batch_size,
                            num_workers=self.n_workers,
                            pin_memory=self.pin_memory,
                            drop_last=self.drop_last)

        if n_epochs == -1:
            while True:
                for source, contexts, targets in loader:
                    batch_size = source.size(0)
                    row = torch.arange(batch_size)
                    col = self.dist.sample((batch_size,))
                    yield source, contexts[row, col], targets[row, col]
        else:
            for _ in range(n_epochs):
                for source, contexts, targets in loader:
                    batch_size = source.size(0)
                    row = torch.arange(batch_size)
                    col = self.dist.sample((batch_size,))
                    yield source, contexts[row, col], targets[row, col]

    def length(self, data: Sequence[Sequence[torch.Tensor]]) -> int:
        return math.ceil(len(self.dataset[0]) / self.batch_size)


class GuidedStudentSampler(BaseSampler):

    def __init__(self, max_seq_len, min_seq_len=0,
                 translator=None, teacher=None, device=None,
                 sample_factor=1, scheduler_type='exponential',
                 **kwargs):
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.device = select_device(None)

        self.sample_factor = sample_factor
        self.scheduler_type = scheduler_type

        self.initialize(translator, teacher)
        super().__init__(**kwargs)

    def initialize(self, translator, teacher):
        if translator is not None:
            self.translator = translator
        if teacher is not None:
            self.teacher = teacher.eval().to(self.device)

    def sample(self,
               data: Sequence[Sequence[torch.Tensor]],
               n_epochs: int = 1) -> Iterator[Tuple[torch.Tensor, ...]]:
        if self.translator is None or self.teacher is None:
            raise ValueError('Cannot sample because one of student or teacher is missing!')

        samp_per_epoch = math.ceil(self.length(data)/self.sample_factor)
        N = n_epochs * samp_per_epoch
        beta = 1.

        if self.scheduler_type == 'linear':
            get_beta = lambda t: 1. - t / N
        elif self.scheduler_type == 'exponential':
            get_beta = lambda t: 200 ** (-t / N)
        elif self.scheduler_type == 'reverse_sigmoid':
            get_beta = lambda t: 1 / (1 + np.exp((t / N - 0.5) * 20))
        elif self.scheduler_type == 'ones':
            get_beta = lambda t: 1.
        elif self.scheduler_type == 'zeros':
            get_beta = lambda t: 0.
        else:
            raise ValueError('Not implemented!')

        if len(data) == 0:
            raise ValueError("No examples provided")

        collate_fn_p = partial(collate_fn, pad=self.pad)
        def collate_with_filter(batch):
            batch = filter(lambda lst: all([len(x) <= self.max_seq_len for x in lst]), batch)
            return collate_fn_p(batch)

        sample_batch_size = self.batch_size * self.sample_factor
        loader = DataLoader(dataset=data,  # type: ignore
                            shuffle=self.shuffle,
                            batch_size=sample_batch_size,
                            collate_fn=collate_with_filter,
                            num_workers=self.n_workers,
                            pin_memory=self.pin_memory,
                            drop_last=self.drop_last)

        for epoch in range(n_epochs):
            for samp_count, batch in enumerate(loader):

                batch = [x.clone() for x in batch]
                src, tgt_context, tgt_words = batch
                max_seq_len = self.translator.max_seq_len
                pad_idx = self.translator.tgt_pad_idx
                tgt_context = pad_to_len(tgt_context, max_seq_len,
                                         pad_idx, dim=1)
                tgt_words = pad_to_len(tgt_words, max_seq_len,
                                       pad_idx, dim=1)

                beta = get_beta(epoch * samp_per_epoch + samp_count)
                dist = Categorical(torch.tensor([beta, 1-beta]))
                samp_mask = (dist.sample((src.size(0),)) == 1)

                if torch.sum(samp_mask).item() > 0:
                    samp_src = src[samp_mask].to(self.device)
                    self.translator.model.eval()

                    with torch.no_grad():
                        samp_tgt_context = self.translator(samp_src)
                        samp_tgt_logits = self.teacher(samp_src,
                                                       samp_tgt_context)
                        _, samp_tgt_words = samp_tgt_logits.max(dim=-1)

                    # Pad words correctly
                    eos_idx = self.translator.tgt_eos_idx
                    pad_idx = self.translator.tgt_pad_idx
                    eos_mask = (samp_tgt_context == eos_idx)
                    eos_mask |= (samp_tgt_context == pad_idx)
                    samp_tgt_words[eos_mask] = pad_idx

                    samp_src = samp_src.cpu()
                    samp_tgt_context = samp_tgt_context.cpu()
                    samp_tgt_words = samp_tgt_words.cpu()

                    tgt_context[samp_mask] = samp_tgt_context
                    tgt_words[samp_mask] = samp_tgt_words

                    max_len = (tgt_words != pad_idx).sum(dim=1).max().item()
                    tgt_context = tgt_context[:, :max_len]
                    tgt_words = tgt_words[:, :max_len]

                    self.translator.model.train()

                B = self.batch_size
                for i in range(self.sample_factor):
                    step = self.sample_factor * (epoch * samp_per_epoch + samp_count) + i
                    log('Training/Beta', beta, step)
                    src_slice = src[i*B:(i+1)*B]
                    tgt_context_slice = tgt_context[i*B:(i+1)*B]
                    tgt_words_slice = tgt_words[i*B:(i+1)*B]
                    yield src_slice, tgt_context_slice, tgt_words_slice

    def length(self, data: Sequence[Sequence[torch.Tensor]]) -> int:
        downsample = self.downsample or 1
        return downsample * len(data) // self.batch_size
