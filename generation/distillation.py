import math
from typing import Dict, List, Optional, Any, Tuple, Iterator, Union
import numpy as np
from tqdm import tqdm

import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_

from flambe.nn import Module
from flambe.logging import log
from flambe.compile import State

from generation.train import Seq2SeqTrainer
from generation.translation.translator import Translator
from generation.sampler import TensorSampler
from generation.utils import pad_and_cat, pad_to_len, delete_extra_pads
from generation.parallel import DataParallelModel, DataParallelCriterion
from generation.metric import CrossEntropyLoss

import pdb

from tqdm import tqdm

import time

BYTES_IN_GB = 1024 ** 3


class Seq2SeqDistillationTrainer(Seq2SeqTrainer):

    def __init__(self,
                 teacher: Module,
                 scheduler_type: str = 'linear',
                 top_k: Union[int, str] = 'None',
                 use_iter: bool = False,
                 encode_during_sampling: bool = False,
                 teacher_translator: Translator = None,
                 **kwargs) -> None:

        super().__init__(**kwargs)

        self.teacher = teacher.eval()
        self.teacher.to(self.device)

        self.beta = 1.

        self.use_iter = use_iter
        if use_iter:
            N = self.iter_per_step * self.max_steps
        else:
            N = self.max_steps
        self.global_iter = 0
        self.encode_during_sampling = encode_during_sampling

        if scheduler_type == 'linear':
            self.get_beta = lambda t: 1. - t / N
        elif scheduler_type == 'exponential':
            self.get_beta = lambda t: 200 ** (-t / N)
        elif scheduler_type == 'reverse_sigmoid':
            self.get_beta = lambda t: 1 / (1 + np.exp((t / N - 0.5) * 20))
        elif scheduler_type == 'ones':
            self.get_beta = lambda t: 1.
        elif scheduler_type == 'zeros':
            self.beta = 0.
            self.get_beta = lambda t: 0.
        else:
            raise ValueError('Not implemented!')

        self.top_k = top_k

        self.teacher_translator = teacher_translator

    def _compute_loss(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if self.use_iter:
            tb_prefix = f"{self.tb_log_prefix} " if self.tb_log_prefix else ""
            log(f'{tb_prefix}Training/Beta', self.beta, self.global_iter)

        if self.top_k != 'None':
            loss = self._compute_loss_top_k(batch)

            if self.use_iter:
                self.global_iter += 1
                self.beta = self.get_beta(self.global_iter)

            return loss

        if self.encode_during_sampling:
            loss = self._encode_and_sample_rnn(batch)

            if self.use_iter:
                self.global_iter += 1
                self.beta = self.get_beta(self.global_iter)

            return loss

        src, tgt_context, tgt_words = batch

        # Sample from translator
        self.model.eval()
        dist = Categorical(torch.tensor([self.beta, 1-self.beta]))
        samp_mask = (dist.sample((src.size(0),)) == 1)

        if torch.sum(samp_mask).item() > 0:
            samp_src = src[samp_mask]
            with torch.no_grad():
                A = time.time()
                samp_tgt_context = self.translator(samp_src)
                print('FIRST BLOCK ' + str(time.time() - A))

                A = time.time()
                samp_tgt_logits = self.teacher(samp_src, samp_tgt_context)
                _, samp_tgt_words = samp_tgt_logits.max(dim=-1)
                print('SECOND BLOCK ' + str(time.time() - A))

            eos_mask = (samp_tgt_context == self.translator.tgt_eos_idx)
            eos_mask |= (samp_tgt_context == self.translator.tgt_pad_idx)
            samp_tgt_words[eos_mask] = self.translator.tgt_pad_idx

            # Merge original and sampled data
            orig_src = src[~samp_mask]
            orig_tgt_context = tgt_context[~samp_mask]
            orig_tgt_words = tgt_words[~samp_mask]

            # Add padding if necessary
            tgt_pad_idx = torch.tensor(self.translator.tgt_pad_idx)
            diff = samp_tgt_context.size(1) - orig_tgt_context.size(1)
            if diff > 0:
                extra = tgt_pad_idx.repeat((orig_tgt_context.size(0),
                                            diff)).to(self.device)
                orig_tgt_context = torch.cat([orig_tgt_context, extra],
                                             dim=1)
            diff = samp_tgt_words.size(1) - orig_tgt_words.size(1)
            if diff > 0:
                extra = tgt_pad_idx.repeat((orig_tgt_words.size(0),
                                            diff)).to(self.device)
                orig_tgt_words = torch.cat([orig_tgt_words, extra],
                                           dim=1)

            new_src = torch.cat([orig_src, samp_src], dim=0)
            new_tgt_context = torch.cat([orig_tgt_context,
                                         samp_tgt_context], dim=0)
            new_tgt_words = torch.cat([orig_tgt_words,
                                       samp_tgt_words], dim=0)
        else:
            new_src = src
            new_tgt_context = tgt_context
            new_tgt_words = tgt_words

        # Train model on merged data
        A = time.time()
        self.model.train()
        pred, target = self.model(new_src, new_tgt_context,
                                  new_tgt_words)
        loss = self.loss_fn(pred, target)
        print('THIRD BLOCK ' + str(time.time() - A))

        if self.use_iter:
            self.global_iter += 1
            self.beta = self.get_beta(self.global_iter)

        return loss

    def _compute_loss_top_k(self, batch):

        src, tgt_context, tgt_words = batch

        # Sample from translator
        self.model.eval()
        dist = Categorical(torch.tensor([self.beta, 1-self.beta]))
        samp_mask = (dist.sample((src.size(0),)) == 1)

        samp_src = src[samp_mask]
        orig_src = src[~samp_mask]

        with torch.no_grad():
            if samp_src.size(0) > 0:
                samp_tgt_context = self.translator(samp_src)
                orig_tgt_context = tgt_context[~samp_mask]

                # Add padding if necessary
                tgt_pad_idx = torch.tensor(self.translator.tgt_pad_idx)
                diff = samp_tgt_context.size(1)-orig_tgt_context.size(1)
                if diff > 0:
                    extra = tgt_pad_idx.repeat((orig_tgt_context.size(0),
                                        diff)).to(self.device)
                    orig_tgt_context = torch.cat([orig_tgt_context,
                                                  extra], dim=1)

                new_src = torch.cat([orig_src, samp_src], dim=0)
                new_tgt_context = torch.cat([orig_tgt_context,
                                             samp_tgt_context], dim=0)
            else:
                new_src = src
                new_tgt_context = tgt_context

            new_tgt_logits = self.teacher(new_src, new_tgt_context)

            if self.top_k == -1:
                new_tgt_dists = torch.softmax(new_tgt_logits, dim=-1)
            else:
                topk, indices = torch.topk(new_tgt_logits,
                                           self.top_k, dim=-1)
                topk = torch.softmax(topk, dim=-1)
                new_tgt_dists = torch.zeros_like(new_tgt_logits)
                new_tgt_dists.scatter_(-1, indices, topk)

        eos_mask = (new_tgt_context == self.translator.tgt_eos_idx)
        eos_mask |= (new_tgt_context == self.translator.tgt_pad_idx)

        # Train model on merged data
        self.model.train()
        logits = self.model(new_src, new_tgt_context)
        pred = torch.log_softmax(logits, dim=-1)
        pred = pred[~eos_mask]
        new_tgt_dists = new_tgt_dists[~eos_mask]

        kl_loss = F.kl_div(pred, new_tgt_dists, reduction='batchmean')
        return kl_loss

    def _encode_and_sample_rnn(self, batch):

        A = time.time()

        src, tgt_context, tgt_words = batch
        dist = Categorical(torch.tensor([self.beta, 1-self.beta]))
        samp_mask = (dist.sample((src.size(0),)) == 1)

        samp_src = src[samp_mask]
        if samp_src.size(0) == 0:
            pred, target = self.model(src, tgt_context, tgt_words)
            return self.loss_fn(pred, target)

        orig_src = src[~samp_mask]
        new_src = torch.cat([samp_src, orig_src], dim=0)

        N = src.size(0)
        n_samp = samp_src.size(0)
        start = torch.tensor(self.translator.tgt_sos_idx).repeat((N, 1))
        start = start.to(self.device)

        beam = [start]
        beam_finished = (start == self.translator.tgt_pad_idx)

        memory, src_padding_mask = self.model.encode(new_src)
        new_tgt_context = start

        max_seq_len = self.translator.max_seq_len
        all_tgt_logits = []

        orig_tgt_context = tgt_context[~samp_mask]
        if orig_tgt_context.size(1) < max_seq_len:
            diff = max_seq_len - orig_tgt_context.size(1)
            zeros = torch.zeros(N - n_samp, diff).long()
            zeros = zeros.to(self.device)
            orig_tgt_context = torch.cat([orig_tgt_context, zeros],
                                         dim=1)
        orig_tgt_words = tgt_words[~samp_mask]
        if orig_tgt_words.size(1) < max_seq_len:
            diff = max_seq_len - orig_tgt_words.size(1)
            zeros = torch.zeros(N - n_samp, diff).long()
            zeros = zeros.to(self.device)
            orig_tgt_words = torch.cat([orig_tgt_words, zeros],
                                        dim=1)

        print('FIRST BLOCK ' + str(time.time() - A))

        A = time.time()

        for t in range(1, max_seq_len+1):
            output = self.model.decode(new_tgt_context, memory,
                                       src_padding_mask)

            tgt_encoding, state = output
            memory = (memory[0], state)
            tgt_logits = self.model.apply_output_layer(tgt_encoding)
            all_tgt_logits.append(tgt_logits)

            if t < max_seq_len:
                _, tokens = tgt_logits.detach().max(dim=2)
                tokens[beam_finished] = self.translator.tgt_pad_idx

                new_tokens = torch.cat([tokens[:n_samp],
                                        orig_tgt_context[:, t:t+1]],
                                        dim=0)
                beam_finished |= (new_tokens == self.translator.tgt_eos_idx)

                new_tgt_context = new_tokens
                beam.append(new_tokens)

            if torch.all(beam_finished).item():
                diff = max_seq_len - len(beam)
                zeros = torch.zeros(batch_size, 1).long()
                zeros = zeros.to(self.device)
                beam = beam + [zeros] * diff
                break

        print('SECOND BLOCK ' + str(time.time() - A))

        A = time.time()

        new_tgt_context = torch.cat(beam, dim=1)
        samp_tgt_context = new_tgt_context[:n_samp]

        with torch.no_grad():
            _, samp_tgt_words = self.teacher(samp_src,
                                             samp_tgt_context).max(dim=2)
        new_tgt_words = torch.cat([samp_tgt_words, orig_tgt_words],
                                  dim=0)

        pred = torch.cat(all_tgt_logits, dim=1)
        mask = (new_tgt_context != 0)
        loss = self.loss_fn(pred[mask], new_tgt_words[mask])

        print('THIRD BLOCK ' + str(time.time() - A))

        return loss


    def _eval_step(self) -> None:

        if self.teacher_translator is not None and self._step == 0:
            self.teacher_translator.initialize(self.teacher)

            iterator = self.train_sampler.sample(self.dataset.train, 1)

            batch_size = self.train_sampler.batch_size
            drop_last = self.train_sampler.drop_last
            shuffle = self.train_sampler.shuffle

            srcs = []
            new_contexts = []
            new_words = []

            for batch in iterator:

                batch = (t.to(self.device) for t in batch)
                src, tgt_context, tgt_words = batch

                with torch.no_grad():
                    new_tgt, new_src = self.teacher_translator(src, src)
                new_tgt_context = new_tgt[:, :-1]
                new_tgt_words = new_tgt[:, 1:]

                srcs.append(new_src[:, :-1].cpu())
                new_contexts.append(new_tgt_context.cpu())
                new_words.append(new_tgt_words.cpu())

            srcs = torch.cat(srcs, dim=0)
            new_contexts = torch.cat(new_contexts, dim=0)
            new_words = torch.cat(new_words, dim=0)

            original_data = (srcs, new_contexts, new_words)
            train_sets = [original_data]

            self.train_sampler = TensorSampler(train_sets,
                                               probs=[1.0],
                                               batch_size=batch_size,
                                               drop_last=drop_last,
                                               shuffle=shuffle)
            self._create_train_iterator()

        super()._eval_step()

        tb_prefix = f"{self.tb_log_prefix} " if self.tb_log_prefix else ""
        # Log beta
        if not self.use_iter:
            self.beta = self.get_beta(self._step)
            log(f'{tb_prefix}Training/Beta', self.beta, self._step)


class Seq2SeqDistillTrainer(Seq2SeqTrainer):

    def __init__(self,
                 teacher: Module,
                 scheduler_type: str = 'linear',
                 batches_per_samp: int = 1,
                 top_k: Union[int, str] = 'None',
                 sample_with_dropout: bool = False,
                 exp_base: float = 200.0,
                 warmup: float = 0.0,
                 final_beta: float = 0.0,
                 **kwargs) -> None:

        super().__init__(**kwargs)

        if self.device_count > 1:
            teacher = DataParallelModel(teacher)
            print("Let's use", self.device_count, "GPUs!")

        self.teacher = teacher.eval()
        self.teacher.to(self.device)

        self.beta = 1.
        N = self.iter_per_step * self.max_steps

        self.batches_per_samp = batches_per_samp
        self.sample_with_dropout = sample_with_dropout

        if scheduler_type == 'linear':
            self.get_beta = lambda t: 1. - (1. - final_beta) * t / N
        elif scheduler_type == 'exponential':
            self.get_beta = lambda t: exp_base ** (-(t / N - warmup)) if t / N > warmup else 1.
        elif scheduler_type == 'reverse_sigmoid':
            self.get_beta = lambda t: 1 / (1 + np.exp((t / N - 0.5) * 20))
        elif scheduler_type == 'ones':
            self.get_beta = lambda t: 1.
        elif scheduler_type == 'zeros':
            self.beta = 0.
            self.get_beta = lambda t: 0.
        elif scheduler_type == 'half':
            self.beta = 0.5
            self.get_beta = lambda t: 0.5
        else:
            raise ValueError('Not implemented!')

        self.top_k = top_k

        print('ITER PER STEP: %d' % self.iter_per_step)

        if self.top_k == -1 and self.device_count > 1:
            self.distill_fn = DataParallelCriterion(CrossEntropyLoss())

    def sample(self, data, n_epochs):
        iterator = self.train_sampler.sample(self.dataset.train, self.n_epochs)

        while True:
            batches = []
            for _ in range(self.batches_per_samp):
                batch = next(iterator)
                # batches.append(batch)
                batches.append([b.detach() for b in batch])

            batch_sizes = [b[0].size(0) for b in batches]

            pad = self.translator.tgt_pad_idx
            src = pad_and_cat([b[0] for b in batches], pad)
            tgt_context = pad_and_cat([b[1] for b in batches], pad)
            tgt_words = pad_and_cat([b[2] for b in batches], pad)

            length = self.translator.max_seq_len
            tgt_context = pad_to_len(tgt_context, length, pad, dim=1)
            tgt_words = pad_to_len(tgt_words, length, pad, dim=1)

            beta = self.beta
            dist = Categorical(torch.tensor([beta, 1-beta]))
            samp_mask = (dist.sample((src.size(0),)) == 1)

            n_samp = torch.sum(samp_mask).item()
            eos = self.translator.tgt_eos_idx

            if n_samp > 0:
                samp_src = src[samp_mask]
                samp_src = delete_extra_pads(samp_src, pad_idx=pad)
                if self.device_count == 1:
                    samp_src = samp_src.to(self.device)
                if self.sample_with_dropout:
                    self.translator.train()
                else:
                    self.translator.eval()

                with torch.no_grad():
                    samp_tgt_context = self.translator(samp_src)
                    if self.device_count > 1:
                        samp_tgt_context = torch.cat([x.cpu() for x in samp_tgt_context], dim=0)
                    tgt_context[samp_mask] = samp_tgt_context.cpu()

                self.translator.train()

            if n_samp > 0 and self.top_k == 'None':
                with torch.no_grad():
                    try:
                        samp_tgt_logits = self.teacher(samp_src,
                                                       samp_tgt_context)
                    except RuntimeError:

                        for i in range(self.device_count):
                            torch.cuda.set_device(i)
                            torch.cuda.empty_cache()
                            print(torch.cuda.memory_cached())

                        torch.cuda.set_device(0)
                        print('EMPTIED CACHE FOR BATCH')
                        samp_tgt_logits = self.teacher(samp_src,
                                                       samp_tgt_context)

                    if self.device_count > 1:
                        samp_tgt_words = torch.cat([x.max(dim=-1)[1].cpu() for x in samp_tgt_logits], dim=0)
                        # samp_tgt_logits = torch.cat([x.cpu() for x in samp_tgt_logits], dim=0)
                    else:
                        _, samp_tgt_words = samp_tgt_logits.max(dim=-1)

                # Pad words correctly
                eos_mask = (samp_tgt_context == eos)
                eos_mask |= (samp_tgt_context == pad)
                samp_tgt_words[eos_mask] = pad

                tgt_words[samp_mask] = samp_tgt_words.cpu()

                # torch.cuda.empty_cache()

            elif self.top_k == -1:
                if self.device_count > 1:
                    print("HI")
                    print(src.size(), tgt_context.size(), tgt_words.size())
                    with torch.no_grad():
                        tgt_logits = self.teacher(src, tgt_context)
                        tgt_probs = [torch.softmax(x, dim=-1) for x in tgt_logits]
                        tgt_words = tgt_probs

                    print('YIELD ONE')
                    yield src, tgt_context, tgt_words

                    continue
                else:
                    with torch.no_grad():
                        tgt_logits = self.teacher(src.to(self.device),
                                                  tgt_context.to(self.device))
                        tgt_probs = torch.softmax(tgt_logits, dim=-1)
                    tgt_words = tgt_probs

            elif self.top_k != 'None':
                with torch.no_grad():
                    tgt_logits = self.teacher(src.to(self.device),
                                              tgt_context.to(self.device))
                    topk, indices = torch.topk(tgt_logits,
                                               self.top_k, dim=-1)
                    topk = torch.softmax(topk, dim=-1)
                    tgt_probs = torch.zeros_like(tgt_logits)
                    tgt_probs.scatter_(-1, indices, topk)
                tgt_words = tgt_probs

            for i in range(self.batches_per_samp):
                start = sum(batch_sizes[:i])
                end = sum(batch_sizes[:(i+1)])

                # Get slice
                src_ = src[start:end]
                tgt_context_ = tgt_context[start:end]
                tgt_words_ = tgt_words[start:end]

                mask_ = (src_ != pad)
                max_len = mask_.sum(dim=1).max().item()
                src_ = src_[:, :max_len]

                mask_ = (tgt_context_ != pad) & (tgt_context_ != eos)
                mask_[:, 0] = True # First token SOS
                max_len = mask_.sum(dim=1).max().item()
                tgt_context_ = tgt_context_[:, :max_len]
                tgt_words_ = tgt_words_[:, :max_len]

                yield src_, tgt_context_, tgt_words_

    def _compute_kl_loss(self, batch):
        if self.device_count > 1:
            src, tgt_context, tgt_words = batch
            logits = self.model(src, tgt_context)
            log_probs = [torch.log_softmax(x, dim=-1) for x in logits]

            eos = self.translator.tgt_eos_idx
            pad = self.translator.tgt_pad_idx
            mask = (tgt_context != eos) & (tgt_context != pad)
            mask[:, 0] = True # First token (SOS)
            device_ids = self.distill_fn.device_ids

            mask, _ = self.distill_fn.scatter([mask], {}, device_ids)
            mask = [m[0] for m in mask]

            log_probs = [lp[m] for lp, m in zip(log_probs, mask)]
            tgt_words = [tw[m] for tw, m in zip(tgt_words, mask)]

            return self.distill_fn(log_probs, tgt_words)

        src, tgt_context, tgt_words = batch
        logits = self.model(src, tgt_context)
        log_probs = torch.log_softmax(logits, dim=-1)

        eos = self.translator.tgt_eos_idx
        pad = self.translator.tgt_pad_idx
        mask = (tgt_context != eos) & (tgt_context != pad)

        flat_tgt_words = tgt_words[mask]
        flat_log_probs = log_probs[mask]

        if self.top_k == -1:
            entropies = -torch.sum(flat_tgt_words * flat_log_probs, dim=1)
            kl_loss = entropies.mean(dim=0)
        else:
            nz_mask = (flat_tgt_words != 0)
            kl_loss = -torch.mean(flat_tgt_words[nz_mask] * flat_log_probs[nz_mask]) * self.top_k
        return kl_loss

    def _create_train_iterator(self):
        self._train_iterator = self.sample(self.dataset.train, self.n_epochs)

    def _train_step(self) -> None:
        """Run a training step over the training data."""
        self.model.train()
        metrics_with_states: List[Tuple] = [(metric, {}) for metric in self.training_metrics]
        self._last_train_log_step = 0

        log_prefix = f"{self.tb_log_prefix} " if self.tb_log_prefix else ""
        log_prefix += 'Training'

        with torch.enable_grad():
            for i in range(self.iter_per_step):

                # print('MEMORY ALLOCATED %f' % float(torch.cuda.memory_allocated() / BYTES_IN_GB))
                # print('MEMORY CACHED %f' % float(torch.cuda.memory_cached() / BYTES_IN_GB))

                t = time.time()
                # Zero the gradients and clear the accumulated loss
                self.optimizer.zero_grad()
                accumulated_loss = 0.0
                for _ in range(self.batches_per_iter):

                    # Get next batch
                    try:
                        batch = next(self._train_iterator)
                    except StopIteration:
                        self._create_train_iterator()
                        batch = next(self._train_iterator)

                    if self.device_count == 1:
                        batch = self._batch_to_device(batch)

                    # Compute loss
                    if self.top_k == 'None':
                        _, _, loss = self._compute_batch(batch, metrics_with_states)
                    else:
                        loss = self._compute_kl_loss(batch)
                    print('LOSS')
                    print(loss)
                    accumulated_loss += loss.item() / self.batches_per_iter

                    loss.backward()
                    # try:
                    #     loss.backward()
                    # except RuntimeError:
                    #     torch.cuda.empty_cache()
                    #     print('EMPTIED CACHE FOR LOSS')
                    #     continue

                # Log loss
                global_step = (self.iter_per_step * self._step) + i
                self.beta = self.get_beta(global_step)

                # Clip gradients if necessary
                if self.max_grad_norm:
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if self.max_grad_abs_val:
                    clip_grad_value_(self.model.parameters(), self.max_grad_abs_val)

                log(f'{log_prefix}/Loss', accumulated_loss, global_step)
                if self.device_count > 1:
                    log(f'{log_prefix}/Gradient_Norm', self.model.module.gradient_norm,
                        global_step)
                    log(f'{log_prefix}/Parameter_Norm', self.model.module.parameter_norm,
                        global_step)
                else:
                    log(f'{log_prefix}/Gradient_Norm', self.model.gradient_norm,
                        global_step)
                    log(f'{log_prefix}/Parameter_Norm', self.model.parameter_norm,
                        global_step)
                log(f'{log_prefix}/Beta', self.beta, global_step)

                # Optimize
                self.optimizer.step()

                # Update iter scheduler
                if self.iter_scheduler is not None:
                    lr = self.optimizer.param_groups[0]['lr']  # type: ignore
                    log(f'{log_prefix}/LR', lr, global_step)
                    self.iter_scheduler.step()  # type: ignore

                # Zero the gradients when exiting a train step
                self.optimizer.zero_grad()
                # logging train metrics
                if self.extra_training_metrics_log_interval > self._last_train_log_step:
                    self._log_metrics(log_prefix, metrics_with_states, global_step)
                    self._last_train_log_step = i

                print('TOTAL TIME: %f' % (time.time() - t))
            if self._last_train_log_step != i:
                # log again at end of step, if not logged at the end of
                # step before
                self._log_metrics(log_prefix, metrics_with_states, global_step)

    def _compute_batch(self, batch: Tuple[torch.Tensor, ...],
                       metrics: List[Tuple] = [],
                       model = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if model is None:
            model = self.model
            model_passed_in = False
        else:
            model_passed_in = True

        if self.device_count == 1:
            batch = self._batch_to_device(batch)

        if self.device_count > 1:
            out = model(*batch)
            pred, target = zip(*out)
        else:
            pred, target = model(*batch)

        for metric, state in metrics:
            if self.device_count > 1:
                for p, t in zip(pred, target):
                    metric.aggregate(state, p, t)
            else:
                metric.aggregate(state, pred, target)
        if not model_passed_in:
            loss = self.loss_fn(pred, target)
        else:
            loss = 0
        return pred, target, loss

    def _state(self,
               state_dict: State,
               prefix: str,
               local_metadata: Dict[str, Any]) -> State:
        state_dict[prefix + 'optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state_dict[prefix + 'scheduler'] = self.scheduler.state_dict()
        if self.teacher is not None:
            if (prefix + 'teacher') in state_dict:
                state_dict[prefix + 'teacher'] = None
        return state_dict
