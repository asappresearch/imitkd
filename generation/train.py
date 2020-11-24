import math
from typing import Dict, List, Optional, Any, Tuple, Iterator

import torch
from torch import nn
from generation.parallel import DataParallelModel, DataParallelCriterion
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_

from flambe.dataset import Dataset
from flambe.compile import Schema, State, Component, Link
from flambe.learn import Trainer
from flambe.nn import Module
from flambe.sampler import Sampler
from flambe.metric import Metric
from flambe.logging import log

from generation.metric import Bleu
from generation.translation.translator import Translator


class Seq2SeqTrainer(Trainer):
    """Implement a Trainer block.

    A `Trainer` takes as input data, model and optimizer,
    and executes training incrementally in `run`.

    Note that it is important that a trainer run be long enough
    to not increase overhead, so at least a few seconds, and ideally
    multiple minutes.

    """

    def __init__(self,
                 dataset: Dataset,
                 train_sampler: Sampler,
                 val_sampler: Sampler,
                 model: Module,
                 loss_fn: Metric,
                 metric_fn: Metric,
                 translator: Translator,
                 bleu_fn: Bleu,
                 optimizer: Optimizer,
                 scheduler: Optional[_LRScheduler] = None,
                 iter_scheduler: Optional[_LRScheduler] = None,
                 device: Optional[str] = None,
                 max_steps: int = 10,
                 epoch_per_step: float = 1.0,
                 iter_per_step: Optional[int] = None,
                 batches_per_iter: int = 1,
                 lower_is_better: bool = False,
                 max_grad_norm: Optional[float] = None,
                 max_grad_abs_val: Optional[float] = None,
                 extra_validation_metrics: Optional[List[Metric]] = None,
                 eval_translator: Optional[Translator] = None) -> None:
        """Initialize an instance of Trainer

        Parameters
        ----------
        dataset : Dataset
            The dataset to use in training the model
        train_sampler : Sampler
            The sampler to use over training examples during training
        val_sampler : Sampler
            The sampler to use over validation examples
        model : Module
            The model to train
        loss_fn: Metric
            The loss function to use in training the model
        metric_fn: Metric
            The metric function to use in evaluation
        optimizer : torch.optim.Optimizer
            The optimizer to use
        scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            An optional learning rate scheduler to run after each step
        iter_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            An optional learning rate scheduler to run after each batch
            (i.e iteration)
        device: str, optional
            The device to use in the computation.
        max_steps : int, optional
            The maximum number of training steps to run
        epoch_per_step : float, optional
            Fraction of an epoch to perform in a single training step
            (i.e before a checkpoint.) Defaults to 1.
            Overridden by `iter_per_step`, if given.
        iter_per_step : int, optional
            Number of iterations to perform in a single training step.
            Overrides `epoch_per_step` if given.
        batches_per_iter : int, optional
            Number of batches to pass through the model before
            calling optimizer.step. Requires the sampler to have
            drop_last set to True. (default set to 1 so optimizer.step
            is called after every batch)
        lower_is_better : bool, optional
            If true, the lowest val metric is considered best,
            otherwise the highest. Defaults to False.
        max_grad_norm : float, optional
            Maximum Euclidean norm of gradient after clipping.
        max_grad_abs_val: float, optional
            Maximum absolute value of all gradient vector components
            after clipping.
        extra_validation_metrics: Optional[List[Metric]]
            A list with extra metrics to show in each step
            but which don't guide the training procedures
            (i.e model selection through early stopping)

        """
        self.device_count = torch.cuda.device_count()

        if self.device_count > 1:
            model = DataParallelModel(model)
            loss_fn = DataParallelCriterion(loss_fn)
            print("Let's use", self.device_count, "GPUs!")

        super().__init__(dataset, train_sampler, val_sampler, model, loss_fn, metric_fn, optimizer, scheduler, iter_scheduler, device, max_steps, epoch_per_step, iter_per_step, batches_per_iter, lower_is_better, max_grad_norm,
        max_grad_abs_val, extra_validation_metrics)

        self.translator = translator
        if self.device_count > 1:
            self.translator.initialize(self.model.module)

            self.translator = DataParallelModel(self.translator)
            self.translator.tgt_sos_idx = self.translator.module.tgt_sos_idx
            self.translator.tgt_eos_idx = self.translator.module.tgt_eos_idx
            self.translator.tgt_pad_idx = self.translator.module.tgt_pad_idx
            self.translator.max_seq_len = self.translator.module.max_seq_len
        else:
            self.translator.initialize(self.model)

        self.bleu = bleu_fn
        self._best_bleu = None

        init_num_params = sum([len(p.view(-1)) for p in self.model.parameters()])
        print('Model has ' + str(init_num_params) + ' parameters.')

        self.eval_translator = eval_translator
        if self.eval_translator is not None:
            if self.device_count > 1:
                self.eval_translator.initialize(self.model.module)
                self.eval_translator = DataParallelModel(self.eval_translator)
            else:
                self.eval_translator.initialize(self.model)


    def _eval_step(self) -> None:
        """Run an evaluation step over the validation data."""
        self.model.eval()
        translator = self.eval_translator or self.translator
        translator.eval()

        metric_fn_state: Dict[Metric, Dict] = {}
        metrics_with_states: List[Tuple] = \
            [(metric, {}) for metric in self.extra_validation_metrics]

        # Initialize a 1-epoch iteration through the validation set
        val_iterator = self.val_sampler.sample(self.dataset.val)
        bleu_iterator = self.val_sampler.sample(self.dataset.val)
        bleu_iterator = map(lambda x: (x[0], x[2]), bleu_iterator)

        with torch.no_grad():
            loss = []
            for batch in val_iterator:
                _, _, batch_loss = self._compute_batch(
                    batch, [(self.metric_fn, metric_fn_state), *metrics_with_states])
                loss.append(batch_loss.item())
            val_loss = np.NaN if loss == [] else sum(loss) / len(loss)
            val_metric = self.metric_fn.finalize(metric_fn_state)

            bleu_state = {}
            for batch in bleu_iterator:
                _ = self._compute_batch(batch, [(self.bleu, bleu_state)], model=translator)
            bleu = self.bleu.finalize(bleu_state)
            # preds, targets = self._aggregate_trans_preds(bleu_iterator)
            # bleu = self.bleu(preds, targets).item()

        # Update best model
        sign = (-1)**(self.lower_is_better)
        if self._best_bleu is None or bleu > self._best_bleu:
            self._best_bleu = bleu
            best_model_state = self.model.state_dict()
            for k, t in best_model_state.items():
                best_model_state[k] = t.cpu().detach()
            self._best_model = best_model_state

        # Update scheduler
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                # torch's _LRScheduler.step DOES have a default value
                # so passing in no args is fine; it will automatically
                # compute the current epoch
                self.scheduler.step()  # type: ignore

        tb_prefix = f"{self.tb_log_prefix} " if self.tb_log_prefix else ""

        # Log metrics
        log(f'{tb_prefix}Validation/Loss', val_loss, self._step)
        log(f'{tb_prefix}Validation/{self.metric_fn}', val_metric, self._step)
        log(f'{tb_prefix}Validation/{self.bleu}', bleu, self._step)
        log(f'{tb_prefix}Best/{self.bleu}', self._best_bleu, self._step)
        for (metric, state) in metrics_with_states:
            log(f'{tb_prefix}Validation/{metric}',
                metric.finalize(state), self._step)  # type: ignore

    def _train_step(self) -> None:
        """Run a training step over the training data."""
        self.model.train()
        metrics_with_states: List[Tuple] = [(metric, {}) for metric in self.training_metrics]
        self._last_train_log_step = 0

        log_prefix = f"{self.tb_log_prefix} " if self.tb_log_prefix else ""
        log_prefix += 'Training'

        with torch.enable_grad():
            for i in range(self.iter_per_step):

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

                    _, _, loss = self._compute_batch(batch, metrics_with_states)
                    accumulated_loss += loss.item() / self.batches_per_iter

                    loss.backward()

                # Log loss
                global_step = (self.iter_per_step * self._step) + i

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

            if self._last_train_log_step != i:
                # log again at end of step, if not logged at the end of
                # step before
                self._log_metrics(log_prefix, metrics_with_states, global_step)

    def _compute_batch(self, batch: Tuple[torch.Tensor, ...],
                       metrics: List[Tuple] = [],
                       model = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Computes a batch.

        Does a model forward pass over a batch, and returns prediction,
        target and loss.

        Parameters
        ----------
        batch: Tuple[torch.Tensor, ...]
            The batch to train on.

        """
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

    def _load_state(self,
                   state_dict: State,
                   prefix: str,
                   local_metadata: Dict[str, Any],
                   strict: bool,
                   missing_keys: List[Any],
                   unexpected_keys: List[Any],
                   error_msgs: List[Any]) -> None:
       self.optimizer.load_state_dict(state_dict[prefix + 'optimizer'])
       if self.scheduler is not None:
           self.scheduler.load_state_dict(state_dict[prefix + 'scheduler'])
       # Useful when loading the model after training
       self._step = state_dict[prefix + '_step']
       self._best_model = state_dict[prefix + '_best_model']
       # done = self._step >= self.max_steps
       self.model.load_state_dict(state_dict[prefix + '_best_model'], strict=False)
