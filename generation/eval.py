from typing import Optional

import torch

from flambe.compile import Component
from flambe.dataset import Dataset
from flambe.learn.utils import select_device
from flambe.nn import Module
from flambe.metric import Metric
from flambe.sampler import Sampler, BaseSampler
from flambe.logging import log


class Evaluator(Component):

    def __init__(self,
                 dataset: Dataset,
                 model: Module,
                 metric_fn: Metric,
                 eval_sampler: Optional[Sampler] = None,
                 eval_data: str = 'test',
                 device: Optional[str] = None,
                 save_preds: bool = False,
                 teacher: Module = None,
                 save_targets: bool = False,
                 gen_style: str = 'greedy') -> None:
        self.eval_sampler = eval_sampler or BaseSampler(batch_size=16, shuffle=False)
        self.model = model
        self.metric_fn = metric_fn
        self.eval_metric = None
        self.dataset = dataset

        self.device = select_device(device)

        data = getattr(dataset, eval_data)
        self._eval_iterator = self.eval_sampler.sample(data)

        # By default, no prefix applied to tb logs
        self.tb_log_prefix = None

        self.save_preds = save_preds
        self.decode_data = None
        self.teacher = teacher
        self.save_targets = save_targets
        self.targets = None
        self.gen_style = gen_style
        self.register_attrs('decode_data', 'targets')

    def run(self, block_name: str = None) -> bool:
        self.model.to(self.device)
        self.model.eval()
        if self.teacher is not None:
            self.teacher.to(self.device)
            self.teacher.eval()

        with torch.no_grad():
            preds, targets = [], []
            if self.save_preds:
                sources = []
            if self.teacher is not None:
                teacher_preds = []

            for batch in self._eval_iterator:
                batch = [t.to(self.device) for t in batch]
                pred = self.model(batch[0], gen_style=self.gen_style)
                if self.save_preds:
                    source = batch[0]
                    if source.size(1) < 50:
                        extra = torch.zeros((source.size(0), 50-source.size(1))).long().to(self.device)
                        source = torch.cat([source, extra], dim=1)
                    sources.append(source)
                    if self.teacher is not None:
                        student_context = pred[:, :-1].to(self.device)
                        _, teacher_pred = self.teacher(batch[0], student_context).max(dim=2)
                        teacher_pred[student_context == self.model.eos_idx] = self.model.pad_idx
                        teacher_pred[student_context == self.model.pad_idx] = self.model.pad_idx
                        teacher_preds.append(teacher_pred.cpu())
                target = batch[1]
                if target.size(1) < 50:
                    extra = torch.zeros((target.size(0), 50-target.size(1))).long().to(self.device)
                    target = torch.cat([target, extra], dim=1)
                preds.append(pred.cpu())
                targets.append(target.cpu())

            preds = torch.cat(preds, dim=0)  # type: ignore
            if self.save_preds:
                sources = torch.cat(sources, dim=0)
                if self.teacher is not None:
                    teacher_preds = torch.cat(teacher_preds, dim=0)
                    self.decode_data = (sources, preds[:, :-1], teacher_preds)
                else:
                    self.decode_data = (sources, preds[:, :-1], preds[:, 1:])
            targets = torch.cat(targets, dim=0)  # type: ignore
            if self.save_targets:
                self.targets = targets
            self.eval_metric = self.metric_fn(preds, targets).item()

            tb_prefix = f"{self.tb_log_prefix} " if self.tb_log_prefix else ""

            log(f'{tb_prefix}Eval {self.metric_fn}',  # type: ignore
                self.eval_metric, global_step=0)  # type: ignore

        return False

    def metric(self) -> Optional[float]:
        return self.eval_metric
