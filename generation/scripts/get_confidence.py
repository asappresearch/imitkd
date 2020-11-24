from typing import List
import torch
from torch.utils.data import TensorDataset, DataLoader

from flambe.compile import Component
from flambe.sampler.base import collate_fn
from flambe.learn.utils import select_device

from generation.modules.seq2seq import Seq2Seq
from generation.translation.translator import Translator


class GetConfidence(Component):

    def __init__(self,
                 original_data: List[List[torch.Tensor]],
                 max_seq_len: int,
                 batch_size: int,
                 model: Seq2Seq,
                 translator: Translator,
                 teacher: Seq2Seq):

        super().__init__()

        device = select_device(None)
        self.device = device
        self.teacher = teacher.to(self.device)
        self.teacher = teacher.eval()

        self.translator = translator
        self.model = model.to(self.device)
        self.model = model.eval()
        self.translator.initialize(self.model)

        original_data = filter(lambda lst: all([len(x) <= max_seq_len for x in lst]), original_data)
        original_data = collate_fn(original_data, pad=0)

        self.data  = original_data
        self.batch_size = batch_size

        self.old_words = None
        self.new_contexts = None
        self.new_words = None
        self.new_confs = None

        self.register_attrs('old_words', 'new_contexts',
                            'new_words', 'new_confs')

    def run(self) -> bool:

        loader = DataLoader(dataset=TensorDataset(*self.data),
                   shuffle=False,
                   batch_size=self.batch_size,
                   drop_last=False)
        old_words = []
        new_contexts = []
        new_words = []
        new_confs = []
        for batch in loader:
            batch = (t.to(self.device) for t in batch)
            src, tgt_context, tgt_words = batch

            with torch.no_grad():
                new_tgt_context = self.translator(src)
                new_tgt_logits = self.teacher(src, new_tgt_context)
                probs = torch.softmax(new_tgt_logits, dim=-1)
                new_tgt_confs, new_tgt_words = probs.max(dim=-1)

            mask = (new_tgt_context == self.translator.tgt_eos_idx)
            mask |= (new_tgt_context == self.translator.tgt_pad_idx)
            new_tgt_words[mask] = self.translator.tgt_pad_idx

            old_words.append(tgt_words.cpu())
            new_contexts.append(new_tgt_context.cpu())
            new_words.append(new_tgt_words.cpu())
            new_confs.append(new_tgt_confs.cpu())

        old_words = torch.cat(old_words, dim=0)
        new_contexts = torch.cat(new_contexts, dim=0)
        new_words = torch.cat(new_words, dim=0)
        new_confs = torch.cat(new_confs, dim=0)

        self.old_words = old_words
        self.new_contexts = new_contexts
        self.new_words = new_words
        self.new_confs = new_confs

        return False
