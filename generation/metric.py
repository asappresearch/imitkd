import torch
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import sent_tokenize
import sacrebleu
import rouge
import torch.nn.functional as F
import re

from flambe.metric import Metric
from flambe.learn.utils import select_device


class Bleu(Metric):

    def __init__(self, vocab, specials=[], method='nltk', tokenizer='13a'):
        """Initalizes the Perplexity metrc."""
        self.vocab = vocab
        self.itos = list(vocab)
        self.specials = specials
        self.method = method
        self.tokenizer = tokenizer
        if method not in ['nltk', 'sacrebleu']:
            raise ValueError("Invalid method for BLEU metric: " + method)

    def _convert_to_strings(self, int_lst):
        str_lst = [self.itos[x] for x in int_lst]
        str_lst = [x for x in str_lst if x not in self.specials]
        return str_lst

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.cpu().tolist()
        target = target.cpu().tolist()
        references = []
        hypotheses = []
        for p, t in zip(pred, target):
            ref = self._convert_to_strings(t)
            hyp = self._convert_to_strings(p)

            # For BPE
            ref = ' '.join(ref).replace('@@ ', '')
            hyp = ' '.join(hyp).replace('@@ ', '')
            references.append(ref)
            hypotheses.append(hyp)

        if self.method == 'nltk':
            references = [[r.split(' ')] for r in references]
            hypotheses = [h.split(' ') for h in hypotheses]
            score = corpus_bleu(references, hypotheses)
        else:
            references = [references]
            score = sacrebleu.corpus_bleu(hypotheses, references,
                            tokenize=self.tokenizer).score
        return torch.tensor(score)


class Rouge(Metric):

    def __init__(self, vocab=None, specials=[], type='n', n=1,
                 max_length=None, tokenizer=None, alpha=0.5, which='f', tokenize_sent=False, eos='\n', dump_file=None):
        """Initalizes the Perplexity metrc."""
        self.vocab = vocab
        self.specials = specials
        self.n = n
        self.type = type
        self.rouge = rouge.Rouge(metrics=['rouge-' + type],
                                 max_n=n,
                                 limit_length=max_length is not None,
                                 length_limit=max_length,
                                 length_limit_type='words',
                                 apply_avg=True,
                                 apply_best=False,
                                 alpha=alpha, # Default F1_score
                                 weight_factor=1.0, # Correct bug
                                 stemming=True)
        self.tokenizer = tokenizer
        self.which = which
        self.tokenize_sent = tokenize_sent
        self.eos = eos
        self.dump_file = dump_file
        nltk.download('punkt', quiet=True)

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.cpu().tolist()
        target = target.cpu().tolist()
        references = []
        hypotheses = []
        if self.tokenizer is not None:
            refs = [[self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)] for g in target]
            hyps = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in pred]
        else:
            raise ValueError('Not Implemented!')

        if self.tokenize_sent:
            refs = [[self.sent_tokenize(x[0])] for x in refs]
            hyps = [self.sent_tokenize(x) for x in hyps]

        if self.dump_file is not None:
            with open(self.dump_file + '_target.txt', 'w') as file:
                for i, ref in enumerate(refs):
                    file.write(ref[0])
                    if i < len(refs) - 1:
                        file.write('\n')

            with open(self.dump_file + '_pred.txt', 'w') as file:
                for i, hyp in enumerate(hyps):
                    file.write(hyp)
                    if i < len(hyps) - 1:
                        file.write('\n')

        refs = [[x[0].replace('EOS', '\n')] for x in refs]
        hyps = [x.replace('EOS', '\n') for x in hyps]

        score = self.rouge.get_scores(hyps, refs)
        if self.type == 'n':
            return torch.tensor(score['rouge-' + str(self.n)][self.which])
        else:
            return torch.tensor(score['rouge-' + self.type][self.which])

    def sent_tokenize(self, out_string):
        dec_replace = lambda x: x.group(0).replace(' ', '')
        out_string = re.sub(r'\d+ \. \d+', dec_replace, out_string)

        out_string = out_string.replace(' .', '.')
        eos = ' ' + self.eos.strip() + ' '
        out_string = eos.join(sent_tokenize(out_string)) + eos
        out_string = out_string.replace('.', ' .')

        dec_replace = lambda x: x.group(0).replace('.', ' . ')
        out_string = re.sub(r'\d+\.\d+', dec_replace, out_string)
        return out_string

    def collapse_punc(self, out_string):
        # Replace decimals
        dec_replace = lambda x: x.group(0).replace(' ', '')
        out_string = re.sub(r'\d+ \. \d+', dec_replace, out_string)

        # Replace everything else
        out_string = (
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace("n ' t ", "n't ")
            .replace(" ' m ", "'m ")
            .replace(" do not ", " don't ")
            .replace(" ' s ", "'s ")
            .replace(" ' ve ", "'ve ")
            .replace(" ' re ", "'re ")
            .replace(" ' d ", "'d ")
        )
        return out_string



class LabelSmoothingLoss(Metric, torch.nn.Module):
    """
    Adapted from https://github.com/OpenNMT/OpenNMT-py/blob/e8622eb5c6117269bb3accd8eb6f66282b5e67d9/onmt/utils/loss.py#L186.
    """
    def __init__(self,
                 alpha: float,
                 tgt_vocab_size: int,
                 ignore_index: int = 0):
        assert 0.0 < alpha <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = alpha / (tgt_vocab_size - 2)
        self.one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        self.one_hot[self.ignore_index] = 0

        self.confidence = 1.0 - alpha

    def compute(self, pred: torch.Tensor, target: torch.Tensor):
        pred = torch.log_softmax(pred, dim=-1)

        device = select_device(None)
        prob = self.one_hot.repeat(target.size(0), 1).to(device)
        prob.scatter_(1, target.unsqueeze(1), self.confidence)
        prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(pred, prob, reduction='batchmean')


class TVLoss(Metric):

    def __init__(self, one_hot_target: bool = True):
        self.one_hot_target = one_hot_target

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(pred, dim=-1)

        if self.one_hot_target:
            tv = 1 - torch.gather(probs, 1, index=target.unsqueeze(1))
            return tv.mean()
        else:
            tv = 0.5 * torch.abs(probs - target).sum(dim=1)
            return tv.mean()


class CrossEntropyLoss(Metric, torch.nn.Module):

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        entropies = -torch.sum(target * pred, dim=1)
        kl_loss = entropies.mean(dim=0)
        return kl_loss
