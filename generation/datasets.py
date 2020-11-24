from typing import List, Tuple, Optional, Dict, Union, Iterable
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
import tarfile
import zipfile
import numpy as np
import dill
import os

from torch import Tensor
import torch

from flambe.dataset import TabularDataset
from flambe.field import Field
from flambe.sampler import Sampler

from generation.translation.translator import Translator


class IWSLTDataset(TabularDataset):
    """The official IWSLT dataset."""

    IWSLT_URL = "https://wit3.fbk.eu/archive"

    def __init__(self,
                 src: str = 'de',
                 tgt: str = 'en',
                 year: str = '2014',
                 val_frac_from_train: float = 0.04,
                 val_files: List[str] = [],
                 test_files: List[str] = ['dev2010', 'tst2010', 'tst2011', 'tst2012'],
                 cache: bool = False,
                 seed: Optional[int] = 1234,
                 transform: Dict[str, Union[Field, Dict]] = None) -> None:
        self.URL = f"{self.IWSLT_URL}/{year}-01/texts/{src}/{tgt}/{src}-{tgt}.tgz"
        self.src = src
        self.tgt = tgt
        self.yr = year[-2:]
        self.seed = seed
        if seed is not None:
            np.random.seed(self.seed)

        file_tmp = urllib.request.urlretrieve(self.URL, filename=None)[0]
        tar = tarfile.open(file_tmp)

        dir = f'{src}-{tgt}/'

        train_prefix = dir + f'train.tags.{src}-{tgt}.'
        src_train_txt = tar.extractfile(train_prefix + src).read()
        trg_train_txt = tar.extractfile(train_prefix + tgt).read()
        train = self._process_text_pair(src_train_txt, trg_train_txt)

        if val_frac_from_train > 0:
            n_train = len(train)
            n_val = int(val_frac_from_train * n_train)
            val_idx = set(np.random.choice(np.arange(n_train),
                                           replace=False, size=n_val))
            val = [x for i, x in enumerate(train) if i in val_idx]
            train = [x for i, x in enumerate(train) if i not in val_idx]
            val = val + self._process_xlm_files(tar, dir, val_files)
        else:
            val = self._process_xlm_files(tar, dir, val_files)

        test = self._process_xlm_files(tar, dir, test_files)

        super().__init__(
            train=train,
            val=val,
            test=test,
            cache=cache,
            named_columns=['src', 'tgt'],
            transform=transform
        )

    def _process_text_pair(self, src_txt, tgt_txt):
        src_lines = src_txt.decode('utf-8').split('\n')
        tgt_lines = tgt_txt.decode('utf-8').split('\n')
        assert (len(src_lines) == len(tgt_lines))

        zipped_lines = []
        for x, y in zip(src_lines, tgt_lines):
            if len(x) > 0 and len(y) > 0 and x[0] != '<' and y[0] != '<':
                zipped_lines.append((x, y))
        return zipped_lines

    def _process_xlm(self, txt):
        txt = txt.decode('utf-8').replace('& amp ;', '&amp;')
        lines = []
        etree = ET.fromstring(txt)
        for seg in etree.iter(tag='seg'):
            lines.append(seg.text.strip())
        return lines

    def _process_xlm_files(self, tar, dir, files_lst):
        data = []
        yr = self.yr
        for file in files_lst:
            prefix = f'{dir}IWSLT{yr}.TED.{file}.{self.src}-{self.tgt}.'
            if file == 'dev2012':
                prefix = f'{dir}IWSLT{yr}.TEDX.{file}.{self.src}-{self.tgt}.'
            src_txt = tar.extractfile(f"{prefix}{self.src}.xml").read()
            tgt_txt = tar.extractfile(f"{prefix}{self.tgt}.xml").read()
            src_lines = self._process_xlm(src_txt)
            tgt_lines = self._process_xlm(tgt_txt)
            assert (len(src_lines) == len(tgt_lines))
            data += list(zip(src_lines, tgt_lines))
        return data


class WMT16Dataset(TabularDataset):

    URL = 'https://drive.google.com/uc?export=download&confirm=oN_z&id=0B_bZck-ksdkpM25jRUN2X2UxMm8'

    def __init__(self,
                 file: str,
                 val_files: List[str] = ['newstest2013'],
                 test_files: List[str] = ['newstest2014'],
                 use_bpe_tokenized: bool = True,
                 cache: bool = False,
                 transform: Dict[str, Union[Field, Dict]] = None) -> None:

        self.use_bpe_tokenized = use_bpe_tokenized
        if not use_bpe_tokenized:
            raise ValueError('Default train set is BPE tokenized.  Cannot untokenized dataset!')

        tar = tarfile.open(file)

        train = self._process_files(tar, ['train'], clean=True)
        val = self._process_files(tar, val_files)
        test = self._process_files(tar, test_files)

        tar.close()

        super().__init__(
            train=train,
            val=val,
            test=test,
            cache=cache,
            named_columns=['src', 'tgt'],
            transform=transform
        )

    def _process_files(self, tar, files, clean=False):
        data = []
        for f in files:
            if self.use_bpe_tokenized:
                clean_str = 'clean.' if clean else ''
                src_path = f'{f}.tok.{clean_str}bpe.32000.en'
                tgt_path = f'{f}.tok.{clean_str}bpe.32000.de'
            else:
                src_path = f'{f}.en'
                tgt_path = f'{f}.de'
            src_txt = tar.extractfile(src_path).read()
            tgt_txt = tar.extractfile(tgt_path).read()
            src_lines = src_txt.decode('utf-8').split('\n')
            tgt_lines = tgt_txt.decode('utf-8').split('\n')
            src_lines = [x for x in src_lines if len(x) > 0]
            tgt_lines = [x for x in tgt_lines if len(x) > 0]
            assert len(src_lines) == len(tgt_lines)
            data += list(zip(src_lines, tgt_lines))
        return data


class OPUSDataset(TabularDataset):

    OPUS_URL = 'http://opus.nlpl.eu/EMEA.php'

    def __init__(self,
                 src: str = 'de',
                 tgt: str = 'en',
                 reverse: bool = False,
                 corpus: str = 'JRC-Acquis',
                 val_frac: float = 0.05,
                 test_frac: float = 0.1,
                 cache: bool = False,
                 seed: Optional[int] = 1234,
                 transform: Dict[str, Union[Field, Dict]] = None) -> None:
        if corpus == 'JRC-Acquis':
            self.URL = 'https://object.pouta.csc.fi/OPUS-JRC-Acquis/v3.0/moses/de-en.txt.zip'
        elif corpus == 'EMEA':
            self.URL = 'https://object.pouta.csc.fi/OPUS-EMEA/v3/moses/de-en.txt.zip'
        else:
            raise ValueError('Corpus not implemented!')
        self.src = src
        self.tgt = tgt
        self.corpus = corpus
        self.seed = seed
        if seed is not None:
            np.random.seed(self.seed)

        file_tmp = urllib.request.urlretrieve(self.URL, filename=None)[0]
        archive = zipfile.ZipFile(file_tmp)

        src_file = f'{corpus}.{src}-{tgt}.{src}'
        tgt_file = f'{corpus}.{src}-{tgt}.{tgt}'

        src_txt = archive.read(src_file)
        tgt_txt = archive.read(tgt_file)
        train = self._process_text_pair(src_txt, tgt_txt)

        if val_frac + test_frac > 0:
            n_train = len(train)
            n_val = int(val_frac * n_train)
            n_test = int(test_frac * n_train)
            holdout_idx = np.random.choice(np.arange(n_train),
                                           replace=False,
                                           size=n_val+n_test)
            val_idx = set(holdout_idx[:n_val])
            test_idx = set(holdout_idx[n_val:])

            train_data = []
            val_data = []
            test_data = []
            for i, x in enumerate(train):
                if i in val_idx:
                    val_data.append(x)
                elif i in test_idx:
                    test_data.append(x)
                else:
                    train_data.append(x)

        if reverse:
            train_data = [(x[1], x[0]) for x in train_data]
            val_data = [(x[1], x[0]) for x in val_data]
            test_data = [(x[1], x[0]) for x in test_data]

        super().__init__(
            train=train_data,
            val=val_data,
            test=test_data,
            cache=cache,
            named_columns=['src', 'tgt'],
            transform=transform
        )

    def _process_text_pair(self, src_txt, tgt_txt):
        src_lines = src_txt.decode('utf-8').split('\n')
        tgt_lines = tgt_txt.decode('utf-8').split('\n')
        assert (len(src_lines) == len(tgt_lines))

        zipped_lines = []
        for x, y in zip(src_lines, tgt_lines):
            if len(x) > 0 and len(y) > 0 and x[0] != '<' and y[0] != '<':
                zipped_lines.append((x, y))
        return zipped_lines


class CNNDMDataset(TabularDataset):

    def __init__(self,
                 file: str,
                 cache: bool = False,
                 transform: Dict[str, Union[Field, Dict]] = None) -> None:

        archive = zipfile.ZipFile(file)

        read = lambda x: archive.read(x).decode('utf-8').split('\n')

        train_src = read('cnn_dm/train.source')
        train_tgt = read('cnn_dm/train.target')
        assert len(train_src) == len(train_tgt)
        train = list(zip(train_src, train_tgt))

        val_src = read('cnn_dm/val.source')
        val_tgt = read('cnn_dm/val.target')
        assert len(val_src) == len(val_tgt)
        val = list(zip(val_src, val_tgt))

        test_src = read('cnn_dm/test.source')
        test_tgt = read('cnn_dm/test.target')
        assert len(test_src) == len(test_tgt)
        test = list(zip(test_src, test_tgt))

        super().__init__(
            train=train,
            val=val,
            test=test,
            cache=cache,
            named_columns=['src', 'tgt'],
            transform=transform
        )


class TensorDataset(TabularDataset):

    def __init__(self,
                 base_dir: str,
                 train_file: Union[str, List[str]],
                 val_file: str,
                 test_file: str,
                 others_file: str,
                 columns: List[str]):

        if base_dir[-4:] == '.zip':
            file = zipfile.ZipFile(base_dir)
            base_dir = os.path.dirname(os.path.realpath(base_dir))
            file.extractall(base_dir)
            print('EXTRACTED FILE')

        if isinstance(train_file, list):
            train = []
            for file in train_file:
                with open(base_dir + '/' + file, 'rb') as f:
                    x = torch.load(f)
                    train = train + x
        else:
            with open(base_dir + '/' + train_file, 'rb') as f:
                train = torch.load(f)

        with open(base_dir + '/' + val_file, 'rb') as f:
            val = torch.load(f)

        with open(base_dir + '/' + test_file, 'rb') as f:
            test = torch.load(f)

        with open(base_dir + '/' + others_file, 'rb') as f:
            src_n_vocab, tgt_n_vocab, tgt_vocab = torch.load(f)

        self.src_vocab_size = src_n_vocab
        self.tgt_vocab_size = tgt_n_vocab
        self.tgt_vocab = tgt_vocab

        super().__init__(
            train=train,
            val=val,
            test=test,
            cache=False,
            named_columns=columns,
            transform=None
        )
