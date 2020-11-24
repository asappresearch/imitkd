from typing import List
import nltk
from nltk.tokenize import word_tokenize

from flambe.tokenizer import Tokenizer
import fastBPE


class BPETokenizer(Tokenizer):
    """Implement a subword level tokenizer using
       byte pair encoding.  Tokenization is done using
       fastBPE (https://github.com/glample/fastBPE) and
       requires a fastBPE codes file.

    """

    def __init__(self,
                 codes_path: str,
                 nltk_tokenize_first: bool = False) -> None:
        """Initialize the tokenizer.

        Parameters
        ----------
        codes_path : str
            Path to codes file created using
            fastBPE.

        """
        self.bpe = fastBPE.fastBPE(codes_path)
        self.nltk_tokenize_first = nltk_tokenize_first
        nltk.download('punkt', quiet=True)

    def tokenize(self, example: str) -> List[str]:
        """Tokenize an input example.

        Parameters
        ----------
        example : str
            The input example, as a string

        Returns
        -------
        List[str]
            The output subword tokens, as a list of strings

        """
        if self.nltk_tokenize_first:
            example = ' '.join(word_tokenize(example))
        return self.bpe.apply([example])[0].split(" ")
