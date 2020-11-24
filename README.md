# ImitKD

This repository contains code for the paper "Autoregressive Knowledge Distillation through Imitation Learning" by Alexander Lin, Jeremy Wohlwend, Howard Chen, and Tao Lei.

To get started, simply clone the repository and run: 
```
pip install -r requirements.txt
pip install -e .
```

Relevant PyTorch objects can be found in the `generation` directory and ported into Python code.  For example, 
```python
import torch
from torch.nn import Embedding, Tanh, Linear
from generation import (
	Seq2Seq, 
	RNNEncoder, 
	RNNDecoder, 
	DotProductAttention
)

vocab_size = 10
model = Seq2Seq(
	src_embedding=Embedding(vocab_size, 128),
	tgt_embedding=Embedding(vocab_size, 128),
	encoder=RNNEncoder(
		input_size=128,
		n_layers=2,
		hidden_size=128,
		rnn_type='sru',
		bidirectional=True),
	decoder=RNNDecoder(
		input_size=128,
		n_layers=2,
		hidden_size=256,
		rnn_type='sru',
		attention=DotProductAttention(),
		activation=Tanh()
	),
	output_layer=Linear(256, vocab_size)
)

src = torch.randint(vocab_size, (32, 20))
tgt = torch.randint(vocab_size, (32, 20))
pred = model(src, tgt)
print(pred.size()) # batch_size x seq_len x vocab_size
```

Hyperparameter configurations for all experiments can be found in the `configs` directory.  

To use this repository with Flambé, first fix all resource absolute 
paths to work with your machine (this only needs to be done once):
```
python fix_resources.py
```

Then, simply execute the desired config, e.g.:
```
flambe configs/iwslt/rnn/vanilla.yaml
```