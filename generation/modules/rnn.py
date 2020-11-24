from typing import Optional, Tuple, cast
import warnings
import logging
import sys
import traceback

import torch
from torch import nn
from torch import Tensor

from flambe.nn.module import Module

logger = logging.getLogger(__name__)


class RNNEncoder(Module):
    """Implements a multi-layer RNN.

    This module can be used to create multi-layer RNN models, and
    provides a way to reduce to output of the RNN to a single hidden
    state by pooling the encoder states either by taking the maximum,
    average, or by taking the last hidden state before padding.

    Padding is dealt with by using torch's PackedSequence.

    Attributes
    ----------
    rnn: nn.Module
        The rnn submodule

    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_layers: int = 1,
                 rnn_type: str = 'lstm',
                 dropout: float = 0,
                 bidirectional: bool = False,
                 layer_norm: bool = False,
                 highway_bias: float = 0,
                 rescale: bool = True,
                 enforce_sorted: bool = False,
                 **kwargs) -> None:
        """Initializes the RNNEncoder object.

        Parameters
        ----------
        input_size : int
            The dimension the input data
        hidden_size : int
            The hidden dimension to encode the data in
        n_layers : int, optional
            The number of rnn layers, defaults to 1
        rnn_type : str, optional
           The type of rnn cell, one of: `lstm`, `gru`, `sru`
           defaults to `lstm`
        dropout : float, optional
            Amount of dropout to use between RNN layers, defaults to 0
        bidirectional : bool, optional
            Set to use a bidrectional encoder, defaults to False
        layer_norm : bool, optional
            [SRU only] whether to use layer norm
        highway_bias : float, optional
            [SRU only] value to use for the highway bias
        rescale : bool, optional
            [SRU only] whether to use rescaling
        enforce_sorted: bool
            Whether rnn should enforce that sequences are ordered by
            length. Requires True for ONNX support. Defaults to False.
        kwargs
            Additional parameters to be passed to SRU when building
            the rnn.

        Raises
        ------
        ValueError
            The rnn type should be one of: `lstm`, `gru`, `sru`

        """
        super().__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.enforce_sorted = enforce_sorted
        self.output_size = 2 * hidden_size if bidirectional else hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        if rnn_type in ['lstm', 'gru']:
            if kwargs:
                logger.warn(f"The following '{kwargs}' will be ignored " +
                            "as they are only considered when using 'sru' as " +
                            "'rnn_type'")

            rnn_fn = nn.LSTM if rnn_type == 'lstm' else nn.GRU
            self.rnn = rnn_fn(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=n_layers,
                              dropout=dropout,
                              bidirectional=bidirectional)
        elif rnn_type == 'sru':
            from sru import SRU
            try:
                self.rnn = SRU(input_size,
                               hidden_size,
                               num_layers=n_layers,
                               dropout=dropout,
                               bidirectional=bidirectional,
                               layer_norm=layer_norm,
                               rescale=rescale,
                               highway_bias=highway_bias,
                               **kwargs)
            except TypeError:
                raise ValueError(
                    f"Unknown kwargs passed to SRU: {kwargs} "
                    f"({sys.exc_info()[1], traceback.print_tb(sys.exc_info()[2])})"
                )
        else:
            raise ValueError(f"Unkown rnn type: {rnn_type}, use of of: gru, sru, lstm")

    def forward(self,
                data: Tensor,
                state: Optional[Tensor] = None,
                padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Performs a forward pass through the network.

        Parameters
        ----------
        data : Tensor
            The input data, as a float tensor of shape [B x S x E]
        state: Tensor
            An optional previous state of shape [L x B x H]
        padding_mask: Tensor, optional
            The padding mask of shape [B x S], dtype should be bool

        Returns
        -------
        Tensor
            The encoded output, as a float tensor of shape [B x S x H]
        Tensor
            The encoded state, as a float tensor of shape [L x B x H]

        """
        data = data.transpose(0, 1)
        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1)

        if padding_mask is None:
            # Default RNN behavior
            output, state = self.rnn(data, state)
        elif self.rnn_type == 'sru':
            # SRU takes a mask instead of PackedSequence objects
            # ~ operator negates bool tensor in torch 1.3
            output, state = self.rnn(data, state, mask_pad=(~padding_mask))
        else:
            # Deal with variable length sequences
            lengths = padding_mask.long().sum(dim=0)
            # Pass through the RNN
            packed = nn.utils.rnn.pack_padded_sequence(data, lengths,
                                                       enforce_sorted=self.enforce_sorted)
            output, state = self.rnn(packed, state)
            output, _ = nn.utils.rnn.pad_packed_sequence(output)

        if self.rnn_type in ['gru', 'lstm']:
            if isinstance(state, tuple):
                state = tuple(self._rearrange_state(s) for s in state)
            else:
                state = self._rearrange_state(state)

        # TODO investigate why PyTorch returns type Any for output
        return output.transpose(0, 1).contiguous(), state  # type: ignore

    def _rearrange_state(self, state: Tensor) -> Tensor:
        """Ensure the state is returned with the correct dimensions.

        Parameters
        ----------
        state: Tensor
            The state to reshape

        Returns
        -------
        Tensor
            The state with shape [L x B x H*num_directions]

        """
        return state.view(self.n_layers, -1, self.output_size)


class RNNDecoder(Module):
    """Implements a multi-layer RNN with (optionally) attention.

    Attributes
    ----------
    rnn: nn.Module
        The rnn submodule

    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_layers: int = 1,
                 rnn_type: str = 'lstm',
                 dropout: float = 0,
                 layer_norm: bool = False,
                 highway_bias: float = 0,
                 rescale: bool = True,
                 enforce_sorted: bool = False,
                 attention: Optional[nn.Module] = None,
                 activation: Optional[nn.Module] = None,
                 **kwargs) -> None:
        """Initializes the RNNDecoder object.
        """
        super().__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.enforce_sorted = enforce_sorted
        if rnn_type in ['lstm', 'gru']:
            if kwargs:
                logger.warn(f"The following '{kwargs}' will be ignored " +
                            "as they are only considered when using 'sru' as " +
                            "'rnn_type'")

            rnn_fn = nn.LSTM if rnn_type == 'lstm' else nn.GRU
            self.rnn = rnn_fn(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=n_layers,
                              dropout=dropout)
        elif rnn_type == 'sru':
            from sru import SRU
            try:
                self.rnn = SRU(input_size,
                               hidden_size,
                               num_layers=n_layers,
                               dropout=dropout,
                               layer_norm=layer_norm,
                               rescale=rescale,
                               highway_bias=highway_bias,
                               **kwargs)
            except TypeError:
                raise ValueError(f"Unknown kwargs passed to SRU: {kwargs}")
        else:
            raise ValueError(f"Unknown rnn type: {rnn_type}, use of of: gru, sru, lstm")

        self.attention = attention

        if self.attention is not None:
            self.linear = torch.nn.Linear(in_features=hidden_size * 2,
                                          out_features=hidden_size)

        self.activation = activation

    def forward(self,  # type: ignore
                tgt: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                memory: Optional[Tuple[torch.Tensor]] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Performs a forward pass through the network.

        Parameters
        ----------
        tgt : Tensor
            The input data, as a float tensor of shape [B x S x E]
        state: Tensor
            An optional previous state of shape [L x B x H]
        padding_mask: Tensor, optional
            The padding mask of shape [B x S]

        Returns
        -------
        Tensor
            The encoded output, as a float tensor of shape [B x S x H]
        Tensor
            The encoded state, as a float tensor of shape [L x B x H]

        """

        tgt = tgt.transpose(0, 1)
        padding_mask = padding_mask.transpose(0, 1)

        if memory is None:
            src_encoding = None
            state = None
        elif isinstance(memory, tuple):
            src_encoding, state = memory
        else:
            src_encoding = memory
            state = None

        if padding_mask is None:
            # Default RNN behavior
            output, state = self.rnn(tgt, state)
        elif self.rnn_type == 'sru':
            # SRU takes a mask instead of PackedSequence objects
            # ~ operator negates bool tensor in torch 1.3
            output, state = self.rnn(tgt, state, mask_pad=(~padding_mask).byte())
        else:
            # Deal with variable length sequences
            lengths = padding_mask.long().sum(dim=0)

            lengths[lengths == 0] = 1 # trick to avoid pack_padded_sequence throwing error for length = 0

            # Pass through the RNN
            packed = nn.utils.rnn.pack_padded_sequence(tgt, lengths,
                                                       enforce_sorted=self.enforce_sorted)
            output, state = self.rnn(packed, state)
            output, _ = nn.utils.rnn.pad_packed_sequence(output)

        if self.attention is not None:
            if src_encoding is None:
                raise ValueError('Must provide `memory` from encoder to RNNDecoder with attention submodule.')

            src_encoding = src_encoding.transpose(0, 1)
            attn_states, _ = self.attention(query=output,
                                            key=src_encoding,
                                            value=src_encoding,
                                            key_padding_mask=~memory_key_padding_mask)
            output = torch.cat([output, attn_states], dim=2)
            output = self.linear(output)

        if self.activation is not None:
            output = self.activation(output)

        return output.transpose(0, 1).contiguous(), state
