class TransformerDecoder(Module):
    """TransformerDecoder is a stack of N decoder layers"""

    def __init__(self,
                 input_size: int,
                 d_model: int,
                 nhead: int,
                 num_layers: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1) -> None:
        """Initialize the TransformerDecoder.

        Parameters
        ---------
        input_size : int
            The embedding dimension of the model.  If different from
            d_model, a linear projection layer is added.
        d_model : int
            The number of expected features in encoder/decoder inputs.
        nhead : int, optional
            The number of heads in the multiheadattention.
        num_layers : int
            The number of sub-encoder-layers in the encoder (required).
        dim_feedforward : int, optional
            The inner feedforard dimension, by default 2048.
        dropout : float, optional
            The dropout percentage, by default 0.1.

        """
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model

        if input_size != d_model:
            self.proj = nn.Linear(input_size, d_model)

        layer = TransformerDecoderLayer(d_model,
                                        nhead,
                                        dim_feedforward,
                                        dropout)

        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers

        self._reset_parameters()

    def forward(self,  # type: ignore
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pass the inputs (and mask) through the decoder layer in turn.

        Parameters
        ----------
        tgt: torch.Tensor
            The sequence to the decoder (required).
        memory: torch.Tensor
            The sequence from the last layer of the encoder (required).
        tgt_mask: torch.Tensor, optional
            The mask for the tgt sequence (optional).
        memory_mask: torch.Tensor, optional
            The mask for the memory sequence (optional).
        padding_mask: torch.Tensor, optional
            The mask for the tgt keys per batch (optional).
            Should be True for tokens to leave untouched, and False
            for padding tokens.
        memory_key_padding_mask: torch.Tensor, optional
            The mask for the memory keys per batch (optional).

        Returns
        -------
        torch.Tensor

        """
        output = tgt.transpose(0, 1)
        memory = memory.transpose(0, 1)

        if self.input_size != self.d_model:
            output = self.proj(output)

        if tgt_mask is None:
            T = output.size(0) # sequence length
            tgt_mask = torch.tensor(float('-inf')).repeat((T, T))
            device = select_device(None)
            tgt_mask = torch.triu(tgt_mask, diagonal=1).to(device)

        for i in range(self.num_layers):
            output = self.layers[i](output,
                                    memory,
                                    tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    padding_mask=padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)

        return output.transpose(0, 1)

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
