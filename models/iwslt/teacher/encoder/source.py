class TransformerEncoder(Module):
    """TransformerEncoder is a stack of N encoder layers."""

    def __init__(self,
                 input_size: int = 512,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1) -> None:
        """Initialize the TransformerEncoder.

        Parameters
        ---------
        input_size : int
            The embedding dimension of the model.  If different from
            d_model, a linear projection layer is added.
        d_model : int
            the number of expected features in encoder/decoder inputs.
            Default ``512``.
        nhead : int, optional
            the number of heads in the multiheadattention
            Default ``8``.
        num_layers : int
            the number of sub-encoder-layers in the encoder (required).
            Default ``6``.
        dim_feedforward : int, optional
            the inner feedforard dimension. Default ``2048``.
        dropout : float, optional
            the dropout percentage. Default ``0.1``.

        """
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model

        if input_size != d_model:
            self.proj = nn.Linear(input_size, d_model)

        layer = TransformerEncoderLayer(d_model,
                                        nhead,
                                        dim_feedforward,
                                        dropout)

        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers

        self._reset_parameters()

    def forward(self,  # type: ignore
                src: torch.Tensor,
                memory: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pass the input through the endocder layers in turn.

        Parameters
        ----------
        src: torch.Tensor
            The sequence to the encoder (required).
        memory: torch.Tensor, optional
            Optional memory, unused by default.
        mask: torch.Tensor, optional
            The mask for the src sequence (optional).
        padding_mask: torch.Tensor, optional
            The mask for the src keys per batch (optional).
            Should be True for tokens to leave untouched, and False
            for padding tokens.

        """
        output = src.transpose(0, 1)

        if self.input_size != self.d_model:
            output = self.proj(output)

        for i in range(self.num_layers):
            output = self.layers[i](output,
                                    memory=memory,
                                    src_mask=mask,
                                    padding_mask=padding_mask)

        return output.transpose(0, 1)

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
