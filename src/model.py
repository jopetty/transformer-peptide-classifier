"""Defines a transformer encoder for sequence classification."""

import math

# from torchtune import modules as tt_modules
import torch
from mamba2 import InferenceCache, Mamba2, Mamba2Config, RMSNorm
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerPeptideClassifier(nn.Module):
    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def nhead(self) -> int:
        return self._nhead

    @property
    def dim_feedforward(self) -> int:
        return self._dim_feedforward

    @property
    def dropout(self) -> float:
        return self._dropout

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def pooling_dimension(self) -> int:
        return self._pooling_dimension

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        vocab_size: int,
        pooling_dimension: int,
    ):
        super().__init__()

        # Set properties
        self._num_layers = num_layers
        self._d_model = d_model
        self._nhead = nhead
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout
        self._vocab_size = vocab_size
        self._pooling_dimension = pooling_dimension

        # Construct model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                bias=False,
            ),
            num_layers=num_layers,
        )
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, x):
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(x.size(1))

        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = x.mean(dim=self.pooling_dimension)
        x = self.classifier(x)
        return x


class SRNPeptideClassifier(nn.Module):
    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def dropout(self) -> float:
        return self._dropout

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def pooling_dimension(self) -> int:
        return self._pooling_dimension

    @property
    def bidirectional(self) -> bool:
        return self._bidirectional

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __init__(
        self,
        dropout: float,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        vocab_size: int,
        bidirectional: bool,
        pooling_dimension: int,
    ):
        super().__init__()

        # Set properties
        self._num_layers = num_layers
        self._input_size = input_size
        self._dropout = dropout
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._pooling_dimension = pooling_dimension
        self._bidirectional = bidirectional

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=input_size
        )
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
            bias=False,
        )
        self.classifier = nn.Linear(hidden_size * (2 if self.bidirectional else 1), 2)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        x = self.embedding(x)

        if h is None:
            h = x.new_zeros(
                self.num_layers * (2 if self.bidirectional else 1),
                x.size(0),
                self.hidden_size,
            )

        x, h = self.rnn(input=x, hx=h)
        x = x.mean(dim=self.pooling_dimension)
        x = self.classifier(x)
        return x


class LSTMPeptideClassifier(nn.Module):
    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def dropout(self) -> float:
        return self._dropout

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def pooling_dimension(self) -> int:
        return self._pooling_dimension

    @property
    def bidirectional(self) -> bool:
        return self._bidirectional

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __init__(
        self,
        dropout: float,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        vocab_size: int,
        bidirectional: bool,
        pooling_dimension: int,
    ):
        super().__init__()

        # Set properties
        self._num_layers = num_layers
        self._input_size = input_size
        self._dropout = dropout
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._pooling_dimension = pooling_dimension
        self._bidirectional = bidirectional

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=input_size
        )
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
            bias=False,
        )
        self.classifier = nn.Linear(hidden_size * (2 if self.bidirectional else 1), 2)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
        c: torch.Tensor | None = None,
    ):
        x = self.embedding(x)

        if h is None:
            h = x.new_zeros(
                self.num_layers * (2 if self.bidirectional else 1),
                x.size(0),
                self.hidden_size,
            )

        if c is None:
            c = x.new_zeros(
                self.num_layers * (2 if self.bidirectional else 1),
                x.size(0),
                self.hidden_size,
            )

        x, _ = self.lstm(input=x, hx=(h, c))
        x = x.mean(dim=self.pooling_dimension)
        x = self.classifier(x)
        return x


class Mamba2PeptideClassifier(nn.Module):
    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def n_layer(self) -> int:
        return self._n_layer

    @property
    def d_state(self) -> int:
        return self._d_state

    @property
    def d_conv(self) -> int:
        return self._d_conv

    @property
    def expand(self) -> int:
        return self._expand

    @property
    def headdim(self) -> int:
        return self._headdim

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def pad_vocab_size_multiple(self) -> int:
        return self._pad_vocab_size_multiple

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __init__(
        self,
        d_model: int,  # model dimension (D)
        n_layer: int = 24,  # number of Mamba-2 layers in the language model
        d_state: int = 128,  # state dimension (N)
        d_conv: int = 4,  # convolution kernel size
        expand: int = 2,  # expansion factor (E)
        headdim: int = 64,  # head dimension (P)
        chunk_size: int = 64,  # matrix partition size (Q)
        vocab_size: int = 50277,
        pad_vocab_size_multiple: int = 16,
    ):
        super().__init__()

        self._d_model = d_model
        self._n_layer = n_layer
        self._d_state = d_state
        self._d_conv = d_conv
        self._expand = expand
        self._headdim = headdim
        self._chunk_size = chunk_size
        self._vocab_size = vocab_size
        self._pad_vocab_size_multiple = pad_vocab_size_multiple

        mamba2_cfg = Mamba2Config(
            d_model=d_model,
            n_layer=n_layer,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
            vocab_size=vocab_size,
            pad_vocab_size_multiple=pad_vocab_size_multiple,
        )

        self.backbone = nn.ModuleDict(
            dict(
                embedding=nn.Embedding(mamba2_cfg.vocab_size, mamba2_cfg.d_model),
                layers=nn.ModuleList(
                    [
                        nn.ModuleDict(
                            dict(
                                mixer=Mamba2(mamba2_cfg),
                                norm=RMSNorm(mamba2_cfg.d_model),
                            )
                        )
                        for _ in range(mamba2_cfg.n_layer)
                    ]
                ),
                norm_f=RMSNorm(mamba2_cfg.d_model),
            )
        )

        self.classifier = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor, h: list[InferenceCache] | None = None):
        # print(x.shape)

        if h is None:
            h = [None for _ in range(self.n_layer)]

        x = self.backbone.embedding(x)
        for i, layer in enumerate(self.backbone.layers):
            y, h[i] = layer.mixer(layer.norm(x), h[i])
            x = y + x

        x = self.backbone.norm_f(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x
