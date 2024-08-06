"""Defines a transformer encoder for sequence classification."""

from torch import nn
from torch.nn import functional as F
# from torchtune import modules as tt_modules
import torch
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
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
        pooling_dimension: int
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
            num_layers=num_layers
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
        pooling_dimension: int
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

        self.embedding=nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size)
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
                self.hidden_size
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
        pooling_dimension: int
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

        self.embedding=nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size)
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
        c: torch.Tensor | None = None
    ):
        x = self.embedding(x)

        if h is None:
            h = x.new_zeros(
                self.num_layers * (2 if self.bidirectional else 1), 
                x.size(0), 
                self.hidden_size
            )
        
        if c is None:
            c = x.new_zeros(
                self.num_layers * (2 if self.bidirectional else 1), 
                x.size(0), 
                self.hidden_size
            )

        x, _ = self.lstm(input=x, hx=(h, c))
        x = x.mean(dim=self.pooling_dimension)
        x = self.classifier(x)
        return x