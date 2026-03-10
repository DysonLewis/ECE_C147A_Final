# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Sequence
from typing import Optional

import torch
from torch import nn


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).\

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


# ─────────────────────────────────────────────────────────────────────────────
# NEW ENCODER MODULES
# Added to support the architecture comparison experiments (cnn_gru_ctc,
# cnn_lstm_ctc, cnn_transformer_ctc, rnn_ctc, rnn_gru_ctc, etc.).
#
# Design invariant shared by ALL encoders below:
#   Input  shape: (T, N, input_size)   — standard TNC format used throughout
#   Output shape: (T, N, input_size)   — feature dim is preserved so encoders
#                                        can be stacked or swapped freely
#
# Each recurrent encoder projects the RNN output back to input_size (via a
# learned linear layer if needed) and adds a LayerNorm + residual connection.
# This mirrors the residual/norm style of TDSConvEncoder and empirically
# stabilises training, especially when combined with gradient clipping.
# ─────────────────────────────────────────────────────────────────────────────


class RNNEncoder(nn.Module):
    """Multi-layer bidirectional vanilla RNN encoder.

    Used either as a standalone temporal model (rnn_ctc) or as a shallow
    "local feature extractor" frontend before a deeper GRU/LSTM/Transformer
    backend (rnn_gru_ctc, rnn_lstm_ctc, rnn_transformer_ctc).

    The output is projected back to ``input_size`` and a residual + LayerNorm
    is applied so the feature dimension is always preserved.

    Args:
        input_size (int): Feature dimension of the input (T, N, input_size).
        hidden_size (int): Hidden units per direction in each RNN layer.
            With bidirectional=True, the raw RNN output has size
            hidden_size * 2, which is then projected to input_size.
        num_layers (int): Number of stacked RNN layers. (default: 2)
        dropout (float): Dropout between RNN layers (only applied when
            num_layers > 1). (default: 0.1)
        bidirectional (bool): Use bidirectional RNN. Strongly recommended
            for offline decoding; gives the model access to future context.
            (default: True)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 384,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        directions = 2 if bidirectional else 1
        # Dropout is only applied between layers; single-layer RNN gets 0.0
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=False,  # expects (T, N, input_size) — TNC format
        )
        rnn_out = hidden_size * directions
        # Project back to input_size if the RNN output dim differs
        self.proj = (
            nn.Linear(rnn_out, input_size) if rnn_out != input_size else nn.Identity()
        )
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, input_size)
        output, _ = self.rnn(inputs)        # (T, N, hidden_size * directions)
        output = self.proj(output)          # (T, N, input_size)
        # Residual connection stabilises gradient flow through long sequences
        return self.layer_norm(output + inputs)


class GRUEncoder(nn.Module):
    """Multi-layer bidirectional GRU encoder.

    GRU is preferred over vanilla RNN for temporal modeling: its gating
    mechanism handles long-range dependencies and avoids vanishing gradients
    more effectively.  Used as the temporal backend in cnn_gru_ctc and
    rnn_gru_ctc.

    Args:
        input_size (int): Feature dimension of the input (T, N, input_size).
        hidden_size (int): Hidden units per direction. With bidirectional=True
            the raw output size is hidden_size * 2, then projected to
            input_size. Set hidden_size = input_size // 2 to keep the feature
            dim constant without a projection. (default: 384)
        num_layers (int): Number of stacked GRU layers. (default: 2)
        dropout (float): Inter-layer dropout (ignored for num_layers=1).
            (default: 0.2)
        bidirectional (bool): Bidirectional GRU. (default: True)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 384,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        directions = 2 if bidirectional else 1
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=False,  # TNC format
        )
        gru_out = hidden_size * directions
        self.proj = (
            nn.Linear(gru_out, input_size) if gru_out != input_size else nn.Identity()
        )
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, input_size)
        output, _ = self.gru(inputs)        # (T, N, hidden_size * directions)
        output = self.proj(output)          # (T, N, input_size)
        return self.layer_norm(output + inputs)


class LSTMEncoder(nn.Module):
    """Multi-layer bidirectional LSTM encoder.

    LSTM's explicit memory cell gives it stronger capacity for long-range
    dependencies compared to GRU, at the cost of ~1.33× more parameters.
    Used as the temporal backend in cnn_lstm_ctc and rnn_lstm_ctc.

    Args:
        input_size (int): Feature dimension of the input.
        hidden_size (int): Hidden units per direction. (default: 384)
        num_layers (int): Number of stacked LSTM layers. (default: 2)
        dropout (float): Inter-layer dropout. (default: 0.2)
        bidirectional (bool): Bidirectional LSTM. (default: True)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 384,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=False,  # TNC format
        )
        lstm_out = hidden_size * directions
        self.proj = (
            nn.Linear(lstm_out, input_size) if lstm_out != input_size else nn.Identity()
        )
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, input_size)
        # LSTM returns (output, (h_n, c_n)); we only need the sequence output
        output, _ = self.lstm(inputs)       # (T, N, hidden_size * directions)
        output = self.proj(output)          # (T, N, input_size)
        return self.layer_norm(output + inputs)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding from "Attention Is All You Need"
    (Vaswani et al., 2017).  Adds fixed position signals to transformer inputs
    so the self-attention layers can reason about temporal order.

    Expects inputs of shape (T, N, d_model) — TNC format.

    Args:
        d_model (int): Model / embedding dimension.
        dropout (float): Dropout applied after adding the encoding. (default: 0.1)
        max_len (int): Maximum supported sequence length. (default: 5000)
    """

    def __init__(
        self, d_model: int, dropout: float = 0.1, max_len: int = 5000
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute the encoding table once and register as a non-parameter buffer
        position = torch.arange(max_len).unsqueeze(1)           # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)                   # (max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, N, d_model)
        T = x.size(0)
        if T <= self.pe.size(0):
            # common case: sequence fits within pre-computed table
            pe = self.pe[:T]
        else:
            # sequence longer than pre-computed table (e.g. full test sessions);
            # compute encodings on the fly for the extra positions
            d_model = self.pe.size(2)
            position = torch.arange(T, device=x.device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, device=x.device) * (-math.log(10000.0) / d_model)
            )
            pe = torch.zeros(T, 1, d_model, device=x.device)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        x = x + pe
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Multi-layer Transformer encoder with sinusoidal positional encoding.

    Uses Pre-LN (norm_first=True) transformer layers which are significantly
    more stable during early training than the original Post-LN variant —
    important for a task where the CTC loss starts very high (~170) before
    the model learns any structure.

    Used as the temporal backend in cnn_transformer_ctc and
    rnn_transformer_ctc.

    Args:
        d_model (int): Model dimension. Must equal the input feature size
            so the residual invariant is preserved. (default: 768)
        nhead (int): Number of attention heads. Must divide d_model evenly.
            Default of 8 gives head_dim = 96 for d_model = 768. (default: 8)
        num_layers (int): Number of stacked transformer encoder layers.
            (default: 4)
        dim_feedforward (int): Inner dimension of the position-wise FFN.
            Typically 2× or 4× d_model. (default: 1536)
        dropout (float): Dropout inside attention and FFN layers. (default: 0.1)
    """

    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0, (
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        )
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,   # expects (T, N, d_model) — TNC format
            norm_first=True,     # Pre-LN: much more stable gradient flow than Post-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, d_model)
        x = self.pos_encoder(inputs)    # add positional signal
        return self.transformer(x)      # (T, N, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# CNNEncoder
#
# A stack of causal-padded 1-D temporal convolutions, used as the frontend in
# CNNCTCModule (standalone CNN model) and as the frontend encoder in hybrid
# configs (cnn_gru_ctc, cnn_lstm_ctc, cnn_transformer_ctc).
#
# Design:
#   1. Input projection  num_features → cnn_channels  (pointwise Conv1d)
#   2. num_cnn_layers of:
#        Conv1d(cnn_channels, cnn_channels, kernel_size, padding=kernel_size//2)
#        → ReLU → Dropout → residual add → LayerNorm
#      padding = kernel_size // 2 keeps T constant (no temporal shrinkage),
#      so T_diff = 0 and emission_lengths == input_lengths for CTC.
#   3. Output projection cnn_channels → num_features   (pointwise Conv1d)
#
# The input/output shape is always (T, N, num_features) — TNC format.
# ─────────────────────────────────────────────────────────────────────────────
class CNNEncoder(nn.Module):
    """Stack of 1-D temporal convolution blocks with residual connections.

    Used as a drop-in CNN frontend that preserves temporal length (no
    striding), so T_diff = 0 and CTC emission lengths equal input lengths.

    Args:
        num_features (int): Input and output feature dimension. Must be
            consistent with NUM_BANDS * mlp_features[-1] upstream.
        cnn_channels (int): Number of channels in each conv layer. (default: 128)
        cnn_kernel_size (int): Temporal kernel size. Odd values recommended
            so symmetric padding keeps T constant. (default: 31)
        num_cnn_layers (int): Number of conv blocks. (default: 4)
        dropout (float): Dropout applied after ReLU in each block. (default: 0.1)
    """

    def __init__(
        self,
        num_features: int,
        cnn_channels: int = 128,
        cnn_kernel_size: int = 31,
        num_cnn_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert cnn_kernel_size % 2 == 1, "cnn_kernel_size should be odd for symmetric padding"

        # Step 1: project input features into the CNN channel space
        self.input_proj = nn.Conv1d(num_features, cnn_channels, kernel_size=1)

        # Step 2: stack of conv blocks with residual + LayerNorm
        # Each block keeps T constant via padding = kernel_size // 2
        blocks: list[nn.Module] = []
        for _ in range(num_cnn_layers):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        cnn_channels,
                        cnn_channels,
                        kernel_size=cnn_kernel_size,
                        padding=cnn_kernel_size // 2,  # symmetric: T is preserved
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        self.conv_blocks = nn.ModuleList(blocks)
        # One LayerNorm per block (over the channel dim after transposing back)
        self.norms = nn.ModuleList(
            [nn.LayerNorm(cnn_channels) for _ in range(num_cnn_layers)]
        )

        # Step 3: project back to num_features so downstream modules see the
        # same feature dimension regardless of cnn_channels
        self.output_proj = nn.Conv1d(cnn_channels, num_features, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, num_features)
        T, N, F = inputs.shape

        # Conv1d expects (N, C, L) — rearrange from TNC
        x = inputs.permute(1, 2, 0)        # (N, num_features, T)
        x = self.input_proj(x)             # (N, cnn_channels, T)

        for conv_block, norm in zip(self.conv_blocks, self.norms):
            residual = x
            x = conv_block(x)              # (N, cnn_channels, T)
            x = x + residual               # residual connection
            # LayerNorm over channel dim: permute to (N, T, C), norm, permute back
            x = norm(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = self.output_proj(x)            # (N, num_features, T)
        return x.permute(2, 0, 1)          # (T, N, num_features)
