# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
)
from emg2qwerty.transforms import Transform


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    # Feed the entire session at once without windowing/padding
                    # at test time for more realism
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )


class TDSConvCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Model
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.model = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
            # (T, N, num_classes)
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


# ─────────────────────────────────────────────────────────────────────────────
# FlexibleCTCModule
#
# A single LightningModule that supports all architecture variants used in the
# architecture comparison experiments by composing interchangeable encoder stages.
#
# Architecture:
#   SpectrogramNorm
#   → MultiBandRotationInvariantMLP   (per-band, rotation-invariant feature extraction)
#   → Flatten                         (concat left + right band features)
#   → [optional] frontend encoder    (e.g. TDSConvEncoder for local CNN features,
#                                      or RNNEncoder for recurrent feature extraction)
#   → [optional] backend encoder     (e.g. GRUEncoder, LSTMEncoder, TransformerEncoder
#                                      for longer-range sequential modelling)
#   → Linear → LogSoftmax → CTC
#
# Naming convention for model configs:
#   cnn_ctc             frontend=TDSConvEncoder,  backend=None
#   rnn_ctc             frontend=None,             backend=RNNEncoder
#   cnn_gru_ctc         frontend=TDSConvEncoder,  backend=GRUEncoder
#   cnn_lstm_ctc        frontend=TDSConvEncoder,  backend=LSTMEncoder
#   cnn_transformer_ctc frontend=TDSConvEncoder,  backend=TransformerEncoder
#   rnn_gru_ctc         frontend=RNNEncoder,       backend=GRUEncoder
#   rnn_lstm_ctc        frontend=RNNEncoder,       backend=LSTMEncoder
#   rnn_transformer_ctc frontend=RNNEncoder,       backend=TransformerEncoder
#
# All encoders are required to preserve the feature dimension (input_size ==
# output_size) so they can be swapped or stacked freely without touching the
# head.  See modules.py for implementation details.
# ─────────────────────────────────────────────────────────────────────────────
class FlexibleCTCModule(pl.LightningModule):
    """Flexible CTC module with configurable frontend + backend encoders.

    See the block comment above for the full architecture description and
    the naming convention used by the model YAML configs.

    Args:
        in_features (int): Input feature size to RotationInvariantMLP per band.
            For log-spectrogram with n_fft=64: (n_fft//2 + 1) * 16 = 528.
        mlp_features (list[int]): Output sizes for each MLP layer.
            num_features = NUM_BANDS * mlp_features[-1] (= 768 by default).
        frontend (DictConfig | None): Hydra config for the frontend encoder
            (instantiated via hydra.utils.instantiate). Pass null to skip.
        backend (DictConfig | None): Hydra config for the backend encoder.
            Pass null to skip.
        optimizer (DictConfig): Hydra config for the optimiser.
        lr_scheduler (DictConfig): Hydra config for the LR scheduler.
        decoder (DictConfig): Hydra config for the CTC decoder.
    """

    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        frontend: Optional[DictConfig] = None,
        backend: Optional[DictConfig] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # num_features is the feature dimension flowing through both encoders
        # and into the CTC head.  All encoder modules must preserve this dim.
        num_features = self.NUM_BANDS * mlp_features[-1]

        # ── Shared pre-encoder: normalisation + rotation-invariant MLP ──────
        self.spec_norm = SpectrogramNorm(
            channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS
        )
        self.mlp = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=mlp_features,
            num_bands=self.NUM_BANDS,
        )

        # ── Optional encoder stages (instantiated from Hydra DictConfig) ────
        # frontend: local / spatial feature extractor (CNN blocks or shallow RNN)
        # backend:  longer-range temporal sequential model (GRU, LSTM, Transformer)
        #
        # BUG FIX: Previous attempts passed the encoder configs directly into
        # nn.Sequential which caused shape errors and gradient flow issues.
        # Each encoder is now instantiated as a standalone nn.Module so PyTorch
        # correctly tracks its parameters and gradients.
        self.frontend_encoder = instantiate(frontend) if frontend is not None else None
        self.backend_encoder = instantiate(backend) if backend is not None else None

        # ── CTC head ─────────────────────────────────────────────────────────
        self.linear = nn.Linear(num_features, charset().num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # ── Criterion ────────────────────────────────────────────────────────
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoder = instantiate(decoder)

        # ── Metrics ──────────────────────────────────────────────────────────
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, bands=2, C=16, freq)
        x = self.spec_norm(inputs)          # (T, N, 2, 16, freq)
        x = self.mlp(x)                     # (T, N, 2, mlp_features[-1])
        x = x.flatten(start_dim=2)          # (T, N, num_features)

        # Pass through the (optional) encoder stages.
        # Each encoder preserves the (T, N, num_features) shape; only the
        # TDSConvEncoder slightly shortens T due to causal conv receptive field.
        if self.frontend_encoder is not None:
            x = self.frontend_encoder(x)    # (T', N, num_features)
        if self.backend_encoder is not None:
            x = self.backend_encoder(x)     # (T', N, num_features)

        x = self.linear(x)                  # (T', N, num_classes)
        return self.log_softmax(x)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch size

        emissions = self.forward(inputs)

        # T_diff captures any temporal shrinkage caused by the convolutional
        # receptive field (TDSConvEncoder).  For pure RNN / Transformer models
        # T_diff = 0 and emission_lengths == input_lengths.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,                    # (T, N, num_classes)
            targets=targets.transpose(0, 1),        # (N, T)
            input_lengths=emission_lengths,         # (N,)
            target_lengths=target_lengths,          # (N,)
        )

        # Decode emissions for CER computation
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update per-phase metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CNNCTCModule
#
# Standalone CNN model using the custom CNNEncoder (1-D temporal conv blocks).
# Replaces TDSConvCTCModule for the cnn_ctc experiment so that the CNN
# architecture is directly controlled by cnn_channels / cnn_kernel_size /
# num_cnn_layers rather than TDS-specific block_channels / kernel_width.
# ─────────────────────────────────────────────────────────────────────────────
from emg2qwerty.modules import CNNEncoder  # noqa: E402  (import after class defs)


class CNNCTCModule(pl.LightningModule):
    """CTC module with a stack of 1-D temporal CNN blocks as the encoder.

    Architecture:
        SpectrogramNorm
        → MultiBandRotationInvariantMLP
        → Flatten
        → CNNEncoder  (num_cnn_layers of Conv1d + residual + LayerNorm)
        → Linear → LogSoftmax → CTC

    Args:
        in_features (int): Input feature size to RotationInvariantMLP per band.
        mlp_features (list[int]): MLP layer output sizes.
        cnn_channels (int): Channels inside each CNN block. (default: 128)
        cnn_kernel_size (int): Temporal kernel size (odd). (default: 31)
        num_cnn_layers (int): Number of CNN blocks. (default: 4)
        dropout (float): Dropout in CNN blocks. (default: 0.1)
        optimizer (DictConfig): Hydra optimizer config.
        lr_scheduler (DictConfig): Hydra LR scheduler config.
        decoder (DictConfig): Hydra CTC decoder config.
    """

    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        cnn_channels: int = 128,
        cnn_kernel_size: int = 31,
        num_cnn_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        self.model = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
            CNNEncoder(
                num_features=num_features,
                cnn_channels=cnn_channels,
                cnn_kernel_size=cnn_kernel_size,
                num_cnn_layers=num_cnn_layers,
                dropout=dropout,
            ),
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)

        # CNNEncoder preserves T (no striding), so T_diff = 0
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


# ─────────────────────────────────────────────────────────────────────────────
# RNNCTCModule
#
# Standalone RNN model using the existing RNNEncoder (vanilla bidirectional
# RNN). No CNN frontend — the MLP feeds directly into the RNN.
# ─────────────────────────────────────────────────────────────────────────────
from emg2qwerty.modules import RNNEncoder as _RNNEncoder  # noqa: E402


class RNNCTCModule(pl.LightningModule):
    """CTC module with a bidirectional vanilla RNN encoder.

    Architecture:
        SpectrogramNorm
        → MultiBandRotationInvariantMLP
        → Flatten
        → RNNEncoder  (num_rnn_layers bidirectional vanilla RNN)
        → Linear → LogSoftmax → CTC

    Args:
        in_features (int): Input feature size to RotationInvariantMLP per band.
        mlp_features (list[int]): MLP layer output sizes.
        rnn_hidden_size (int): Hidden units per direction. With bidirectional=True
            raw output = rnn_hidden_size * 2, projected back to num_features.
            (default: 384)
        num_rnn_layers (int): Number of stacked RNN layers. (default: 2)
        dropout (float): Inter-layer dropout (ignored for num_rnn_layers=1).
            (default: 0.1)
        bidirectional (bool): Bidirectional RNN. (default: True)
        optimizer (DictConfig): Hydra optimizer config.
        lr_scheduler (DictConfig): Hydra LR scheduler config.
        decoder (DictConfig): Hydra CTC decoder config.
    """

    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        rnn_hidden_size: int = 384,
        num_rnn_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        self.spec_norm = SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS)
        self.mlp = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=mlp_features,
            num_bands=self.NUM_BANDS,
        )
        self.rnn_encoder = _RNNEncoder(
            input_size=num_features,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.linear = nn.Linear(num_features, charset().num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.spec_norm(inputs)
        x = self.mlp(x)
        x = x.flatten(start_dim=2)
        x = self.rnn_encoder(x)
        return self.log_softmax(self.linear(x))

    def _step(self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)

        # RNNEncoder preserves T, so T_diff = 0
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
