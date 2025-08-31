#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# File name: ObsCure_MNIST.py
# Author: "RF 2011-1 ft. RF 1996-17 || 2 AF/AS/MJL"
# Date created: 2025-08-31
# Version = "1.0"
# License =  "CC0 1.0"
# Listening = "Senza Una Donna - Zucchero ft. Paul Young."
# =============================================================================
""" Experimenting with MNIST. Nothing exotic, just playing with useful, perhaps
 a tad underused techniques.
Ranked 12th on Kaggle (99.9% accuracy - within top 5 scores, densely ranked).

Training script for a compact WideSmallResNet on MNIST with GhostBatchNorm,
MixUp/CutMix augmentation, Lookahead optimizer wrapper and SWA support.

Includes:
- Dataset/transform setup
- Model definition (GhostBatchNorm2d, BasicBlock, WideSmallResNet)
- MixUp and CutMix utilities
- Lookahead optimizer wrapper
- Trainer class with mixed-precision training, SWA, TTA evaluation, checkpointing,
  and plotting utilities
"""
# =============================================================================


# Standard libraries
import os
import re
import glob
import json
import math
import random
from pathlib import Path
from typing import List, Tuple, Optional

# Third-party libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Torch optimizers / schedulers / utilities
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

# AMP (automatic mixed precision)
from torch.amp import autocast


# -------------------------
# Config
# -------------------------
DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")

SEED: int = 42
BATCH_SIZE: int = 256
GHOST_BATCH: int = 32
GHOST_BN_UPDATE_BATCH: int = 512
NUM_CLASSES: int = 10

INITIAL_EPOCHS: int = 1
EXTRA_EPOCHS: int = 1
TOTAL_EPOCHS: int = INITIAL_EPOCHS + EXTRA_EPOCHS
RESUME: bool = False
CHECKPOINT_PATH: str = "checkpoint_epoch100.pth"

MIXPROB: float = 0.102
MIXUP_ALPHA: float = 0.091
CUTMIX_BETA: float = 0.35
USE_CUTMIX_PROB: float = 0.8

FINAL_FRAC: float = 0.25
SWA_START: int = int(TOTAL_EPOCHS * 0.80)

BASE_LR: float = 0.01
RESUME_LR: float = 5e-4
ETA_MIN: float = 1e-6
MOMENTUM: float = 0.9
WEIGHT_DECAY: float = 1.8e-5

CHKPT_EPOCH_RE = re.compile(r"epoch(\d+)\.pth$")
TTA_RUNS: int = 5
SAVE_PREFIX: str = f"mnist_seed{SEED}"
NUM_WORKERS: int = 4

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# -------------------------
# Transforms & datasets
# -------------------------
mean, std = (0.1307,), (0.3081,)


train_transform_strong = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomRotation(8),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15))
])
train_transform_light = transforms.Compose([
    transforms.RandomCrop(28, padding=2),
    transforms.RandomRotation(4),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_ds_strong = datasets.MNIST(
    '.', train=True, download=True, transform=train_transform_strong)
train_ds_light = datasets.MNIST(
    '.', train=True, download=True, transform=train_transform_light)
train_ds_eval = datasets.MNIST(
    '.', train=True, download=True, transform=test_transform)
test_ds = datasets.MNIST(
    '.', train=False, download=True, transform=test_transform)

train_loader_strong: DataLoader = DataLoader(train_ds_strong, batch_size=BATCH_SIZE, shuffle=True,
                                             num_workers=NUM_WORKERS, pin_memory=True)
train_loader_light: DataLoader = DataLoader(train_ds_light, batch_size=BATCH_SIZE, shuffle=True,
                                            num_workers=NUM_WORKERS, pin_memory=True)
bn_update_loader: DataLoader = DataLoader(train_ds_eval, batch_size=GHOST_BN_UPDATE_BATCH, shuffle=False,
                                          num_workers=2, pin_memory=True)
test_loader: DataLoader = DataLoader(test_ds, batch_size=512,
                                     shuffle=False, num_workers=2, pin_memory=True)


# -------------------------
# Model / GhostBatchNorm (vectorized)
# -------------------------
class GhostBatchNorm2d(nn.Module):
    """Vectorized Ghost BatchNorm module.

    This module applies BatchNorm separately to "ghost" sub-batches by reshaping
    the incoming tensor. Works the same as nn.BatchNorm2d when not training or
    when the batch size is <= ghost_batch.

    Args:
        num_features: Number of feature channels.
        ghost_batch: Virtual batch size for computing batch statistics.
        eps: Value added to denominator for numerical stability.
        momentum: Momentum for running statistics.
    """

    def __init__(self, num_features: int, ghost_batch: int = 32, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.ghost: int = max(1, ghost_batch)
        self.bn: nn.BatchNorm2d = nn.BatchNorm2d(
            num_features, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Normalized tensor of same shape as input.
        """
        if not self.training or x.size(0) <= self.ghost:
            return self.bn(x)

        n = x.size(0)
        chunks = (n + self.ghost - 1) // self.ghost
        pad = chunks * self.ghost - n
        if pad:
            pad_tensor = x[-1:].expand(pad, -1, -1, -1)
            x_padded = torch.cat([x, pad_tensor], dim=0)
        else:
            x_padded = x
        x_reshaped = x_padded.view(
            chunks, self.ghost, *x_padded.shape[1:]).reshape(-1, *x_padded.shape[1:])
        out = self.bn(x_reshaped)
        out = out.view(chunks, self.ghost, *
                       out.shape[1:]).reshape(-1, *out.shape[1:])[:n]
        return out


class BasicBlock(nn.Module):
    """Residual basic block using two 3x3 convs and GhostBatchNorm.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        stride: Stride for the first convolution.
        ghost_batch: Virtual batch size for GhostBatchNorm.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, ghost_batch: int = 32) -> None:
        super().__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1: GhostBatchNorm2d = GhostBatchNorm2d(out_ch, ghost_batch)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)
        self.conv2: nn.Conv2d = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2: GhostBatchNorm2d = GhostBatchNorm2d(out_ch, ghost_batch)
        self.down: Optional[nn.Module] = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1,
                          stride=stride, bias=False),
                GhostBatchNorm2d(out_ch, ghost_batch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the basic block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after residual addition and activation.
        """
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.down is not None:
            identity = self.down(identity)
        out = out + identity
        return self.relu(out)


class WideSmallResNet(nn.Module):
    """Compact wide ResNet-like architecture for small inputs (e.g., MNIST).

    Args:
        block: Block class (e.g., BasicBlock).
        layers: Number of blocks per stage.
        channels: Tuple of channel widths for stages.
        ghost_batch: Virtual batch size for GhostBatchNorm.
        num_classes: Number of output classes.
    """

    def __init__(self, block, layers: List[int], channels: Tuple[int, int, int] = (32, 64, 128),
                 ghost_batch: int = 32, num_classes: int = 10) -> None:
        super().__init__()
        self.in_ch: int = channels[0]
        self.conv1: nn.Conv2d = nn.Conv2d(
            1, self.in_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1: GhostBatchNorm2d = GhostBatchNorm2d(self.in_ch, ghost_batch)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)

        self.layer1: nn.Sequential = self._make_layer(
            block, channels[0], layers[0], stride=1, ghost_batch=ghost_batch)
        self.layer2: nn.Sequential = self._make_layer(
            block, channels[1], layers[1], stride=2, ghost_batch=ghost_batch)
        self.layer3: nn.Sequential = self._make_layer(
            block, channels[2], layers[2], stride=2, ghost_batch=ghost_batch)

        self.avgpool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)
        self.fc: nn.Linear = nn.Linear(channels[2], num_classes)

    def _make_layer(self, block, out_ch: int, blocks: int, stride: int, ghost_batch: int) -> nn.Sequential:
        """Create a sequential layer consisting of `blocks` BasicBlocks."""
        layers: List[nn.Module] = []
        for i in range(blocks):
            s = stride if i == 0 else 1
            layers.append(block(self.in_ch, out_ch, s, ghost_batch))
            self.in_ch = out_ch
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the network.

        Args:
            x: Input tensor of shape (B, 1, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x).view(x.size(0), -1)
        return self.fc(x)

def build_model(device: torch.device = DEVICE) -> nn.Module:
    """Build and return the model on the requested device.

    Args:
        device: Device to move the model to.

    Returns:
        Instantiated and device-cast nn.Module.
    """
    return WideSmallResNet(BasicBlock, layers=[3, 3, 3], channels=(32, 64, 128),
                           ghost_batch=GHOST_BATCH, num_classes=NUM_CLASSES).to(device)


# -------------------------
# MixUp / CutMix utilities
# -------------------------
def rand_bbox(size: torch.Size, lam: float) -> Tuple[int, int, int, int]:
    """Generate a random bounding box for CutMix.

    Args:
        size: Tensor size, expected (B, C, H, W).
        lam: Lambda (proportion) controlling box area.

    Returns:
        Tuple of (bbx1, bby1, bbx2, bby2) coordinates.
    """
    W, H = size[3], size[2]  # size = (B, C, H, W)
    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W - 1)
    cy = random.randint(0, H - 1)
    bbx1 = max(0, cx - cut_w // 2)
    bby1 = max(0, cy - cut_h // 2)
    bbx2 = min(W, cx + cut_w // 2)
    bby2 = min(H, cy + cut_h // 2)
    return bbx1, bby1, bbx2, bby2

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Perform MixUp on a batch.

    Args:
        x: Input tensor (B, C, H, W).
        y: Target labels (B,).
        alpha: Beta distribution alpha parameter.

    Returns:
        Tuple of (mixed_x, y, y_shuffled, lambda).
    """
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

def cutmix_data(x: torch.Tensor, y: torch.Tensor, beta: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Perform CutMix on a batch.

    Args:
        x: Input tensor (B, C, H, W).
        y: Target labels (B,).
        beta: Beta distribution beta parameter.

    Returns:
        Tuple of (cutmixed_x, y, y_shuffled, adjusted_lambda).
    """
    if beta <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(beta, beta))
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.shape, lam)
    x_cut = x.clone()
    x_cut[:, :, bby1:bby2, bbx1:bbx2] = x[idx, :, bby1:bby2, bbx1:bbx2]
    area = (bbx2 - bbx1) * (bby2 - bby1)
    lam_adj = 1 - area / (x.size(2) * x.size(3))
    return x_cut, y, y[idx], lam_adj


# -------------------------
# Lookahead optimizer
# -------------------------
class Lookahead(torch.optim.Optimizer):
    """Lookahead optimizer wrapper.

    Wraps a base optimizer and maintains "slow" weights that are periodically
    interpolated with the base optimizer's fast weights.

    Args:
        base_optimizer: An instance of torch.optim.Optimizer to wrap.
        k: Number of steps between slow weight updates.
        alpha: Interpolation factor for slow updates.

    Raises:
        ValueError: If base_optimizer is not an Optimizer.
    """

    def __init__(self, base_optimizer: torch.optim.Optimizer, k: int = 5, alpha: float = 0.5) -> None:
        if not isinstance(base_optimizer, torch.optim.Optimizer):
            raise ValueError(
                "base_optimizer must be an instance of torch.optim.Optimizer")
        self.base_optimizer: torch.optim.Optimizer = base_optimizer
        self.k: int = int(k)
        self.alpha: float = float(alpha)
        self._step: int = 0
        # expose param_groups for convenience
        self.param_groups = self.base_optimizer.param_groups

        # initialize slow params in optimizer state
        for group in self.param_groups:
            for p in group['params']:
                self.base_optimizer.state[p]['slow_param'] = p.data.clone(
                ).detach()

    def zero_grad(self) -> None:
        """Zero gradients on the base optimizer."""
        self.base_optimizer.zero_grad()

    def step(self, closure=None):
        """Step the base optimizer, and update slow weights every k steps.

        Args:
            closure: Optional closure for optimizers that support it.

        Returns:
            The return value of the base optimizer.step().
        """
        loss = self.base_optimizer.step(closure)
        self._step += 1
        if self._step % self.k == 0:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.base_optimizer.state[p]
                    slow = state.get('slow_param', None)
                    if slow is None:
                        slow = p.data.clone().detach()
                    new_slow = slow + self.alpha * (p.data - slow)
                    state['slow_param'] = new_slow.clone()
                    p.data.copy_(new_slow)
        return loss

    def state_dict(self) -> dict:
        """Return the wrapped optimizer's state dict (contains slow params)."""
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state dict into the wrapped base optimizer.

        Args:
            state_dict: State dict from a compatible optimizer.
        """
        self.base_optimizer.load_state_dict(state_dict)


# -------------------------
# Trainer encapsulation
# -------------------------
class Trainer:
    """Training harness for model, data, and scheduling.

    Attributes:
        device: Device used for training.
        model: The neural network model.
        criterion: Loss function.
        base_opt: Base optimizer (SGD).
        optimizer: Lookahead wrapped optimizer.
        scheduler: Learning rate scheduler operating on base optimizer.
        swa_model: AveragedModel for SWA.
        history: Training history dictionary.
    """

    def __init__(self) -> None:
        self.device: torch.device = DEVICE
        self.model: nn.Module = build_model(self.device)
        self.criterion: nn.Module = nn.CrossEntropyLoss()
        self.base_opt: torch.optim.Optimizer = SGD(self.model.parameters(), lr=BASE_LR,
                                                   momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        self.optimizer: Lookahead = Lookahead(self.base_opt, k=5, alpha=0.25)
        self.remaining_epochs: int = EXTRA_EPOCHS
        self.scheduler: CosineAnnealingLR = CosineAnnealingLR(self.base_opt, T_max=max(
            1, self.remaining_epochs), eta_min=ETA_MIN)
        self.swa_model: AveragedModel = AveragedModel(self.model)
        self.swa_started: bool = False
        self.swa_scheduler = None
        self.scaler = torch.amp.GradScaler()  # mixed precision

        self.best_val: float = 0.0
        self.start_epoch: int = 1

        self.history: dict = {'epoch': [],
                              'train_loss': [], 'train_acc': [], 'val_acc': []}
        self.results_dir: Path = Path("results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_load_state(model: nn.Module, state_dict: dict) -> None:
        """Load a state dict into a model with a fallback for 'module.' prefixes.

        Args:
            model: The model to load state into.
            state_dict: State dict possibly containing a 'module.' prefix.
        """
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            new_state = {}
            for k, v in state_dict.items():
                nk = k.replace('module.', '') if k.startswith('module.') else k
                new_state[nk] = v
            model.load_state_dict(new_state)

    def save_checkpoint(self, epoch: int, swa: bool = False) -> str:
        """Save a checkpoint (regular or SWA) to disk.

        Args:
            epoch: Epoch number to include in filename.
            swa: Whether to save the SWA model state.

        Returns:
            The filesystem path of the saved checkpoint.
        """
        name: str = f"{SAVE_PREFIX}_{'swa' if swa else 'ckpt'}_epoch{epoch}.pth"
        state = {
            'epoch': epoch,
            'model_state_dict': (self.swa_model.module.state_dict() if swa and hasattr(self.swa_model, 'module') else (self.swa_model.state_dict() if swa else (self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()))),
            'optimizer_state_dict': self.optimizer.state_dict(),
            "val_acc": self.history.get("val_acc"),
            "train_loss": self.history.get("train_loss"),
        }
        torch.save(state, name)
        print("Saved", name)
        return name

    def load_checkpoint(self, ckpt_path: str, load_optimizer: bool = False) -> None:
        """Load a checkpoint from disk into the trainer.

        Args:
            ckpt_path: Path to checkpoint file.
            load_optimizer: Whether to restore optimizer state.

        Notes:
            If optimizer state fails to load it will continue with a fresh optimizer.
        """
        d = torch.load(ckpt_path, map_location=self.device)
        st = d.get('model_state_dict', d)
        self._safe_load_state(self.model, st)
        if load_optimizer and 'optimizer_state_dict' in d:
            try:
                self.optimizer.load_state_dict(d['optimizer_state_dict'])
            except Exception:
                print(
                    "Failed to load optimizer state fully — continuing with fresh optimizer.")
        self.start_epoch = d.get('epoch', self.start_epoch)

    def train_one_epoch(self, loader: DataLoader, use_mix: bool) -> Tuple[float, float]:
        """Train model for a single epoch.

        Args:
            loader: DataLoader providing (input, target) batches.
            use_mix: Whether to use MixUp/CutMix augmentation probabilistically.

        Returns:
            Tuple of (average_loss, accuracy_percent).
        """
        self.model.train()
        running_loss: float = 0.0
        correct: int = 0
        total: int = 0
        for xb, yb in loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            if use_mix and random.random() < MIXPROB:
                if random.random() < USE_CUTMIX_PROB:
                    inputs, a, b, lam = cutmix_data(xb, yb, CUTMIX_BETA)
                    with autocast(str(DEVICE)):
                        preds = self.model(inputs)
                        loss = lam * \
                            self.criterion(preds, a) + (1 - lam) * \
                            self.criterion(preds, b)
                else:
                    inputs, a, b, lam = mixup_data(xb, yb, MIXUP_ALPHA)
                    with autocast(str(DEVICE)):
                        preds = self.model(inputs)
                        y_a = F.one_hot(a, NUM_CLASSES).float()
                        y_b = F.one_hot(b, NUM_CLASSES).float()
                        soft = lam * y_a + (1 - lam) * y_b
                        loss = -(soft.to(self.device) *
                                 F.log_softmax(preds, dim=1)).sum(dim=1).mean()
            else:
                with autocast(str(DEVICE)):
                    preds = self.model(xb)
                    loss = self.criterion(preds, yb)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Step the base optimizer once via the GradScaler
            self.scaler.step(self.base_opt)
            self.scaler.update()

            # Perform Lookahead's slow-weight interpolation WITHOUT re-running base_optimizer.step()
            # (increment the internal counter and update slow params manually)
            self.optimizer._step += 1
            if self.optimizer._step % self.optimizer.k == 0:
                for group in self.optimizer.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        state = self.base_opt.state[p]
                        slow = state.get('slow_param', None)
                        if slow is None:
                            slow = p.data.clone().detach()
                        new_slow = slow + \
                            self.optimizer.alpha * (p.data - slow)
                        state['slow_param'] = new_slow.clone()
                        p.data.copy_(new_slow)

            running_loss += loss.item() * xb.size(0)
            total += xb.size(0)
            _, predicted = preds.detach().max(1)
            correct += predicted.eq(yb).sum().item()

        return running_loss / total, 100.0 * correct / total

    @torch.no_grad()
    def test_tta(self, model_eval: nn.Module, tta: int = 5) -> float:
        """Evaluate model with simple test-time augmentation (random shifts).

        Args:
            model_eval: Model to evaluate (can be SWA model).
            tta: Number of TTA samples per input.

        Returns:
            Accuracy percentage on test set.
        """
        model_eval.eval()
        correct: int = 0
        total: int = 0
        for xb, yb in test_loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            probs_sum = torch.zeros(
                xb.size(0), NUM_CLASSES, device=self.device)
            # generate random shifts in batch to vectorize a bit
            for _ in range(tta):
                sx = random.randint(-2, 2)
                sy = random.randint(-2, 2)
                xb_t = torch.roll(xb, shifts=(sx, sy), dims=(2, 3))
                preds = model_eval(xb_t)
                probs_sum += F.softmax(preds, dim=1)
            probs = probs_sum / float(tta)
            _, predicted = probs.max(1)
            correct += predicted.eq(yb).sum().item()
            total += xb.size(0)
        return 100.0 * correct / total

    def update_bn_for_swa(self) -> None:
        """Update batch-norm statistics for the SWA model using the training set."""
        print("Updating BN for SWA model using full train set...")
        self.swa_model.to(self.device)
        update_bn(bn_update_loader, self.swa_model, device=self.device)

    def train(self, start_epoch: int = 1, total_epochs: int = TOTAL_EPOCHS) -> None:
        """Full training loop including SWA and periodic evaluation.

        Args:
            start_epoch: Epoch number to start training from.
            total_epochs: Number of epochs to train (inclusive).
        """
        for epoch in range(start_epoch, total_epochs + 1):
            is_final_phase = (epoch > total_epochs * (1.0 - FINAL_FRAC))
            loader = train_loader_light if is_final_phase else train_loader_strong
            use_mix = not is_final_phase

            loss, acc = self.train_one_epoch(loader, use_mix=use_mix)

            # scheduler step per epoch (cosine anneal over remaining epochs)
            self.scheduler.step()

            print(f"Epoch {epoch}: train loss {loss:.4f} train acc {acc:.3f}")

            # SWA
            if epoch >= SWA_START:
                if not self.swa_started:
                    print("SWA start at epoch", epoch)
                    self.swa_started = True
                    self.swa_scheduler = SWALR(self.base_opt, swa_lr=0.01)
                self.swa_model.update_parameters(self.model)
                if self.swa_scheduler is not None:
                    self.swa_scheduler.step()

            # periodic eval
            val_acc = self.test_tta(self.model, tta=1)
            print(f"Epoch {epoch}: val acc {val_acc:.4f}")
            if val_acc > self.best_val:
                self.best_val = val_acc
                self.save_checkpoint(epoch, swa=False)
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(loss)
            self.history['train_acc'].append(acc)
            self.history['val_acc'].append(val_acc)

        # finalize
        if self.swa_started:
            self.update_bn_for_swa()
            swa_acc = self.test_tta(self.swa_model, tta=TTA_RUNS)
            print(f"SWA final (TTA={TTA_RUNS}) acc: {swa_acc:.4f}")
            self.save_checkpoint(total_epochs, swa=True)
        else:
            final_acc = self.test_tta(self.model, tta=TTA_RUNS)
            print(f"Final model (TTA={TTA_RUNS}) acc: {final_acc:.4f}")
            self.save_checkpoint(total_epochs, swa=False)

    def save_history(self) -> None:
        """Save training history JSON to results directory."""
        path: Path = self.results_dir / f"{SAVE_PREFIX}_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f)
        print("Saved history to", path)

    def plot_learning_curves(self) -> None:
        """Plot and save training loss and accuracy curves based on history."""
        epochs = self.history['epoch']
        if not epochs:
            print("No history to plot.")
            return
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.history['train_loss'],
                 label='Train Loss', color='C0')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        plt.tight_layout()
        p1 = self.results_dir / f"{SAVE_PREFIX}_train_loss.png"
        plt.savefig(p1)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.history['train_acc'],
                 label='Train Acc', color='C1')
        plt.plot(epochs, self.history['val_acc'], label='Val Acc', color='C2')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Train / Val Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        p2 = self.results_dir / f"{SAVE_PREFIX}_acc.png"
        plt.savefig(p2)
        plt.close()
        print("Saved plots to", p1, "and", p2)


# -------------------------
# Ensemble utilities
# -------------------------
def load_state_into_model(model_obj: nn.Module, ckpt_path: str, device: torch.device = DEVICE) -> None:
    """Load a checkpoint state dict into a model with 'module.' prefix handling.

    Args:
        model_obj: Model instance to load weights into.
        ckpt_path: Path to checkpoint file.
        device: Device for map_location when loading.
    """
    sd = torch.load(ckpt_path, map_location=device)
    state = sd.get('model_state_dict', sd)
    try:
        model_obj.load_state_dict(state)
    except RuntimeError:
        new_state = {}
        for k, v in state.items():
            nk = k.replace('module.', '') if k.startswith('module.') else k
            new_state[nk] = v
        model_obj.load_state_dict(new_state)

def _ckpt_epoch_from_name(path: str) -> int:
    m = CHKPT_EPOCH_RE.search(path)
    return int(m.group(1)) if m else -1

def select_top_k(paths: List[str], k: int = 3) -> List[str]:
    """Return top-k checkpoint paths sorted by:
       1) val_acc desc (higher better)
       2) train_loss asc (lower better)
       3) epoch desc (later epoch wins)
    Missing val_acc -> -inf, missing train_loss -> +inf (ranked worst for ties).
    """
    entries: List[Tuple[float, float, int, str]] = []
    for p in paths:
        try:
            sd = torch.load(p, map_location='cpu')
            val = sd.get("val_acc")
            train_loss = sd.get("train_loss")
            val_num = float(val) if val is not None else float("-inf")
            loss_num = float(
                train_loss) if train_loss is not None else float("inf")
        except Exception:
            val_num = float("-inf")
            loss_num = float("inf")
        epoch = _ckpt_epoch_from_name(p)
        entries.append((val_num, loss_num, epoch, p))

    # sort: val_desc, loss_asc, epoch_desc
    entries.sort(key=lambda x: (-x[0], x[1], -x[2]))
    selected = [e[3] for e in entries[:k]]
    return selected

@torch.no_grad()
def predict_probs_from_ckpt(ckpt_path: str, tta: int = 5) -> torch.Tensor:
    """Predict class probabilities for the test set from a single checkpoint.

    Args:
        ckpt_path: Checkpoint file path.
        tta: Number of TTA augmentations to average.

    Returns:
        Concatenated tensor of probabilities for the full test set.
    """
    m: nn.Module = build_model(DEVICE)
    load_state_into_model(m, ckpt_path)
    m.eval()
    probs_list: List[torch.Tensor] = []
    for xb, _ in test_loader:
        xb = xb.to(DEVICE)
        probs_sum = torch.zeros(xb.size(0), NUM_CLASSES, device=DEVICE)
        for _ in range(tta):
            sx = random.randint(-2, 2)
            sy = random.randint(-2, 2)
            xb_t = torch.roll(xb, shifts=(sx, sy), dims=(2, 3))
            preds = m(xb_t)
            probs_sum += F.softmax(preds, dim=1)
        probs_list.append((probs_sum / float(tta)).cpu())
    return torch.cat(probs_list, dim=0)

def evaluate_ensemble(checkpoint_paths: List[str], tta: int = 5) -> float:
    """Evaluate an ensemble of checkpoints on the test set.

    Args:
        checkpoint_paths: List of checkpoint file paths.
        tta: Number of TTA augmentations used per model.

    Returns:
        Ensemble accuracy percentage.
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoints provided.")
    agg: torch.Tensor = None  # type: ignore
    for p in checkpoint_paths:
        probs = predict_probs_from_ckpt(p, tta=tta)
        agg = probs if agg is None else agg + probs
    agg /= float(len(checkpoint_paths))
    all_labels = torch.tensor([y for _, y in test_ds]).long()
    _, preds = agg.max(1)
    correct = preds.cpu().eq(all_labels).sum().item()
    acc: float = 100.0 * correct / len(all_labels)
    print(
        f"Ensemble ({len(checkpoint_paths)} models, TTA={tta}) acc: {acc:.4f}")
    return acc


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    trainer = Trainer()
    if RESUME and os.path.exists(CHECKPOINT_PATH):
        trainer.load_checkpoint(CHECKPOINT_PATH, load_optimizer=False)
        trainer.start_epoch = trainer.start_epoch + 1

    trainer.train(start_epoch=trainer.start_epoch, total_epochs=TOTAL_EPOCHS)

    trainer.save_history()
    trainer.plot_learning_curves()

    # gather checkpoints
    ckpt_pattern = f"{SAVE_PREFIX}_ckpt_epoch*.pth"
    swa_pattern = f"{SAVE_PREFIX}_swa_epoch*.pth"
    ckpts = sorted(glob.glob(ckpt_pattern))
    swas = sorted(glob.glob(swa_pattern))

    # evaluate ensemble
    all_ckpts = ckpts + swas
    if not all_ckpts:
        raise SystemExit("No checkpoints found.")

    selected = select_top_k(all_ckpts)
    print("Selected top:", selected)
    evaluate_ensemble(selected, tta=TTA_RUNS)
