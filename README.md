# ObsCure MNIST
**Missing those classic handwritten digits.**

![Banner Image](<img/tin_drum.png> "a post office scene. MNIST was designed to tackle the challenge of the digitization of handwritten digits number.")
<br>*Post office scene ("The tin drum" by Volker Schlöndorff, 1979).*

>**The origins:** in the late 1980s, the US Census Bureau was interested in automatic digitization of handwritten census forms. Over time MNIST became the de facto starting point for evaluating new machine learning architectures and training techniques. More on [Wikipedia](https://en.wikipedia.org/wiki/MNIST)

This repo contains the source of a submission to the [Kaggle MNIST competition](https://www.kaggle.com/competitions/digit-recognizer). This project grew out of a few productive rainy afternoons spent refining a compact, ResNet-inspired PyTorch model with MixUp/CutMix, GhostBatchNorm, Lookahead, AMP, and SWA.

## Quick start

1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Train:

```bash
python3 ObsCure_MNIST.py
```

## Components

- **Model:** WideSmallResNet (BasicBlock) with GhostBatchNorm2d.
- **Augmentation:** strong/light transforms, MixUp and CutMix utilities.
- **Optimizers:** SGD base optimizer wrapped by a Lookahead implementation.
- **Mixed precision:** torch.amp GradScaler + autocast used in training.
- **SWA:** torch.optim.swa_utils AveragedModel and SWALR used in later epochs.
- *Utilities:* ensemble evaluation, test-time augmentation (TTA), BN update for SWA.

## Architecture

<details>
<summary>Contents - click to expand</summary>

![Banner Image](</img/model_graph.png> "ObsCure_MNIST model architecture.")

</details>

## Configuration

<details>
<summary>Contents - click to expand</summary>

| Parameter | Value | Purpose / Explanation |
|---|---:|---|
| **DEVICE** | torch.device("cuda" if torch.cuda.is_available() else "cpu") | Specifies where tensors and model run: GPU if available, otherwise CPU. |
| **SEED** | 42 | Random seed for reproducibility (controls RNG for torch, numpy, etc.). |
| **BATCH_SIZE** | 256 | Number of samples per training batch. Larger batches speed throughput but use more memory. |
| **GHOST_BATCH** | 32 | Mini-batch size used inside Ghost Batch Normalization to simulate smaller-batch statistics within a large BATCH_SIZE. |
| **GHOST_BN_UPDATE_BATCH** | 512 | Batch size used when updating Ghost BatchNorm running statistics (e.g., a larger aggregate used for more stable updates). |
| **NUM_CLASSES** | 10 | Number of target classes (MNIST digits 0–9). |
| **INITIAL_EPOCHS** | 2 | Initial phase epochs (e.g., warmup or base training). |
| **EXTRA_EPOCHS** | 2 | Additional training epochs (e.g., fine-tuning or further training). |
| **TOTAL_EPOCHS** | INITIAL_EPOCHS + EXTRA_EPOCHS (4) | Total number of training epochs. |
| **RESUME** | False | Whether to resume training from a checkpoint. |
| **CHECKPOINT_PATH** | "checkpoint_epoch100.pth" | Path to checkpoint file to load when RESUME is True. |
| **MIXPROB** | 0.102 | Overall probability of applying a mix augmentation (mixup or cutmix) to a batch. |
| **MIXUP_ALPHA** | 0.091 | Alpha parameter for Beta distribution when sampling mixup interpolation coefficient. |
| **CUTMIX_BETA** | 0.35 | Beta parameter for Beta distribution when sampling cutmix area ratios. |
| **USE_CUTMIX_PROB** | 0.8 | Given that a mix augmentation is applied, probability of choosing CutMix vs MixUp (0.8 → 80% CutMix, 20% MixUp). |
| **FINAL_FRAC** | 0.25 | Fraction of training near the end for special schedules/behavior (commonly final LR fraction or final epochs fraction for SWA). |
| **SWA_START** | int(TOTAL_EPOCHS * 0.80) → 3 | Epoch to start Stochastic Weight Averaging (SWA). With TOTAL_EPOCHS=4, SWA starts at epoch 3. |
| **BASE_LR** | 0.01 | Base learning rate for optimizer/scheduler. |
| **RESUME_LR** | 5e-4 | Learning rate to use when resuming training from checkpoint. |
| **ETA_MIN** | 1e-6 | Minimum learning rate for cosine/annealing schedulers. |
| **MOMENTUM** | 0.9 | Momentum term for SGD optimizer. |
| **WEIGHT_DECAY** | 1.8e-5 | L2 regularization coefficient applied to weights. |
| **TTA_RUNS** | 5 | Number of Test Time Augmentation runs to average predictions for evaluation. |
| **SAVE_PREFIX** | f"mnist_seed{SEED}" → "mnist_seed42" | Prefix used when saving models/checkpoints/outputs. |
| **NUM_WORKERS** | 4 | Number of subprocesses for data loading (DataLoader num_workers). |

>[!IMPORTANT]
>Hyperparameter optimization achieved with [Optuna](optuna.org).

</details>

## Output

- **Better checkpoints:** PREFIX_ckpt_epoch{N}.pth and PREFIX_swa_epoch{N}.pth
- **History JSON:** results/PREFIX_history.json
- **Plots:** results/PREFIX_train_loss.png and results/PREFIX_acc.png

## Training results

![Banner Image](</img/training.png> "training curves for ObsCure_MNIST.")

**Best overall checkpoint:**

```
train loss 0.0675 train acc 97.188
val acc 99.7800
```

>[!NOTE]
>Ranked 12th (top 5 best scores) on Kaggle MNIST leaderbord with a **99.9%** accuracy with the platform's validation dataset.

## License

The source code is provided under the [CC0](https://creativecommons.org/public-domain/cc0/) license. See the [LICENSE](/LICENSE) file for details.