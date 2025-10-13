# PACT — Parametric Clipping Activation for Quantized Neural Networks

> Reimplementation of **PACT** (Choi *et al.*, 2018) for CIFAR‑10 (ResNet‑20) and ImageNet (ResNet‑50/ResNet‑18), with training and evaluation pipelines in PyTorch.

**Paper:** *"PACT: Parameterized Clipping Activation for Quantized Neural Networks"* — https://arxiv.org/pdf/1805.06085

---

## Highlights

- **PACT activations** with learnable clipping parameter `α` (see `models/pact_activation.py`)
- **k‑bit quantized weights & activations** (set in config), with STE for rounding
- **ResNet backbones** with quantized conv/linear layers (`models/resnet_pact.py`, `models/quant_layers.py`)
- **Config‑driven experiments** via YAML (see `configs/`)
- **CIFAR‑10** and **ImageNet** pipelines (`utils/datasets.py`)
- **Training & evaluation CLIs** (`train.py`, `evaluate.py`) with optional confusion matrix and prediction dumps
- Simple logging, checkpointing, and results directory structure

---

## Repository layout

```
PACT-main/
├── article/
│   └── pact_article.pdf         # The original paper (for reference)
├── configs/
│   ├── cifar10_resnet20.yaml    # Example: ResNet‑20 on CIFAR‑10
│   └── imagenet_resnet50.yaml   # Example: ResNet‑50 on ImageNet
├── models/
│   ├── pact_activation.py       # PACT activation (learnable clip + k‑bit quant)
│   ├── quant_layers.py          # Quantized Conv/Linear wrappers
│   └── resnet_pact.py           # ResNet backbones with quantized layers
├── utils/
│   ├── config.py                # YAML → SimpleNamespace loader
│   ├── datasets.py              # CIFAR‑10 / SVHN / ImageNet loaders
│   └── logger.py                # Training log/Checkpoint helper
├── train.py                     # Training entrypoint
├── evaluate.py                  # Evaluation / inference entrypoint
├── requirements.txt
└── README.md                    # (this file)
```

---

## Quick start

### 1) Environment

- Python 3.9+ recommended
- PyTorch & TorchVision (match to your CUDA/cuDNN stack)
- Other deps listed in `requirements.txt`

```bash
python -m venv .venv
source .venv/bin/activate           # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> 💡 For CUDA builds of PyTorch, follow the official install selector first, then install the remaining packages from `requirements.txt`.

### 2) Datasets

- **CIFAR‑10**: will auto‑download to `~/.torch` by default (handled by `torchvision`).
- **ImageNet** (ILSVRC2012): requires manual download and layout. By default the code expects:
  - `./data/imagenet/train/` and `./data/imagenet/val/`
  - You can change the root paths in the YAML configs if you keep a different structure.

---

## Training

### CIFAR‑10 / ResNet‑20 (PACT)

```bash
python train.py --config configs/cifar10_resnet20.yaml
```

### ImageNet / ResNet‑50 (PACT)

```bash
python train.py --config configs/imagenet_resnet50.yaml
```

General notes:
- Choose the **bit‑width** in the config under `model.pact.bit_width`.
- Optimizer/LR schedule, batch size, and augmentation are all driven from the config.
- Mixed precision (`fp16`) can be toggled for ImageNet in the config (if supported by your hardware).

---

## Evaluation & inference

Evaluate a saved checkpoint and (optionally) produce a confusion matrix or dump predictions to disk.

### CIFAR‑10

```bash
python evaluate.py --config configs/cifar10_resnet20.yaml   --checkpoint results/cifar10_resnet20/best_model.pth
```

### ImageNet

```bash
python evaluate.py --config configs/imagenet_resnet50.yaml   --checkpoint results/imagenet_resnet50/best_model.pth   --confusion --dump_preds
```

Common CLI flags (see `evaluate.py` for the authoritative list):
- `--checkpoint`: path to `.pth` weights to load
- `--confusion`: also compute and save a confusion matrix
- `--dump_preds`: write class predictions to a file under the run directory

---

## Configuration

Configs live in `configs/*.yaml` and are loaded with `utils/config.py`. Typical sections:

```yaml
experiment_name: "PACT_CIFAR10_ResNet20"

dataset:
  name: cifar10            # or imagenet
  image_size: 32
  num_classes: 10

model:
  architecture: resnet20   # or resnet18 / resnet50 (per file)
  pact:
    bit_width: 4           # k‑bit activations
    init_alpha: 6.0        # initial clip value
  weights:
    bit_width: 4           # k‑bit weights

training:
  batch_size: 128
  epochs: 200
  optimizer:
    name: sgd
    lr: 0.1
    momentum: 0.9
    weight_decay: 1e-4
  lr_scheduler:
    name: cosine
    min_lr: 1e-5
  regularization:
    type: l2
    alpha_reg: true        # (optional) regularize the PACT α

logging:
  save_dir: ./results/cifar10_resnet20/
  log_interval: 50
  save_best_only: true
  plot_alpha_evolution: true
  record_bitwidth_sweep: [2,3,4,5]

hardware:
  device: cuda
  num_workers: 4
  fp16: false              # set true on ImageNet if desired
```

> If you add a new experiment, duplicate one of the example YAMLs and adjust the fields. Anything not recognized may be ignored by the current code.

---

## Results & checkpoints

By default, artifacts are written under `logging.save_dir`, e.g.:

```
results/
└── cifar10_resnet20/
    ├── best_model.pth
    ├── last_checkpoint.pth
    ├── train_log.json
    ├── metrics.csv
    ├── confusion_matrix.png         # if requested
    └── preds.csv                    # if requested
```

---

## How PACT is implemented here (high level)

- **PACT activation** (`models/pact_activation.py`): ReLU with learnable `α` that clamps activations to `[0, α]`, then performs **uniform k‑bit quantization** with a **straight‑through estimator (STE)** for backprop through the rounding op.
- **Weight quantization** (`models/quant_layers.py`): quantizes Conv/Linear weights to k bits; activations are quantized by PACT modules attached to each layer.
- **Backbones** (`models/resnet_pact.py`): ResNet‑20/18/50 variants wired with `QuantConv2d`/`QuantLinear`, optional identity activations where appropriate (e.g., post‑add in residual blocks).

For paper details and ablations, see the PDF in `article/`.

---

## Reproducing baseline numbers

Exact numbers depend on training length, augmentation, LR schedule, and bit‑width. Start from the provided configs; for reference:

- **CIFAR‑10 / ResNet‑20, 4‑bit W/A**: typically ~92–93% top‑1 with standard training.
- **ImageNet / ResNet‑50, 4‑bit W/A**: mid‑70s top‑1 with careful tuning and longer schedules.

> These are indicative only; you may need to adjust epochs, LR schedule, and data augmentation to match the paper.

---

## Troubleshooting

- **RuntimeError: CUDA out of memory** → Reduce `batch_size`, or disable `fp16` if stability is an issue.
- **ImageNet path errors** → Ensure `./data/imagenet/{train,val}/` exist, or modify the dataset root in the config.
- **Accuracy collapse at very low bit‑widths** → Warm up with higher precision; regularize/initialize `α` sensibly; try a milder LR schedule.

---

## Citation

If you use this repo, please cite the original paper:

```
@inproceedings{choi2018pact,
  title={PACT: Parameterized Clipping Activation for Quantized Neural Networks},
  author={Choi, Jungwook and Wang, Zhuo and Venkataramani, Swagath and Chuang, Ping and Srinivasan, Vijayalakshmi and Gopalakrishnan, Kailash},
  booktitle={arXiv preprint arXiv:1805.06085},
  year={2018}
}
```
