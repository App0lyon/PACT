# PACT Baseline Training

Full-precision baselines that mirror the setups in the PACT paper (arXiv:1805.06085). Hyper-parameters are baked in to match the referenced training recipes; the only change you will make later is swapping ReLU ? PACT.

## Dependencies
pip install -r requirements.txt

## Datasets
- CIFAR-10 will auto-download to `--data-root`.
- SVHN will auto-download (train + extra are concatenated).
- ImageNet: if `train/` and `val/` are missing under `--data-root`, training will download `ILSVRC/imagenet-1k` via Hugging Face Datasets (requires `hf auth login` and `datasets>=2.14` for `hf://` paths) into `--data-root/hf_cache`. Otherwise it uses the standard folder layout:
  - `train/` and `val/` subdirectories with class folders.

## Training commands

CIFAR-10 ResNet-20 (SGD 0.1, momentum 0.9, wd 2e-4, milestones 60/120, 200 epochs):
```
python train.py --dataset cifar10 --output-dir runs/cifar10
```

SVHN 7-layer ConvNet (Adam 1e-3, eps 1e-5, wd 1e-7, step 0.5 every 50 epochs, 200 epochs):
```
python train.py --dataset svhn --output-dir runs/svhn
```

ImageNet AlexNet+BN (Adam 1e-4, eps 1e-5, wd 5e-6, milestones 56/64, 100 epochs):
```
python train.py --dataset imagenet --model alexnet_bn --batch-size 128 --workers 8 --output-dir runs/imagenet_alexnet
```

ImageNet ResNet-18/50 (pre-activation, SGD 0.1, momentum 0.9, wd 1e-4, milestones 30/60/85/95, 110 epochs):
```
python train.py --dataset imagenet --model resnet18_preact --batch-size 256 --workers 8 --output-dir runs/imagenet_r18
python train.py --dataset imagenet --model resnet50_preact --batch-size 256 --workers 8 --output-dir runs/imagenet_r50
```

### PACT activation (ReLU replaced)
- All models now use a PACT activation with learnable clipping `alpha` and uniform k-bit quantization (STE on rounding).
- Control activation precision via `--act-bits` (default 4). Pass `--act-bits 0` to disable quantization (still uses learnable clipping).
- First/last layer activations stay unquantized by default; add `--quantize-first-last` to quantize them too.
- L2 regularization on all `alpha` parameters is added to the loss. Control it with `--lambda-alpha` (defaults to the model weight decay).
- Weights use DoReFa-style symmetric quantization per layer during forward. Control with `--w-bits` (default 4); set `--w-bits 0` to disable. First conv and final FC weights stay full-precision unless `--quantize-first-last` is provided.
- Quantization staging: use `--quant-warmup-epochs N` to train the first N epochs with clipping but without quantization (Option B). Default 0 means quantization from the start (Option A).

## Expected full-precision reference (top-1 / top-5)
- CIFAR-10 ResNet-20: ~91.6% top-1
- SVHN ConvNet: ~96%+ top-1
- ImageNet AlexNet+BN: ~55.1 / 77.0
- ImageNet ResNet-18 (pre-act): ~70.4 / 89.6
- ImageNet ResNet-50 (pre-act): ~76.9 / 93.1

Use these as sanity checks before adding PACT quantization.


## Citation Information

```
@article{imagenet15russakovsky,
    Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
    Title = { {ImageNet Large Scale Visual Recognition Challenge} },
    Year = {2015},
    journal   = {International Journal of Computer Vision (IJCV)},
    doi = {10.1007/s11263-015-0816-y},
    volume={115},
    number={3},
    pages={211-252}
}
```
