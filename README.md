# PACT
Reimplementation of the following scientific article: https://arxiv.org/pdf/1805.06085


python train.py --config configs/cifar10_resnet20.yaml
# ou
python train.py --config configs/imagenet_resnet50.yaml


# CIFAR-10 / ResNet20
python evaluate.py --config configs/cifar10_resnet20.yaml --checkpoint results/cifar10_resnet20/best_model.pth

# ImageNet / ResNet50 + matrice de confusion et dump des prédictions
python evaluate.py --config configs/imagenet_resnet50.yaml --checkpoint results/imagenet_resnet50/best_model.pth --confusion --dump_preds
