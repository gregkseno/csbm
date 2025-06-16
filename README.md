<div align="center">

# Categorical Schrödinger Bridge Matching (ICML 2025)

[Grigoriy Ksenofontov](https://scholar.google.com/citations?user=e0mirzYAAAAJ),
[Alexander Korotin](https://scholar.google.ru/citations?user=1rIIvjAAAAAJ)

[![arXiv Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2502.01416)
[![OpenReview Paper](https://img.shields.io/badge/OpenReview-PDF-8c1b13)](https://openreview.net/forum?id=RBly0nOr2h)
[![GitHub](https://img.shields.io/github/stars/gregkseno/csbm?style=social)](https://github.com/gregkseno/csbm)
![GitHub License](https://img.shields.io/github/license/gregkseno/csbm)
[![WandB](https://img.shields.io/badge/W%26B-view-FFCC33?logo=wandb)](https://wandb.ai/gregkseno/csbm)

</div>

This repository contains the official implementation of the paper "Categorical Schrödinger Bridge Matching", submitted to ICML 2025.

## 📌 TL;DR

This paper extends the Schrödinger Bridge problem to work with discrete time and spaces.

<!-- ![teaser](./images/teaser.png) -->

## 📦 Dependencies

Create the Anaconda environment using the following command:

```bash
conda env create -f environment.yml
```

## 🛠️ Preparations

### Download Datasets

1. Use [this link](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/data) to obtain the CelebA dataset;
2. Follow [these instructions](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq) to obtain the AFHQv2 dataset.

Additionally, for the CelebA dataset, rename the main folder to `celeba`, then rename `celeba/img_align_celeba/img_align_celeba` to `celeba/img_align_celeba/raw`.

### Train VQ-GAN

1. Configure the appropriate configuration file `configs/vqgan_*.yaml`. 
2. Run the corresponding `quantize_*.sh` script to save quantized images as `.npy` files in `celeba/img_align_celeba/quantized/` or `afhq/*/*/`.

> [!TIP]
> For more details on training VQ-GAN, refer to [the official repository](https://github.com/CompVis/taming-transformers).

### Train Tokenizer

1. Set `tokenizer.path` in the main config file `configs/amazon.yaml` or `configs/yelp.yaml`
2. Run `train_tokenizer_*.sh` to train the tokenizer.


## 🏋️‍♂️ Training

1. Set the corresponding configuration files;
2. Use the appropriate scripts or notebooks.

| Experiment name                                 | Script/Notebook                    | Configs (`config/`) | Weights |
| ---------------------------------------------- | ----------------------------------- | ------- | ------- |
| Convergence of D-IMF on Discrete Spaces        | `notebooks/convergence_d_imf.ipynb` |  N/A | N/A |
| Illustrative 2D Experiments                    | `train_swiss_roll.sh` |  `swiss_roll.yaml` | N/A |
| Unpaired Translation on Colored MNIST          | `train_cmnist.sh` |  `cmnist.yaml` | N/A |
| Unpaired Translation of CelebA Faces           | `train_celeba.sh` |  `celeba.yaml`, `vqgan_celeba_f8_1024.yaml` | N/A |
| Unpaired Translation of AFHQ Faces             | `train_afhq.sh` |  `afhq.yaml`, `vqgan_afhq_f32_1024.yaml` | N/A |
| Unpaired Text Style Transfer of Amazon Reviews | `train_amazon.sh` |  `amazon.yaml` | N/A |
| Unpaired Text Style Transfer of Yelp Reviews   | `train_yelp.sh` |  `yelp.yaml` | N/A |

> [!TIP] 
> Set the `exp_dir` parameter in any `train_*.sh` script to define a custom path for saving experiment results, following the structure below:
> ```bash
> data.type
> `-- data.dataset
>    `-- prior.type
>        |-- checkpoints 
>        |   |-- forward_*
>        |   |   `-- model.safetensors
>        |   |-- ...
>        |   |-- backward_*
>        |   `-- ...
>        |-- samples      # images of samples
>        |-- trajectories # images of trajectories
>        `-- config.yaml  # copied config
> ```

## 📊 Evaluation

1. Specify the `exp_path` parameter, pointing to the saved experiment folder;
2. Run `eval_*.sh` with appropriate `iteration` argument.

> [!IMPORTANT]
> Reusing an earlier evaluation pipeline for CelebA dataset may yield different results. In the article, images were generated first (see `scripts/generate.py`) and then evaluated with the following metrics (see `notebooks/eval.ipynb`):
> * **FID** from [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
> * **CMMD** from [cmmd-pytorch](https://github.com/sayakpaul/cmmd-pytorch)
> * **LPIPS** from [torchmetrics](https://lightning.ai/docs/torchmetrics/stable/image/learned_perceptual_image_patch_similarity.html)

