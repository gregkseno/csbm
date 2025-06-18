<div align="center">

# Categorical Schrödinger Bridge Matching (CSBM)

[Grigoriy Ksenofontov](https://scholar.google.com/citations?user=e0mirzYAAAAJ),
[Alexander Korotin](https://scholar.google.ru/citations?user=1rIIvjAAAAAJ)

[![arXiv Paper](https://img.shields.io/badge/arXiv-2502.01416-b31b1b)](https://arxiv.org/abs/2502.01416)
[![OpenReview Paper](https://img.shields.io/badge/OpenReview-PDF-8c1b13)](https://openreview.net/forum?id=RBly0nOr2h)
[![GitHub](https://img.shields.io/github/stars/gregkseno/csbm?style=social)](https://github.com/gregkseno/csbm)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-view-green)](https://huggingface.co/gregkseno/csbm)
[![WandB](https://img.shields.io/badge/W%26B-view-green)](https://wandb.ai/gregkseno/csbm)
![GitHub License](https://img.shields.io/github/license/gregkseno/csbm)

</div>

This repository contains the official implementation of the paper "Categorical Schrödinger Bridge Matching", accepted at ICML 2025.

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

| Experiment name                                 | Script/Notebook                    | Configs (`config/`) | Weights (W&B link) |
| ---------------------------------------------- | ----------------------------------- | ------- | ------- |
| Convergence of D-IMF on Discrete Spaces        | `notebooks/convergence_d_imf.ipynb` |  N/A | N/A |
| Illustrative 2D Experiments                    | `train_swiss_roll.sh` |  `swiss_roll.yaml` | N/A |
| Unpaired Translation on Colored MNIST          | `train_cmnist.sh` |  `cmnist.yaml` | [CSBM](https://huggingface.co/gregkseno/csbm/tree/main/checkpoints%20/images/cmnist) |
| Unpaired Translation of CelebA Faces           | `train_celeba.sh` |  `celeba.yaml`, `vqgan_celeba_f8_1024.yaml` | [CSBM](https://huggingface.co/gregkseno/csbm/tree/main/checkpoints%20/quantized_images/celeba), [VQ-GAN](https://huggingface.co/gregkseno/csbm/blob/main/checkpoints%20/vqgan_celeba_f8_1024.ckpt) |
| Unpaired Translation of AFHQ Faces             | `train_afhq.sh` |  `afhq.yaml`, `vqgan_afhq_f32_1024.yaml` | N/A |
| Unpaired Text Style Transfer of Amazon Reviews | `train_amazon.sh` |  `amazon.yaml` | [CSBM](https://huggingface.co/gregkseno/csbm/tree/main/checkpoints%20/texts/amazon), [Tokenizer](https://huggingface.co/gregkseno/csbm/blob/main/checkpoints%20/tokenizer_amazon.json) |
| Unpaired Text Style Transfer of Yelp Reviews   | `train_yelp.sh` |  `yelp.yaml` | N/A |

> [!TIP]
> Set the `exp_dir` parameter in any `train_*.sh` script to define a custom path for saving experiment results, following the structure below:
>
> ```bash
> data.type               # e.g., toy, images, etc.
> `-- data.dataset        # e.g., swiss_roll, cmnist, etc. 
>    `-- prior.type       # e.g., gaussian, uniform, etc. 
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
2. Run `eval_*.sh` with the appropriate `iteration` argument.

> [!IMPORTANT]
> Reusing an earlier evaluation pipeline for the CelebA dataset may yield different results. In the article, images were generated first (see `scripts/generate.py`) and then evaluated with the following metrics (see `notebooks/eval.ipynb`):
>
> - **FID** from [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
> - **CMMD** from [cmmd-pytorch](https://github.com/sayakpaul/cmmd-pytorch)
> - **LPIPS** from [torchmetrics](https://lightning.ai/docs/torchmetrics/stable/image/learned_perceptual_image_patch_similarity.html)

## 🎓 Citation

```bibtex
@inproceedings{
  ksenofontov2025categorical,
  title={Categorical {Schr\"odinger} Bridge Matching},
  author={Grigoriy Ksenofontov and Alexander Korotin},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=RBly0nOr2h}
}
```

## 🙏 Credits

- [Weights & Biases](https://wandb.ai) — experiment-tracking and visualization toolkit;
- [Hugging Face](https://huggingface.co) — Tokenizers and Accelerate libraries for tokenizer implementation, parallel training, and checkpoint hosting on the Hub;
- [D3PM](https://github.com/google-research/google-research/tree/master/d3pm) — reference implementation of discrete-diffusion models;
- [Taming Transformers](https://github.com/CompVis/taming-transformers) — original VQ-GAN codebase;
- [VQ-Diffusion](https://github.com/microsoft/VQ-Diffusion) — vector-quantized diffusion architecture;
- [MDLM](https://github.com/kuleshov-group/mdlm) — diffusion architecture for text-generation experiments;
- [ASBM](https://arxiv.org/abs/2405.14449) — evaluation metrics and baseline models for CelebA face transfer;
- [Balancing the Style-Content Trade-Off in Sentiment Transfer Using Polarity-Aware Denoising](https://arxiv.org/abs/2312.14708) — processed Amazon Reviews dataset and sentiment-transfer baselines;
- [Inkscape](https://inkscape.org/) — an excellent open-source editor for vector graphics.
