# Categorical Schrödinger Bridge Matching

## Requirements

Create the Anaconda environment using the following command:

```bash
conda env create -f environment.yml
```

### Dataset Preparation

Download the CelebA dataset from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/data) and unzip it. Follow these steps to prepare the dataset:

1. Rename the dataset folder to `celeba`.
2. Rename the folder `celeba/img_align_celeba/img_align_celeba` to `celeba/img_align_celeba/raw`.

## Training

### Experiment Directory Structure

By specifying the `exp_dir` parameter in all `train_*.sh` scripts, you can set a custom path for saving experiment results.

The `exp_dir` structure is as follows:

```bash
|-- images
|   `-- gaussian
|       |-- checkpoints 
|       |   |-- forward_*
|       |   |   `-- model.safetensors
|       |   |-- ...
|       |   |-- backward_*
|       |   `-- ...
|       |-- samples (generated images)
|       |-- trajectories (image trajectories)
|       `-- config.yaml (copied config)
|-- quantized_images
|   `-- uniform
|       |-- ...
|       ...
`-- toy
    |-- gaussian
    |   |-- ...
        ...
    `-- uniform
        |-- ...
        ...
```

### 2D Experiment

To train on the *Gaussian-to-Swiss roll* dataset, run:

```bash
bash train_toy.sh
```

To modify training parameters, edit `configs/toy.yaml` and specify Accelerate parameters in `train_toy.sh`.

### Colored MNIST

To train the *3-to-2 digits* translation, run:

```bash
bash train_images.sh
```

Modify training parameters in `configs/images.yaml` and specify Accelerate parameters in `train_images.sh`.

### CelebA

#### VQ-GAN Pretraining

Before training the *male-to-female* translation, you must train a VQ-GAN model. Follow the official repository: [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers).

The config used in our article is `configs/vqgan_celeba_f8_1024.yaml`.

#### Quantizing CelebA Images

CelebA images must be pre-quantized before training. Run the following script:

```bash
bash quantize_images.sh
```

Specify the VQ-GAN config and checkpoint by providing `quantized_images.yaml` in `quantize_images.sh`. This script generates quantized images as vectorized numpy arrays in `celeba/img_align_celeba/quantized`.

#### Training CelebA

To train the quantized CelebA model, run:

```bash
bash train_quantized_images.sh
```

Modify training parameters in `configs/train_quantized_images.yaml` and specify Accelerate parameters in `train_quantized_images.sh`.

## Evaluation

To evaluate the CelebA experiment, run:

```bash
bash generate_quantized_images.sh
```

Specify the `exp_path` parameter, pointing to the saved experiment folder. You can also provide a list of iterations at which models should generate images. The generated images will be saved in:

```bash
exp_path/checkpoints/{forward_or_backward}_{iteration}/generations
```

To calculate FID and CMMD, use any available implementations, such as:

- [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
- [cmmd-pytorch](https://github.com/sayakpaul/cmmd-pytorch)
