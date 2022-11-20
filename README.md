# CQTDiff: Solving audio inverse problems with a diffusion model

Official repository of the paper:

> E. Moliner,J. Lehtinen and V. Välimäki, "Solving audio inverse problems with a diffusion model", submitted to IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), Rhodes, Greece May, 2023


Read the paper in [arXiv](https://arxiv.org/abs/2210.15228)
Listen to our [audio samples](http://research.spa.aalto.fi/publications/papers/icassp23-cqt-diff/)


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eloimoliner/CQTdiff/blob/main/notebooks/demo.ipynb)

## Setup
This repository requires Python 3.8+ and Pytorch 1.10+. Other packages are listed in `requirements.txt`.

To install the requirements in your environment:
```bash
pip install -r requirements.txt
```
To install the pre-trained weights and download a set of audio samples from the MAESTRO test set, run:
```bash
bash download_weights_and_examples.sh
```
## Training
To retrain the model, run:

```bash
mkdir experiments/my_experiment
python train.py  model_dir="experiments/my_experiment"
```

To change the configuration, override the hydra parameters (listed in `conf/conf.yaml`)

By default, the training scripts log to wandb. Set `log=False` if this is not desired.
```bash
python train.py log=False
```

## Testing

To easily test our method, we recommend running the [Colab Notebook](https://colab.research.google.com/github/eloimoliner/CQTdiff/blob/main/notebooks/demo.ipynb), where some of the experiments are implemented.

To run it locally, use:
```bash
python sample.py \
        inference.load.load_mode="from_directory" \
        inference.load.data_directory="$path_to_audio_files" \
        inference.mode=$test_mode
```
The variable `$test_mode` selects the type of experiments. Examples are: "bandwidth_extension", "inpainting" or "declipping". There are many other parameters to select listed in the inference section from `conf/conf.yaml`. Some experiment examples are located in the directory `scripts/`.

## Remarks

The model is trained using the MAESTRO dataset, the performance is expected to decrease in out-of-distribution data.
