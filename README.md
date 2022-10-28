# CQTDiff: Solving audio inverse problems with a diffusion model

Official repository of the paper:

> E. Moliner,J. Lehtinen and V. Välimäki, "Solving audio inverse problems with a diffusion model", submitted to IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), Rhodes, Greece May, 2023


Listen to our [audio samples](http://research.spa.aalto.fi/publications/papers/icassp23-cqt-diff/)
## Abstract
TODO



[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eloimoliner/CQTdiff/blob/main/notebook/demo.ipynb)

## Requirements
Python 3.8

pip install -r requirements.txt


## Training
To retrain the model, follow the instructions:

mkdir experiments/my_experiment
python train.py  model_dir="experiments/my_experiment"

To change the configuration, override the hydra parameters from conf/conf.yaml

By default, the training scripts logs to wandb. Set log=False if this is not desired

python train.py log=False

## Testing

Some of the experiments are implemented in the Colab notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eloimoliner/CQTdiff/blob/main/notebook/demo.ipynb). 

bash scripts/....sh

python sample.py \
        inference.load.load_mode="from_directory" \
        inference.load.data_directory="$path_to_audio_files" \
        inference.mode=$test_mode

the variable $test_mode selects the type of experiments. Examples are: "bandwidth_extension", "inpainting" or "declipping"

## Remarks

The model is trained using the MAESTRO dataset, the performance is expected to decrease in out-of-distribution data
