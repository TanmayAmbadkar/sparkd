# Latent-SPICE

SPICE is a tool for allowing reinforcement learning agents to explore their
environment safely, and is introduced in the paper "Guiding Safe Exploration
with Weakest Preconditions." This repository contains an extension to the original paper, 
allowing environments with larger state spaces to work wih SPICE. 

## Requirements

This code has been tested with Python 3.8.0. For these instructions we will
assume the command `python` refers to Python 3.8 and the command `pip` refers
to an appropriate version of pip. The required packages are listed in
`requirements.txt` and can be installed with

    pip install -r requirements.txt

SPICE also relies on the py-earth package for model learning. Code and
installation instructions can be found at the [py-earth
GitHub](https://github.com/scikit-learn-contrib/py-earth).

## Running

The entry point for all experiments is `main.py`. To replicate the experiments
from the paper, run

    python main_e2c.py --env_name bipedal_walker --num_steps 500000 --red_dim 3

where `bipedal_walker` may be replaced with the name of any other benchmark. 
To see a full list of options, run `python main.py --help`.

## Acknowledgements

The code in `pytorch_soft_actor_critic` along with `main.py` is adapted from
<https://github.com/pranz24/pytorch-soft-actor-critic>. 
