# Temporal Referential Games

## About

This repo provides a new architecture and dataset for Emergent Communication
research. We introduce a variant of the well-known referential games, where we
include a temporal aspect to the communication. This is done through skewing the
target distribution to include target repetitions at random intervals. Through
this we aim to study how and when temporal references can emerge between agents.

## Installation

### Conda

For Linux and Windows:

```shell
conda env create -f environment.yml
```

For Mac M1/M2:

```shell
conda env create -f environment-metal.yml
```

## Running

```shell
python -m trgl.run [OPTIONS]
```

Possible command line options are available with

```shell
python -m trgl.run --help
```

### Reproducing the paper results

The following commands have been used to create the results for our paper. The
data for visualisations was collected from the interaction logs and WandB. The
repeat chance has been varied between \[0.25,0.50,0.75\], the previous horizon
between \[4,8\], the message length between \[5,6\], the vocabulary size between
\[26,48\], and the length penalty between \[0, 0.001\].

The command below was run using Slurm with requested resources of 96GB of RAM,
20 cores of Xeon(R) Silver 4216 CPUs, and an NVidia RTX8000. However, it should
be possible to run using much fewer resources, especially in terms of the GPU
power and memory.

Additionally, to run the command below you will need a WandB account, or to
disable wandb logging with `--wandb_offline`. This will however mean that
running our notebook for analysis will not be possible, until the results are
extracted from the logs, or uploaded to WandB.

```shell
python -m trgl.run \
        --max_epochs 600 \
        --num_objects 20000 \
        --num_distractors 10 \
        --num_properties 8 \
        --num_features 8 \
        --message_length length \
        --vocab_size vocab_size \
        --repeat_chance chance \
        --prev_horizon prev_horizon \
        --length_penalty length_penalty \
        --attention_sender=False \
        --attention_receiver=False \
        --sender_hidden 128 \
        --receiver_hidden 128 \
        --wandb_group  "e600-r05-h${prev_horizon}-lp${length_penalty}-ln${message_length}-v${vocab_size}"
```

## Code Structure and Documentation

The structure is as follows:

```
- trgl # Top layer directory
    - analysis
        - data # Directory to supply the data to analyse
        - langauge_analysis_v2.ipynb # Jupyter Notebook used to analyse the data as generated by run.py
        - has_analysis.ipynb # Jupyter Notebook used to analyse the data using Harris Segmentation Scheme adapted from Ueda et al. 2023.
    - config
        - slurm_config.txt # Slurm configuration for job arrays. Can be run with example slurm script.
    - trgl
        - models
            - temporal_lstm.py # Neural network structure
        - utils
            - gumbel_softmax.py # Code adapted from EGG for GS Discretisation
            - metrics.py # Code for different metrics used in the analysis
            - causal_mask.py # Causal masking for attention layers
            - positional_encoding.py # Positional encoding module for attention
            - harris_segmentation.py # Code for Harris Segmentation Scheme adapted from Ueda et al. 2023.
        - dataset.py # Dataset generation logic
        - run.py # Main entry point
```

The file `run.py` generates different datasets and tests all types of agents on
them. It then saves the generated data and interactions logs to a `logs` folder.

These logs can then be used by `language_analysis_v2.ipynb` to analyse the
language that the agents use.

Most of the code is relatively well-commented and documented, so for more
details than provided above, refer to the Python files themselves.

## Issues

WandB logger seems to hang currently, so a hacky workaround is used.
