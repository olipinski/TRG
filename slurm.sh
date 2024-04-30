#!/bin/bash
# Slurm generic batch parameters
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --array=1-8

# Job naming and output
#SBATCH --job-name=trgl-e600-r05
#SBATCH --output=HPC-%x.%j.out

#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#=============================================
# Actual run commands

conda activate trgl

cd /path/to/trgl/ || exit

config=config/slurm_config.txt

prev_horizon=$(awk -v ArrayTaskID="$SLURM_ARRAY_TASK_ID" '$1==ArrayTaskID {print $2}' $config)
length_penalty=$(awk -v ArrayTaskID="$SLURM_ARRAY_TASK_ID" '$1==ArrayTaskID {print $3}' $config)
message_length=$(awk -v ArrayTaskID="$SLURM_ARRAY_TASK_ID" '$1==ArrayTaskID {print $4}' $config)
vocab_size=$(awk -v ArrayTaskID="$SLURM_ARRAY_TASK_ID" '$1==ArrayTaskID {print $5}' $config)

export WANDB_MODE=offline

python -m trgl.run \
        --max_epochs 600 \
        --num_objects 20000 \
        --num_distractors 10 \
        --num_properties 8 \
        --num_features 8 \
        --message_length "${message_length}" \
        --vocab_size "${vocab_size}" \
        --repeat_chance 0.5 \
        --prev_horizon "${prev_horizon}" \
        --length_penalty "${length_penalty}" \
        --sender_hidden 128 \
        --receiver_hidden 128 \
        --wandb_group "e600-r05-h${prev_horizon}-lp${length_penalty}-ln${message_length}-v${vocab_size}"