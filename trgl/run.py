"""A simple neural network to validate temporal references."""
import json
import os
import platform
import time

import lightning.pytorch as pl
import shortuuid
import torch
import torch.nn.functional as F
import wandb
from absl import app, flags
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from numpyencoder import NumpyEncoder
from torch.utils.data import DataLoader

from trgl.dataset import TemporalDataset
from trgl.models.temporal_gru import TemporalGRU
from trgl.models.temporal_lstm import TemporalLSTM

# FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "seed", None, "The seed to use. If none is provided a random seed is generated."
)
flags.DEFINE_integer("max_epochs", 200, "Maximum number of epochs to train for.")

flags.DEFINE_string("wandb_group", None, "Name of the WandB group to put the run into.")
flags.DEFINE_bool("wandb_offline", False, "Whether to run WandB in offline mode.")

# Environment Flags
flags.DEFINE_integer(
    "num_objects", 10000, "Number of objects for the agents to converse about"
)

flags.DEFINE_integer(
    "num_distractors", 3, "Number of distractor objects to show to the receiver"
)

flags.DEFINE_integer(
    "num_properties",
    4,
    "Number of properties that the distractor objects can have (e.g. colour)",
)

flags.DEFINE_integer(
    "num_features",
    4,
    "Number of features that each property of the distractor object can have"
    "(e.g. blue, red, green)",
)

flags.DEFINE_integer(
    "message_length",
    4,
    "The maximum length of the message that the sender agent can send",
)

flags.DEFINE_integer(
    "vocab_size", 4, "The size of the available vocabulary to the sender"
)

flags.DEFINE_integer(
    "prev_horizon",
    1,
    "Number of recurrence steps for the previous variant. "
    "Setting this to two makes it possible for the environment to have references to previous previous timestep etc.",
)

flags.DEFINE_float(
    "repeat_chance",
    0.5,
    "Chance that an object will repeat. This applies to the temporal datasets.",
)

# Agent Flags

flags.DEFINE_bool(
    "attention",
    True,
    "Whether networks with sender/receiver attention should also be evaluated."
    " This will add an attention layer after the sender/receiver LSTM.",
)

flags.DEFINE_bool(
    "temporal_network", True, "Whether the temporal network should also be evaluated."
)

flags.DEFINE_bool(
    "purely_temporal",
    True,
    "Whether the purely temporal network should also be evaluated.",
)

flags.DEFINE_bool(
    "temporal_loss",
    True,
    "Whether the networks should also be evaluated with a temporal loss. NOT RECOMMENDED.",
)

flags.DEFINE_float(
    "length_penalty",
    0.001,
    "Factor by which length penalty will be multiplied. "
    "Setting to 0 disables the length penalty. Higher values (>0.001) may lead to unstable training.",
)

flags.DEFINE_integer(
    "sender_hidden", 128, "Hidden size of the sender LSTMs and attention."
)

flags.DEFINE_integer(
    "receiver_hidden", 128, "Hidden size of the receiver LSTMs and attention."
)


def main(argv):
    """
    Run the training and evaluation.

    Parameters
    ----------
    argv:
        Unused, as this is handled by absl
    """
    del argv  # Unused.

    # Lightning sets it as dry run instead of offline
    if FLAGS.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    # We want the logs to be contained in the main folder
    # Unless told otherwise
    full_path = os.path.realpath(__file__)
    path = os.path.split(os.path.split(full_path)[0])[0]

    # No support for custom log_dirs for now
    log_dir = os.path.join(path, "logs")

    # Check whether the specified paths exist or not and create them
    # Sleep is to make sure change is committed
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        time.sleep(1)

    if not os.path.exists(os.path.join(log_dir, "lightning_tensorboard")):
        os.makedirs(os.path.join(log_dir, "lightning_tensorboard"))
        time.sleep(1)

    if not os.path.exists(os.path.join(log_dir, "lightning_wandb")):
        os.makedirs(os.path.join(log_dir, "lightning_wandb"))
        time.sleep(1)

    if not os.path.exists(os.path.join(log_dir, "lightning_interactions")):
        os.makedirs(os.path.join(log_dir, "lightning_interactions"))
        time.sleep(1)

    run_uuid = shortuuid.uuid()[:8]

    train_temporal = TemporalDataset(  # noqa: F841
        dataset_type="trg_previous",
        seed=FLAGS.seed,
        num_objects=int(0.8 * FLAGS.num_objects),
        num_distractors=FLAGS.num_distractors,
        num_features=FLAGS.num_features,
        num_properties=FLAGS.num_properties,
        prev_horizon=FLAGS.prev_horizon,
        repeat_chance=FLAGS.repeat_chance,
    )

    train_regular = TemporalDataset(  # noqa: F841
        dataset_type="rg_classic",
        seed=FLAGS.seed,
        num_objects=int(0.8 * FLAGS.num_objects),
        num_distractors=FLAGS.num_distractors,
        num_features=FLAGS.num_features,
        num_properties=FLAGS.num_properties,
        prev_horizon=0,
        repeat_chance=FLAGS.repeat_chance,
    )

    validation_dataset = TemporalDataset(
        dataset_type="trg_previous",
        seed=FLAGS.seed,
        num_objects=int(0.2 * FLAGS.num_objects),
        num_distractors=FLAGS.num_distractors,
        num_features=FLAGS.num_features,
        num_properties=FLAGS.num_properties,
        prev_horizon=FLAGS.prev_horizon,
        repeat_chance=FLAGS.repeat_chance,
    )

    trainer_list = []

    if FLAGS.purely_temporal:
        assert (
            FLAGS.temporal_network
        ), "Temporal network is required to use purely temporal!"

    # agent_flags = [
    #     FLAGS.temporal_network,
    #     FLAGS.temporal_loss,
    #     FLAGS.purely_temporal,
    #     FLAGS.attention,
    # ]
    configs = [
        [False, False, False, False],  # Base
        [False, True, False, False],  # Base+L
        [True, False, True, False],  # Temporal
        [True, True, True, False],  # Temporal+L
        [True, False, False, False],  # TemporalR
        [True, True, False, False],  # TemporalR+L
        [False, False, False, True],  # Attention
        [False, True, False, True],  # Attention+L
    ]

    # Check GPU capability for compile
    compile_ok = False
    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        if device_cap[0] >= 7:
            compile_ok = True
        if platform.uname()[0] == "Windows":
            compile_ok = False

    # LSTM Training
    # Train all requested permutations of training set and temporal aspects.

    # We train on two datasets - temporal and regular.
    # Temporal dataset includes more emphasised temporal relationships through target repetition.
    # Regular is just a randomly generated dataset.
    run_count = 0
    max_runs = len(configs) * 4
    print(f"Global run ID: {run_uuid}")
    for architecture in ["LSTM", "GRU"]:
        for train_dataset_str in ["train_temporal", "train_regular"]:
            for config in configs:
                print(f"Starting training run {run_count+1} out of {max_runs}")
                train_dataset = eval(train_dataset_str)
                tb_logger = TensorBoardLogger(
                    save_dir=os.path.join(log_dir, "lightning_tensorboard"),
                    name=f"run-{run_uuid}-{architecture}-{train_dataset_str}-temporal_net_{config[0]}-"
                    f"temporal_loss_{config[1]}-purely_temporal_{config[2]}-attention_sender_{config[3]}-"
                    f"attention_receiver_{config[3]}",
                )
                wandb_logger = WandbLogger(
                    project="TRGLv6",
                    save_dir=os.path.join(log_dir, "lightning_wandb"),
                    group=FLAGS.wandb_group,
                    name=f"run-{run_uuid}-{architecture}{train_dataset_str}-temporal_net_{config[0]}-"
                    f"temporal_loss_{config[1]}-purely_temporal_{config[2]}-attention_sender_{config[3]}-"
                    f"attention_receiver_{config[3]}",
                )
                checkpoint_callback = ModelCheckpoint(
                    dirpath=os.path.join(
                        log_dir,
                        "checkpoints",
                        f"run-{run_uuid}-{architecture}{train_dataset_str}-temporal_net_{config[0]}-"
                        f"temporal_loss_{config[1]}-purely_temporal_{config[2]}-attention_sender_{config[3]}-"
                        f"attention_receiver_{config[3]}",
                    ),
                    monitor="val_acc",
                    save_top_k=3,
                    mode="max",
                )
                if architecture == "LSTM":
                    network = TemporalLSTM(
                        temporal=config[0],
                        temporal_loss=config[1],
                        purely_temporal=config[2],
                        attention_sender=config[3],
                        attention_receiver=config[3],
                        n_features=FLAGS.num_features,
                        max_length=FLAGS.message_length,
                        vocab_size=FLAGS.vocab_size,
                        length_penalty=FLAGS.length_penalty,
                        prev_horizon=FLAGS.prev_horizon,
                        sender_embedding=FLAGS.sender_hidden,
                        sender_temporal_hidden=FLAGS.sender_hidden,
                        sender_meaning_hidden=FLAGS.sender_hidden,
                        sender_message_hidden=FLAGS.sender_hidden,
                        attention_sender_dim=FLAGS.sender_hidden,
                        receiver_hidden=FLAGS.receiver_hidden,
                        attention_receiver_dim=FLAGS.receiver_hidden,
                    )
                else:
                    network = TemporalGRU(
                        temporal=config[0],
                        temporal_loss=config[1],
                        purely_temporal=config[2],
                        attention_sender=config[3],
                        attention_receiver=config[3],
                        n_features=FLAGS.num_features,
                        max_length=FLAGS.message_length,
                        vocab_size=FLAGS.vocab_size,
                        length_penalty=FLAGS.length_penalty,
                        prev_horizon=FLAGS.prev_horizon,
                        sender_embedding=FLAGS.sender_hidden,
                        sender_temporal_hidden=FLAGS.sender_hidden,
                        sender_meaning_hidden=FLAGS.sender_hidden,
                        sender_message_hidden=FLAGS.sender_hidden,
                        attention_sender_dim=FLAGS.sender_hidden,
                        receiver_hidden=FLAGS.receiver_hidden,
                        attention_receiver_dim=FLAGS.receiver_hidden,
                    )

                if compile_ok:
                    network = torch.compile(network)

                wandb_logger.experiment.config.update(
                    {
                        "architecture": architecture,
                        "num_objects": FLAGS.num_objects,
                        "num_distractors": FLAGS.num_distractors,
                        "num_features": FLAGS.num_features,
                        "num_properties": FLAGS.num_properties,
                        "message_length": FLAGS.message_length,
                        "vocab_size": FLAGS.vocab_size,
                        "prev_horizon": FLAGS.prev_horizon,
                        "repeat_chance": FLAGS.repeat_chance,
                        "seed": FLAGS.seed,
                        "max_epochs": FLAGS.max_epochs,
                        "length_penalty": FLAGS.length_penalty,
                    }
                )
                trainer = pl.Trainer(
                    accelerator="auto",
                    max_epochs=FLAGS.max_epochs,
                    logger=[tb_logger, wandb_logger],
                    callbacks=checkpoint_callback,
                )
                trainer.fit(
                    network,
                    DataLoader(
                        train_dataset,
                        batch_size=128,
                        shuffle=False,
                        num_workers=12,
                        pin_memory=True,
                        persistent_workers=True,
                    ),
                    DataLoader(
                        validation_dataset,
                        batch_size=128,
                        shuffle=False,
                        num_workers=12,
                        pin_memory=True,
                        persistent_workers=True,
                    ),
                )
                # Otherwise WandB logger will re-use the run which we don't want.
                wandb.finish()
                trainer_list.append(
                    (
                        trainer,
                        f"run-{run_uuid}-{architecture}-{train_dataset_str}-temporal_net_{config[0]}-"
                        f"temporal_loss_{config[1]}-purely_temporal_{config[2]}-attention_sender_{config[3]}-"
                        f"attention_receiver_{config[3]}",
                    )
                )
                run_count += 1

    # Test flow, including the dumping of the test dictionaries for later analysis.
    run_count = 0
    max_runs *= 6
    for trainer, trainer_str in trainer_list:
        for dataset_type in [
            "trg_previous",
            "trg_hard",
            "rg_classic",
            "rg_hard",
            "analysis_always_same",
            "analysis_never_same",
        ]:
            print(f"Starting evaluation run {run_count+1} out of {max_runs}")
            print(f"Evaluation predictions for {trainer_str} on {dataset_type}")

            num_objects = int(0.2 * FLAGS.num_objects)

            # Very hacky way to clear the exchange dict...
            trainer.strategy._lightning_module.exchange_dict = {}
            trainer.strategy._lightning_module.exchange_count = 0

            dataset = TemporalDataset(
                dataset_type=dataset_type,
                seed=FLAGS.seed,
                num_objects=num_objects,
                num_distractors=FLAGS.num_distractors,
                num_features=FLAGS.num_features,
                num_properties=FLAGS.num_properties,
                prev_horizon=FLAGS.prev_horizon,
                repeat_chance=FLAGS.repeat_chance,
            )
            # Remove loggers as they break predictions
            # TODO WandB complains about the run having finished
            # Cannot reproduce on small scripts, so not sure what is wrong
            trainer._loggers = []
            predictions_val = trainer.predict(
                dataloaders=DataLoader(dataset, batch_size=128, shuffle=False),
                ckpt_path="last",
            )

            # Save interactions
            # Slightly hacky but it works.
            with open(
                os.path.join(
                    log_dir,
                    "lightning_interactions",
                    f"{trainer_str}-{dataset_type}-interactions.json",
                ),
                "w",
            ) as f:
                json.dump(
                    trainer.strategy._lightning_module.exchange_dict,
                    f,
                    cls=NumpyEncoder,
                )

            predictions_temporal_val = [prediction[1] for prediction in predictions_val]
            predictions_obj_val = [prediction[0] for prediction in predictions_val]

            predictions_temporal = torch.cat(predictions_temporal_val)
            predictions_obj = torch.cat(predictions_obj_val)

            true_temporal = 0
            true_obj = 0
            labels_temporal = []
            labels_obj = []
            for x in range(num_objects):
                labels_temporal.append(dataset[x][2])
                labels_obj.append(dataset[x][1])
                true_temporal += 1 if predictions_temporal[x] == dataset[x][2] else 0
                true_obj += 1 if (predictions_obj[x] == dataset[x][1]) else 0

            loss_temporal = F.cross_entropy(
                predictions_temporal.float(), torch.tensor(labels_temporal).float()
            )

            loss_obj = F.cross_entropy(
                predictions_obj.float(), torch.tensor(labels_obj).float()
            )

            print(f"Validation temporal n_correct: {true_temporal} ")
            print(f"Validation temporal loss: {loss_temporal}")

            print(f"Validation obj n_correct: {true_obj}")
            print(f"Validation obj loss: {loss_obj}")

            run_count += 1


if __name__ == "__main__":
    app.run(main)
