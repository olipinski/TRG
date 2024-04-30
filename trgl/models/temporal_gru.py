"""Model for the temporal GRU-based network."""
from typing import Tuple

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from trgl.utils.causal_mask import get_causal_mask
from trgl.utils.gumbel_softmax import gumbel_softmax_sample
from trgl.utils.positional_encoding import PositionalEncoding


class TemporalGRU(pl.LightningModule):
    """Network for analysing temporal references in emergent communication."""

    def __init__(
        self,
        n_features: int = 4,
        vocab_size: int = 5,
        max_length: int = 6,
        prev_horizon: int = 1,
        gs_temperature: float = 1.0,
        sender_embedding: int = 128,
        sender_meaning_hidden: int = 128,
        sender_temporal_hidden: int = 128,
        sender_message_hidden: int = 128,
        receiver_hidden: int = 128,
        attention_sender: bool = True,
        attention_sender_n_heads: int = 8,
        attention_sender_dim: int = 128,
        attention_sender_dropout: float = 0.1,
        attention_receiver: bool = True,
        attention_receiver_n_heads: int = 8,
        attention_receiver_dim: int = 128,
        attention_receiver_dropout: float = 0.1,
        temporal: bool = True,
        purely_temporal: bool = True,
        temporal_loss: bool = True,
        eos_char: int = 0,
        length_penalty: float = 0.001,
    ):
        """
        Network for analysing temporal references in emergent communication.

        Parameters
        ----------
        n_features: int
            Number of features per object
        vocab_size:
            Vocabulary size available to the agents
        max_length: int
            Max length of the message that can be generated.
        prev_horizon: int
            The size of the horizon for previous repetitions
        gs_temperature: float
            Temperature for the Gumbel-Softmax trick discretisation.
        sender_embedding: int
            Size of the sender embedding
        sender_meaning_hidden: int
            Size of the sender meaning GRU hidden layer
        sender_temporal_hidden: int
            Size of the sender temporal GRU hidden layer
        sender_message_hidden: int
            Size of the sender message GRU hidden layer
        receiver_hidden:
            Size of the receiver message decoding GRU hidden layer
        attention_sender: bool
            Whether to use attention with the sender, after the GRU
        attention_receiver: bool
            Whether to use attention with the receiver, after the GRU
        temporal: bool
            Whether we should run a temporal network or a regular one.
        purely_temporal:
            Whether to remove the meaning  GRU and rely purely on the temporal GRU
        temporal_loss: bool
            Whether we should use a temporal loss. This is *not* exclusive with the temporal flag.
        eos_char: int
            Character to be used as EOS
        length_penalty: float
            Length penalty factor
        """
        super().__init__()

        self.n_features = n_features
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.prev_horizon = prev_horizon

        self.gs_temperature = gs_temperature

        self.sender_embedding = sender_embedding
        self.sender_meaning_hidden = sender_meaning_hidden
        self.sender_temporal_hidden = sender_temporal_hidden
        self.sender_message_hidden = sender_message_hidden

        self.receiver_hidden = receiver_hidden

        self.attention_sender = attention_sender
        self.attention_sender_n_heads = attention_sender_n_heads
        self.attention_sender_dim = attention_sender_dim
        self.attention_sender_dropout = attention_sender_dropout
        self.attention_receiver = attention_receiver
        self.attention_receiver_n_heads = attention_receiver_n_heads
        self.attention_receiver_dim = attention_receiver_dim
        self.attention_receiver_dropout = attention_receiver_dropout
        self.temporal = temporal
        self.purely_temporal = purely_temporal
        self.temporal_loss = temporal_loss

        # Linguistic parsimony pressure
        self.eos_char = eos_char
        # Values higher than 0.001 make training unstable
        self.length_penalty = length_penalty

        # Agent 1
        # GRU to extract the meaning of the object.
        # E.g. what colours are combined etc
        self.sender_meaning_gru = (
            nn.GRU(
                input_size=self.n_features,
                hidden_size=self.sender_meaning_hidden,
                num_layers=1,
                batch_first=True,
            )
            if not self.purely_temporal
            else None
        )
        # GRU to extract temporal relationships.
        # E.g. do certain objects come more often together?
        # Do they repeat?
        self.sender_temporal_gru = (
            nn.GRU(
                input_size=self.n_features,
                hidden_size=self.sender_temporal_hidden,
                num_layers=1,
                batch_first=True,
            )
            if self.temporal
            else None
        )

        # Attention layer after GRU
        self.sender_pos_encoding_layer = (
            PositionalEncoding(
                d_model=self.attention_sender_dim, dropout=self.attention_sender_dropout
            )
            if self.attention_sender
            else None
        )
        self.sender_attention_embedding_layer = (
            nn.Linear(self.sender_meaning_hidden, self.attention_sender_dim)
            if self.attention_sender
            else None
        )
        self.sender_attention_layer = (
            nn.MultiheadAttention(
                num_heads=self.attention_sender_n_heads,
                embed_dim=self.attention_sender_dim,
                dropout=self.attention_sender_dropout,
            )
            if self.attention_sender
            else None
        )

        # Gumbel-Softmax related stuff, adapted from EGG
        self.sos_embedding = nn.Parameter(torch.zeros(self.sender_embedding))
        self.embedding = nn.Linear(
            in_features=self.vocab_size, out_features=self.sender_embedding
        )
        self.message_gru_sender = nn.GRUCell(
            input_size=self.sender_embedding, hidden_size=self.sender_message_hidden
        )
        self.gru_to_msg = nn.Linear(
            in_features=self.sender_message_hidden, out_features=self.vocab_size
        )

        # Agent 2
        # GRU to process the incoming message.
        self.receiver_message_gru = nn.GRU(
            input_size=self.vocab_size,
            hidden_size=self.receiver_hidden,
            num_layers=1,
            batch_first=True,
        )

        # GRU to process messages temporally
        self.receiver_temporal_gru = (
            nn.GRU(
                input_size=self.receiver_hidden,
                hidden_size=self.receiver_hidden,
                num_layers=1,
                batch_first=True,
            )
            if self.temporal
            else None
        )

        # Attention Layer to process the output of the GRU
        self.receiver_pos_encoding_layer = (
            PositionalEncoding(
                d_model=self.attention_sender_dim, dropout=self.attention_sender_dropout
            )
            if self.attention_receiver
            else None
        )
        self.receiver_attention_embedding_layer = (
            nn.Linear(self.receiver_hidden, self.attention_receiver_dim)
            if self.attention_receiver
            else None
        )
        self.receiver_attention_layer = (
            nn.MultiheadAttention(
                num_heads=self.attention_receiver_n_heads,
                embed_dim=self.attention_receiver_dim,
                dropout=self.attention_receiver_dropout,
            )
            if self.attention_receiver
            else None
        )

        # Network to embed objects.
        self.receiver_obj_embed = nn.Linear(self.n_features, self.receiver_hidden)
        # Network to convert from the hidden GRU state to a temporal guess.
        self.receiver_temporal_pred = (
            nn.Linear(self.receiver_hidden, self.prev_horizon + 1)
            if self.temporal_loss
            else None
        )

        # Dictionary for saving interactions and other statistics.
        self.exchange_dict = {}
        self.exchange_count = 0
        self.eval_mode = False

        self.save_hyperparameters()

    def infer(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict the target object and temporal relationships for a given batch.

        This is a convenience function which is used in all the steps for pytorch lightning to save on duplicate code.

        Parameters
        ----------
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Batch to process, consisting of the objects, target labels and temporal labels.

        Returns
        -------
        Tuple(pred, guess):
            Tuple of temporal prediction and guess as to which object is correct.

        """
        # Unpack the tuple.
        objects, target_labels, temporal_labels = batch
        # Cast the objects from int to float for learning.
        objects = objects.float()
        # Cast the labels from int to long, pytorch requires longs.
        target_labels = target_labels.long()

        # Regular targets, as they should be.
        targets = objects[torch.arange(len(objects)), target_labels]

        # We start in the first agent.
        # Agent 1
        # We can play with what the meaning GRU gets and see the convergence change significantly.
        # Depending on the input shape we may not need to unsqueeze.
        if not self.purely_temporal:
            _, embedding_meaning_object = self.sender_meaning_gru(targets.unsqueeze(1))
            embedding_meaning_object = embedding_meaning_object.permute(1, 0, 2)
            embedding_meaning_object = embedding_meaning_object.reshape(
                objects.shape[0], -1
            )

        # Allow for temporal understanding by different batching.
        if self.temporal or self.purely_temporal:
            # Notice the difference in unsqueeze calls:
            # In the first case (meaning) we pass in the batch in form [128,1,4] for one target
            # So the meaning GRU processes 128 sequences of length 1.
            # Below we change to [1,128,4]. So the GRU processes one continuous sequence
            # to understand the relationships within.
            # This is possibly equivalent to a stateful GRU in Tensorflow.
            embedding_meaning_temporal, _ = self.sender_temporal_gru(
                targets.unsqueeze(0)
            )

            # Let's get it back into a batch shape. The return does not have batch first etc.
            embedding_meaning_temporal = embedding_meaning_temporal.permute(1, 0, 2)
            embedding_meaning_temporal = embedding_meaning_temporal.reshape(
                objects.shape[0], -1
            )

            # Purely temporal network only uses the temporal GRU
            if self.purely_temporal:
                embedded_meaning_together = embedding_meaning_temporal
            else:
                embedded_meaning_together = torch.mul(
                    embedding_meaning_temporal, embedding_meaning_object
                )

        # No temporal understanding requested.
        else:
            # The embedded meaning is just the object meaning.
            embedded_meaning_together = embedding_meaning_object

        # Embed targets into the attention space, and pass through multi-head attention
        if self.attention_sender:
            # Pos encoding: [seq_len, batch_size, embedding_dim]
            attn_mask = get_causal_mask(embedded_meaning_together.shape[0]).to(
                self.device
            )
            meaning_embedded = self.sender_attention_embedding_layer(
                embedded_meaning_together
            )
            meaning_pos_encoded = self.sender_pos_encoding_layer(
                meaning_embedded.unsqueeze(1)
            ).squeeze()
            attention_meaning, _ = self.sender_attention_layer(
                meaning_pos_encoded,
                meaning_pos_encoded,
                meaning_pos_encoded,
                attn_mask=attn_mask,
                need_weights=False,
            )
            embedded_meaning_together = torch.mul(
                embedded_meaning_together, attention_meaning
            )

        # Temporary holder for generated probabilities over vocabulary space.
        # The probabilities are generated with the Gumbel-Softmax trick from EGG.
        sequence = []
        # Pre-seed the hidden state of the GRU with the embedding and temporal understanding.
        prev_hidden = embedded_meaning_together
        # Start of sentence embedding is passed first. This follows from how EGG does this.
        character_to_process = torch.stack([self.sos_embedding] * objects.size(0))

        # Let's generate the message character by character to use Gumbel-Softmax after each character.
        # This will allow for later discretisation of the messages.
        for step in range(self.max_length):
            h_t = self.message_gru_sender(character_to_process, prev_hidden)
            # Process the GRU hidden state into vocab size.
            step_logits = self.gru_to_msg(h_t)
            # Here we generate the character probabilities using the Gumbel-Softmax trick.
            character = gumbel_softmax_sample(
                logits=step_logits,
                training=self.training,
                temperature=self.gs_temperature,
            )
            # Use the resulting hidden state.
            prev_hidden = h_t
            # Process the character back into an embedding for the GRU.
            character_to_process = self.embedding(character)
            # Append character to create a message later.
            sequence.append(character)

        # Create a message from all appended characters, and permute back into batch shape.
        message = torch.stack(sequence).permute(1, 0, 2)

        # Now we move onto the second agent.
        # Agent 2
        # Process message and get the last hidden state for each message.
        # We don't care about per-character hidden states in this case.
        _, message_decoded = self.receiver_message_gru(message)

        # Process the messages through time
        if self.temporal:
            message_temporal, _ = self.receiver_temporal_gru(message_decoded)
            message_decoded_together = torch.mul(message_decoded, message_temporal)
        else:
            message_decoded_together = message_decoded

        if self.attention_receiver:
            # Pos encoding: [seq_len, batch_size, embedding_dim]
            message_embedded = self.receiver_attention_embedding_layer(
                message_decoded_together
            )
            message_pos_encoded = self.receiver_pos_encoding_layer(
                message_embedded.permute(1, 0, 2)
            ).permute(1, 0, 2)
            message_decoded_attention, _ = self.receiver_attention_layer(
                message_pos_encoded,
                message_pos_encoded,
                message_pos_encoded,
                need_weights=False,
            )
            message_decoded_together = torch.mul(
                message_decoded, message_decoded_attention
            )

        # Permute and reshape back to batch shape.
        message_decoded_together = message_decoded_together.permute(1, 0, 2)
        message_decoded_together = message_decoded_together.reshape(
            objects.shape[0], -1
        )
        # Embed the objects to an embedding space.
        embedding_obj = self.receiver_obj_embed(objects).relu()
        # Produce guess by using the two embeddings and multiplying.
        guess = torch.matmul(embedding_obj, message_decoded_together.unsqueeze(dim=-1))
        guess = guess.squeeze()

        # If temporal understanding requested also predict the temporal relationship
        # from the message sent.
        if self.temporal_loss:
            pred = self.receiver_temporal_pred(message_decoded_together)
            pred = pred.squeeze()
        # Otherwise temporal prediction is empty.
        else:
            pred = None

        # In evaluation mode we want to save as much info as possible.
        # Using a dict will make it easier to export to json later.
        # Extending this also is just adding a key-value pair.
        # Some values are dependent on the temporality so that is also handled.
        if self.eval_mode:
            for i in range(objects.size(0)):
                self.exchange_dict[f"exchange_{self.exchange_count}"] = {
                    "objects": objects[i].detach().cpu().numpy().astype(np.int32),
                    "message": message[i].argmax(dim=1).detach().cpu().numpy(),
                    "guess": guess.argmax(dim=1)[i]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int32),
                    "target": targets[i].detach().cpu().numpy().astype(np.int32),
                    "target_id": target_labels[i]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int32),
                    "temporal_prediction": pred.argmax(dim=1)[i]
                    .round()
                    .detach()
                    .cpu()
                    .numpy()
                    if self.temporal_loss
                    else 0,
                    "temporal_label": temporal_labels[i]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int32)
                    if self.temporal_loss
                    else 0,
                }
                self.exchange_count += 1

        return pred, guess, message

    def loss_accuracy(
        self,
        guess: torch.Tensor,
        pred: torch.Tensor,
        message: torch.Tensor,
        target_labels: torch.Tensor,
        temporal_labels: torch.Tensor,
    ):
        """
        Calculate the loss and accuracy.

        Loss includes the regular guess, temporal guess, and the length penalty.
        Uses the cross entropy and binary cross entropy losses.

        Parameters
        ----------
        guess: torch.Tensor
        pred: torch.Tensor
        message: torch.Tensor
        target_labels: torch.Tensor
        temporal_labels: torch.Tensor

        Returns
        -------
        Tuple(loss, accuracy, temporal_accuracy):
            Tuple of loss, accuracy, and temporal accuracy.

        """
        temporal_labels = temporal_labels.long()
        target_labels = target_labels.long()

        # Set to 0.0 to have something to return.
        temporal_accuracy = 0.0

        loss = F.cross_entropy(guess, target_labels)
        accuracy = (guess.argmax(dim=1) == target_labels).detach().float().mean()

        if self.temporal_loss:
            loss += F.cross_entropy(pred, temporal_labels)
            temporal_accuracy = (
                (pred.argmax(dim=1) == temporal_labels).detach().float().mean()
            )

        # Length cost
        # In EGG the loss is calculated per step, so we can fake this by creating
        # a per step loss.

        expected_length = 0.0
        step_loss = loss / message.shape[1]
        length_loss = 0
        eos_val_mask = torch.ones(message.shape[0], device=self.device)
        pos = 0
        for pos in range(message.shape[1]):
            eos_mask = message[:, pos, self.eos_char]
            add_mask = eos_mask * eos_val_mask
            length_loss += (
                step_loss * add_mask + self.length_penalty * (1.0 + pos) * add_mask
            )
            expected_length += add_mask.detach() * (1.0 + pos)
            eos_val_mask = eos_val_mask * (1.0 - eos_mask)

        length_loss += (
            step_loss * eos_val_mask + self.length_penalty * (pos + 1.0) * eos_val_mask
        )
        expected_length += (pos + 1) * eos_val_mask

        if self.length_penalty > 0:
            loss = torch.mean(length_loss)

        expected_length = torch.mean(expected_length)

        return loss, accuracy, temporal_accuracy, expected_length

    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        Predict the objects and temporal relationships.

        This function is used only for post-training inferences. Currently, this function still requires a
        batch with shape like in training, so is not a true inference function.

        Parameters
        ----------
        batch: Tuple[torch.Tensor,torch.Tensor,torch.Tensor]
            Batch to process, consisting of the objects, target labels and temporal labels.

        Returns
        -------
        Tuple(pred, guess):
            Tuple of temporal prediction and guess as to which object is correct.
        """
        self.eval_mode = True
        pred, guess, message = self.infer(batch)

        pred = (
            pred.argmax(dim=1) if self.temporal_loss else torch.zeros(batch[0].shape[0])
        )
        guess = guess.argmax(dim=1)
        return guess, pred, message

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        """
        Run the training for a single batch, with a given batch id.

        This is overridden from PyTorch Lightning.

        Parameters
        ----------
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Batch to process, consisting of the objects, target labels and temporal labels.
        batch_idx: int
            ID of the batch.

        Returns
        -------
        loss: torch.Tensor
            The loss for a given batch.
        """
        objects, target_labels, temporal_labels = batch
        pred, guess, message = self.infer(batch)
        loss, acc, temp_acc, expected_length = self.loss_accuracy(
            guess, pred, message, target_labels, temporal_labels
        )
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_temporal_acc", temp_acc, prog_bar=True)
        self.log("expected_length", expected_length, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        """
        Run the validation for a single batch, with a given batch id.

        This function does not return anything, just logs the loss and accuracies to Lightning.

        Parameters
        ----------
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Batch to process, consisting of the objects, target labels and temporal labels.
        batch_idx: int
            ID of the batch.
        """
        objects, target_labels, temporal_labels = batch
        pred, guess, message = self.infer(batch)
        loss, acc, temp_acc, expected_length = self.loss_accuracy(
            guess, pred, message, target_labels, temporal_labels
        )
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        self.log("val_temporal_acc", temp_acc)
        self.log("expected_length", expected_length)

    def configure_optimizers(self):
        """
        Configure the optimizers to be used for the training.

        Returns
        -------
        optimizer: torch.optim.Optimizer
            Optimizer to be used for training.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
