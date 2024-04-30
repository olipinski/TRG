"""
Dataset utilities for the (Temporal) Referential Games.

Generates the required datasets for the environment to use.
"""
import itertools
from collections import deque
from typing import Literal, Optional, Tuple

import numpy as np
from numpy.random import default_rng
from torch.utils import data


class TemporalDataset(data.Dataset):
    """
    The class that implements dataset generation for (Temporal) Referential Games.

    This can generate the data required for the environment,
    with different kinds being generated for the given dataset_type.
    """

    def __init__(
        self,
        dataset_type: Literal[
            "rg_classic",
            "rg_hard",
            "trg_previous",
            "trg_sometime_past",
            "trg_never_past",
            "trg_hard",
            "analysis_always_same",
            "analysis_never_same",
        ],
        seed: Optional[int] = None,
        num_objects: int = 10000,
        num_distractors: int = 3,
        num_properties: int = 4,
        num_features: int = 4,
        prev_horizon: int = 1,
        repeat_chance: float = 0.5,
    ):
        """
        Initialise the dataset class, which also generates the dataset.

        Parameters
        ----------
        dataset_type: Literal
            What type of dataset to generate. "rg_classic" is a classic referential game, "rg_hard" is a hard
            referential game, where the objects are very similar, "trg_previous" is a referential game biased towards
            the LTL "previous" operator, "trg_sometime_past" is a referential game biased towards
            the "sometime in the past" LTL operator, and the "trg_never_past" is a referential game biased towards
            the "never in the past" LTL operator.
        seed: int
            Seed for the random aspects of dataset generation.
        num_objects: int
            Number of total objects to be generated.
        num_distractors: int
            Number of distractors to be generated.
        num_properties: int
            Number of properties of each object.
        num_features: int
            Number of features for each property.
        prev_horizon: int
            Number of recurrence steps for the "previous" variant. Setting this to two makes it possible for the dataset
            to be generated with reference to "previous previous" timestep etc.
        """
        assert num_features > 1, "With num_features = 1, all vectors are 0!"
        self.seed = seed
        self.num_objects = num_objects
        self.num_distractors = num_distractors
        self.num_properties = num_properties
        self.num_features = num_features
        self.dataset_type = dataset_type
        self.repeat_chance = repeat_chance
        self.prev_horizon = prev_horizon

        if self.num_properties * self.num_features < np.iinfo(np.int8).max:
            self.space_dtype = np.int8
        elif self.num_properties * self.num_features < np.iinfo(np.int16).max:
            self.space_dtype = np.int16
        elif self.num_properties * self.num_features < np.iinfo(np.int32).max:
            self.space_dtype = np.int32
        elif self.num_properties * self.num_features < np.iinfo(np.int64).max:
            self.space_dtype = np.int64
        else:
            self.space_dtype = None
            raise OverflowError(
                "The space would be too large to represent, and would result in overflow."
            )

        self.horizon_dtype = np.int32

        self.aux_stats = {}

        self.temporal_labels, self.target_ids, self.dataset = self._generate_dataset()

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int, int]:
        """
        Return the batch item with given index.

        Parameters
        ----------
        index: int
            Index of the item to return.

        Returns
        -------
        item: Tuple[np.ndarray, int, int]
            Objects, target_id and temporal labels from given index.
        """
        return self.dataset[index], self.target_ids[index], self.temporal_labels[index]

    def __len__(self) -> int:
        """
        Return the length of the dataset, which is the number of objects contained within.

        Returns
        -------
        len: int
            Number of objects in the dataset.
        """
        return self.num_objects

    def _generate_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate the dataset, used on __init__().

        Returns
        -------
        dataset: np.ndarray
            Generated dataset array
        """
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
        else:
            rng = default_rng()

        dataset = np.empty(
            (self.num_objects, self.num_distractors + 1, self.num_properties),
            dtype=self.space_dtype,
        )
        target_ids = rng.integers(
            low=0,
            high=self.num_distractors + 1,
            size=self.num_objects,
            dtype=self.space_dtype,
        )
        temporal_labels = np.zeros(self.num_objects, dtype=self.horizon_dtype)
        # If classic referential game then the ordering is random
        if self.dataset_type == "rg_classic":
            for d in range(self.num_objects):
                current_objects = []
                while len(current_objects) < (self.num_distractors + 1):
                    rand_array = rng.integers(
                        size=(1, self.num_properties),
                        low=1,
                        high=self.num_features + 1,
                        dtype=self.space_dtype,
                    )
                    if not any(
                        np.array_equal(x, rand_array[0]) for x in current_objects
                    ):
                        current_objects.append(rand_array[0])
                dataset[d] = np.array(current_objects)
        # If hard referential game then we create very similar objects
        # With only a single property changed between them
        elif self.dataset_type == "rg_hard":
            # If there is only one property, then game is the same as rg_classic
            assert (
                self.num_properties > 1
            ), "rg_hard with one property doesn't make sense, use rg_classic!"
            for d in range(self.num_objects):
                current_objects = []
                rand_array = rng.integers(
                    size=(1, self.num_properties),
                    low=1,
                    high=self.num_features + 1,
                    dtype=self.space_dtype,
                )
                while len(current_objects) < (self.num_distractors + 1):
                    changed_rand_array = rand_array[0]
                    # Change one random property
                    changed_rand_array[
                        rng.integers(0, self.num_properties, 1)[0]
                    ] = rng.integers(1, self.num_features + 1, 1)[0]
                    if not any(
                        np.array_equal(x, changed_rand_array) for x in current_objects
                    ):
                        current_objects.append(changed_rand_array.copy())
                dataset[d] = np.array(current_objects)
        # In the "previous" variant of temporal referential games we add the explicit possibility that an immediately
        # previous target appears again, with a specified % chance. This way we want to encourage agents to
        # just say a phrase which would mean "previous", the same as the temporal logic operator.
        elif self.dataset_type == "trg_previous":
            prev_targets = deque([])
            prev_target = np.empty(self.num_properties, dtype=self.space_dtype)
            for d in range(self.num_objects):
                current_objects = []
                while len(current_objects) < (self.num_distractors + 1):
                    rand_array = rng.integers(
                        size=(1, self.num_properties),
                        low=1,
                        high=self.num_features + 1,
                        dtype=self.space_dtype,
                    )
                    if not any(
                        np.array_equal(x, rand_array[0]) for x in current_objects
                    ):
                        current_objects.append(rand_array[0])
                dataset[d] = np.array(current_objects)
                # 50% chance to be a previously seen object
                if rng.random(dtype=np.float32) < self.repeat_chance and d > 0:
                    # Get a random value for previous recurrence
                    # This means that the target can be from a previous or previous previous or ... timestep.
                    if self.prev_horizon > 1:
                        prev_horizon_target = rng.integers(self.prev_horizon)
                        # We have enough targets to go for this amount of recurrence
                        if len(prev_targets) >= prev_horizon_target:
                            for x in range(1, prev_horizon_target):
                                prev_target = prev_targets.popleft()
                        # Not enough targets, let's do the max amount of recurrence
                        else:
                            prev_horizon_target = len(prev_targets)
                            for x in range(1, prev_horizon_target):
                                prev_target = prev_targets.popleft()

                    # Check if this target is already in the object set
                    # If it is set the new target id to that object
                    # If not then set it to be the new target
                    if not any(np.array_equal(x, prev_target) for x in dataset[d]):
                        dataset[d][target_ids[d]] = prev_target
                    else:
                        target_ids[d] = np.where(
                            (dataset[d] == prev_target).all(axis=1)
                        )[0]

                prev_target = dataset[d][target_ids[d]]
                prev_targets.append(prev_target)
                # Empty queue, so we don't store more than self.prev_horizon, but
                # only if we haven't popped from it recently
                if d >= self.prev_horizon and len(prev_targets) > self.prev_horizon:
                    prev_targets.popleft()
        # In the "sometime in the past" variant of temporal referential games we add the possibility of explicitly
        # repeating a target that has appeared previously. In this way we encourage agents to communicate that this
        # target has appeared before, instead of a full description of the target,
        elif self.dataset_type == "trg_sometime_past":
            for d in range(self.num_objects):
                current_objects = []
                while len(current_objects) < (self.num_distractors + 1):
                    rand_array = rng.integers(
                        size=(1, self.num_properties),
                        low=1,
                        high=self.num_features + 1,
                        dtype=self.space_dtype,
                    )
                    if not any(
                        np.array_equal(x, rand_array[0]) for x in current_objects
                    ):
                        current_objects.append(rand_array[0])
                dataset[d] = np.array(current_objects)
                # 50% chance to be some random object from the past
                if rng.random(dtype=np.float32) < self.repeat_chance and d > 0:
                    old_target = rng.integers(0, d)
                    if not any(
                        np.array_equal(x, dataset[old_target][target_ids[old_target]])
                        for x in dataset[d]
                    ):
                        dataset[d][target_ids[d]] = dataset[old_target][
                            target_ids[old_target]
                        ]
                    else:
                        target_ids[d] = np.where(
                            (
                                dataset[d]
                                == dataset[old_target][target_ids[old_target]]
                            ).all(axis=1)
                        )[0]
        # In the "never in the past" variant of temporal referential games we add the possibility of
        # explicitly creating a target that has never appeared before, up to that point. This encourages the agent
        # to specify in that case that they have not seen this object before.
        elif self.dataset_type == "trg_never_past":
            assert self.num_features**self.num_properties > self.num_objects * (
                self.num_distractors + 1
            ), (
                "Amount of possible objects should be at least double of the objects"
                " to be generated to create trg_never_past."
            )
            for d in range(self.num_objects):
                current_objects = []
                while len(current_objects) < (self.num_distractors + 1):
                    rand_array = rng.integers(
                        size=(1, self.num_properties),
                        low=1,
                        high=self.num_features + 1,
                        dtype=self.space_dtype,
                    )
                    if not any(
                        np.array_equal(x, rand_array[0]) for x in current_objects
                    ):
                        current_objects.append(rand_array[0])
                dataset[d] = np.array(current_objects)
                # % chance to be some random object that has never occurred up to this point
                if rng.random(dtype=np.float32) < self.repeat_chance and d > 0:
                    while True:
                        rand_array = rng.integers(
                            size=(1, self.num_properties),
                            low=1,
                            high=self.num_features + 1,
                            dtype=self.space_dtype,
                        )
                        if not any(
                            np.array_equal(x, rand_array[0])
                            for subset in dataset
                            for x in subset
                        ):
                            dataset[d][target_ids[d]] = rand_array
                            break
        # In the "trg_hard" variant of temporal referential games we add the explicit possibility that an immediately
        # previous target appears again, with a specified % chance. The task is also made harder by objects being
        # very similar to each other.
        elif self.dataset_type == "trg_hard":
            # If there is only one property, then game is the same as rg_classic
            assert (
                self.num_properties > 1
            ), "trg_hard with one property doesn't make sense, use rg_classic!"
            prev_targets = deque([])
            prev_target = np.empty(self.num_properties, dtype=self.space_dtype)
            for d in range(self.num_objects):
                current_objects = []
                rand_array = rng.integers(
                    size=(1, self.num_properties),
                    low=1,
                    high=self.num_features + 1,
                    dtype=self.space_dtype,
                )
                while len(current_objects) < (self.num_distractors + 1):
                    changed_rand_array = rand_array[0]
                    # Change one random property
                    changed_rand_array[
                        rng.integers(0, self.num_properties, 1)[0]
                    ] = rng.integers(1, self.num_features + 1, 1)[0]
                    if not any(
                        np.array_equal(x, changed_rand_array) for x in current_objects
                    ):
                        current_objects.append(changed_rand_array.copy())
                dataset[d] = np.array(current_objects)
                # 50% chance to be a previously seen object
                if rng.random(dtype=np.float32) < self.repeat_chance and d > 0:
                    # Get a random value for previous recurrence
                    # This means that the target can be from a previous or previous previous or ... timestep.
                    if self.prev_horizon > 1:
                        prev_horizon_target = rng.integers(self.prev_horizon)
                        # We have enough targets to go for this amount of recurrence
                        if len(prev_targets) >= prev_horizon_target:
                            for x in range(1, prev_horizon_target):
                                prev_target = prev_targets.popleft()
                        # Not enough targets, let's do the max amount of recurrence
                        else:
                            prev_horizon_target = len(prev_targets)
                            for x in range(1, prev_horizon_target):
                                prev_target = prev_targets.popleft()

                    # Check if this target is already in the object set
                    # If it is set the new target id to that object
                    # If not then set it to be the new target
                    if not any(np.array_equal(x, prev_target) for x in dataset[d]):
                        dataset[d][target_ids[d]] = prev_target
                    else:
                        target_ids[d] = np.where(
                            (dataset[d] == prev_target).all(axis=1)
                        )[0]

                prev_target = dataset[d][target_ids[d]]
                prev_targets.append(prev_target)
                # Empty queue, so we don't store more than self.prev_horizon, but
                # only if we haven't popped from it recently
                if d >= self.prev_horizon and len(prev_targets) > self.prev_horizon:
                    prev_targets.popleft()
        # This game variant os only used for analysis. This will generate a dataset where the target is basically ALWAYS
        # the same, even when the distractors may be different.
        # This will iterate through a subset of possible targets.
        elif self.dataset_type == "analysis_always_same":
            targets = [
                target
                for target in itertools.product(
                    range(1, self.num_features + 1), repeat=self.num_properties
                )
            ]
            sample_size = np.min([len(targets), int(self.num_objects / 10)])
            targets = rng.choice(targets, sample_size, replace=False)
            step = self.num_objects / len(targets)
            switch_points = [round(step * (i + 1)) for i in range(len(targets))]
            current_target = 0
            for d in range(self.num_objects):
                if d == switch_points[current_target]:
                    current_target += 1
                current_objects = []
                while len(current_objects) < (self.num_distractors + 1):
                    rand_array = rng.integers(
                        size=(1, self.num_properties),
                        low=1,
                        high=self.num_features + 1,
                        dtype=self.space_dtype,
                    )
                    if not any(
                        np.array_equal(x, rand_array[0]) for x in current_objects
                    ):
                        current_objects.append(rand_array[0])
                dataset[d] = np.array(current_objects)
                if not any(
                    np.array_equal(x, targets[current_target]) for x in dataset[d]
                ):
                    dataset[d][target_ids[d]] = targets[current_target]
                else:
                    target_ids[d] = np.where(
                        (dataset[d] == targets[current_target]).all(axis=1)
                    )[0]
        # This game variant is only used for analysis. This will generate a dataset where the target is NEVER the same,
        # even when distractors may be different.
        elif self.dataset_type == "analysis_never_same":
            assert (
                self.num_objects <= self.num_features**self.num_properties
            ), "Dataset impossible to generate as num_objects too large"
            targets = [
                target
                for target in itertools.product(
                    range(1, self.num_features + 1), repeat=self.num_properties
                )
            ]
            sample_size = np.min([len(targets), int(self.num_objects)])
            targets = rng.choice(targets, sample_size, replace=False)
            for d in range(self.num_objects):
                current_objects = []
                while len(current_objects) < (self.num_distractors + 1):
                    rand_array = rng.integers(
                        size=(1, self.num_properties),
                        low=1,
                        high=self.num_features + 1,
                        dtype=self.space_dtype,
                    )
                    if not any(
                        np.array_equal(x, rand_array[0]) for x in current_objects
                    ):
                        current_objects.append(rand_array[0])
                dataset[d] = np.array(current_objects)
                if not any(np.array_equal(x, targets[d]) for x in dataset[d]):
                    dataset[d][target_ids[d]] = targets[d]
                else:
                    target_ids[d] = np.where((dataset[d] == targets[d]).all(axis=1))[0]

        # Create temporal labels
        # Create an array of just target objects
        target_array = []
        for idx in range(len(dataset)):
            target_array.append(dataset[idx][target_ids[idx]])

        # Find where all targets repeat and get the indices
        target_array = np.array(target_array)
        vals, inverse, count = np.unique(
            target_array, return_inverse=True, return_counts=True, axis=0
        )
        idx_vals_repeated = np.where(count > 1)[0]
        rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])
        _, inverse_rows = np.unique(rows, return_index=True)
        index_sets = np.split(cols, inverse_rows[1:])

        # Create temporal labels pointing to the last observation of given object
        for idx_set in index_sets:
            first = True
            prev_index = 0
            for idx in idx_set:
                # First time an object appears it will always be 0
                if first:
                    temporal_labels[idx] = 0
                    prev_index = idx
                    first = False
                    continue
                # Find previous index of object and calculate distance
                temporal_labels[idx] = idx - prev_index
                prev_index = idx

        self.aux_stats["over_horizon"] = (temporal_labels > self.prev_horizon).sum()

        # Clear all values greater than the horizon
        # We are not interested in them anyway
        temporal_labels[temporal_labels > self.prev_horizon] = 0

        # Check for repetitions within the request horizon
        self.aux_stats["repetitions"] = (temporal_labels > 0).sum()

        return temporal_labels, target_ids, dataset
