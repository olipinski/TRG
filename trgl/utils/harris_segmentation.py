"""
The Harris' Articulation Scheme based segmentation.

Adapted from "On the Word Boundaries of Emergent Languages Based on Harris's Articulation Scheme",
Ueda et al., ICLR 2023:  https://openreview.net/forum?id=b4t9_XASt6G

Source: https://github.com/wedddy0707/HarrisSegmentation
"""


import itertools
from collections import Counter
from typing import (
    Dict,
    Generic,
    Hashable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import editdistance
import numpy as np
import numpy.typing as npt
from scipy.spatial import distance
from scipy.stats import spearmanr


def compute_topsim(
    messages: Sequence[Sequence[Hashable]],
    meanings: Sequence[Sequence[Hashable]],
    eos_id: int = 0,
):  # noqa D103
    assert len(messages) > 0
    assert len(messages) == len(meanings)
    assert all(len(meanings[0]) == len(meanings[i]) for i in range(len(meanings)))
    messages = [
        (tuple(x)[: tuple(x).index(eos_id)] if eos_id in x else x) for x in messages
    ]
    msg_dist: List[int] = []
    mng_dist: List[int] = []
    for i in range(len(messages)):
        for j in range(i + 1, len(messages)):
            msg_dist.append(editdistance.eval(messages[i], messages[j]))
            mng_dist.append(distance.hamming(meanings[i], meanings[j]))
    topsim: float = spearmanr(msg_dist, mng_dist).correlation
    if np.isnan(topsim):
        topsim = 0
    return topsim


T = TypeVar("T", bound=Hashable)


class EntropyCalculator(Generic[T]):  # noqa D101
    __data: List[Tuple[T, ...]]
    __alpha: Optional[Set[T]]
    __freq: "Optional[Counter[Tuple[T, ...]]]"
    __branching_entropy: Optional[Dict[Tuple[T, ...], float]]
    __conditional_entropy: Optional[Dict[int, float]]
    __boundaries: Optional[List[Set[int]]]
    __segments: Optional[List[Tuple[Tuple[T, ...], ...]]]
    __segment_ids: Optional[Dict[Tuple[T, ...], int]]
    __hashed_segments: Optional[List[Tuple[int, ...]]]
    __random_boundaries: Optional[List[Set[int]]]
    __random_segments: Optional[List[Tuple[Tuple[T, ...], ...]]]
    __random_segment_ids: Optional[Dict[Tuple[T, ...], int]]
    __hashed_random_segments: Optional[List[Tuple[int, ...]]]

    def __init__(
        self,
        data: List[Tuple[T, ...]],
        threshold: float = 0,
        random_seed: int = 0,
        reverse: bool = False,
        verbose: bool = False,
    ):  # noqa D103
        self.reverse = reverse
        self.verbose = verbose
        if self.reverse:
            self.__data = [tuple(reversed(d)) for d in data]
        else:
            self.__data = [tuple(d) for d in data]
        self.__reset_on_init()
        self.threshold = threshold
        self.random_seed = random_seed

    def __reset_on_init(self):  # noqa D103
        self.__alpha = None
        self.__freq = None
        self.__branching_entropy = None
        self.__conditional_entropy = None
        self.__reset_on_setting_threshold()
        self.__reset_on_setting_random_seed()

    def __reset_on_setting_threshold(self):  # noqa D103
        self.__boundaries = None
        self.__segments = None
        self.__segment_ids = None
        self.__hashed_segments = None
        self.__reset_on_setting_random_seed()

    def __reset_on_setting_random_seed(self):  # noqa D103
        self.__random_boundaries = None
        self.__random_segments = None
        self.__random_segment_ids = None
        self.__hashed_random_segments = None

    @property
    def threshold(self) -> float:  # noqa D103
        return self.__threshold

    @threshold.setter
    def threshold(self, x: float):  # noqa D103
        self.__threshold = x
        self.__reset_on_setting_threshold()

    @property
    def random_seed(self) -> int:  # noqa D103
        return self.__random_seed

    @random_seed.setter
    def random_seed(self, x: int):  # noqa D103
        self.__random_seed = x
        self.__reset_on_setting_random_seed()

    @property
    def data(self) -> List[Tuple[T, ...]]:  # noqa D103
        return self.__data

    @property
    def alpha(self) -> Set[T]:  # noqa D103
        if self.__alpha is None:
            self.__alpha = set(itertools.chain.from_iterable(self.data))
        return self.__alpha

    @property
    def freq(self) -> "Counter[Tuple[T, ...]]":  # noqa D103
        if self.__freq is None:
            # get frequencies of non-empty sequences.
            self.__freq = Counter(
                s[i:j]
                for s in self.data
                for i in range(len(s))
                for j in range(i + 1, len(s) + 1)
            )
            # The frequency of empty sequence is defined as follows.
            # This is just for the convenience.
            self.__freq[tuple()] = sum(len(s) for s in self.data)
        return self.__freq

    @property
    def branching_entropy(
        self,
    ) -> Dict[Tuple[T, ...], float]:  # noqa D103
        if self.__branching_entropy is None:
            self.__branching_entropy = dict()
            for context, context_freq in self.freq.items():
                succ_freq_list = [self.freq[context + (a,)] for a in self.alpha]
                # if sum(succ_freq_list) == 0:
                #     continue
                self.__branching_entropy[context] = (
                    -1
                    * sum(
                        succ_freq * (np.log2(succ_freq) - np.log2(context_freq))
                        for succ_freq in succ_freq_list
                        if succ_freq > 0
                    )
                    / context_freq
                )
        return self.__branching_entropy

    @property
    def conditional_entropy(
        self,
    ) -> Dict[int, float]:  # noqa D103
        if self.__conditional_entropy is None:
            self.__conditional_entropy = dict()
            length_to_total_freq: Dict[int, int] = dict()
            for seq, ent in self.branching_entropy.items():
                seq_len = len(seq)
                if seq_len not in self.__conditional_entropy:
                    self.__conditional_entropy[seq_len] = 0
                if seq_len not in length_to_total_freq:
                    length_to_total_freq[seq_len] = 0
                self.__conditional_entropy[seq_len] += self.freq[seq] * ent
                length_to_total_freq[seq_len] += self.freq[seq]
            for length, total_freq in length_to_total_freq.items():
                self.__conditional_entropy[length] /= total_freq
        return self.__conditional_entropy

    @property
    def boundaries(self) -> List[Set[int]]:  # noqa D103
        if self.__boundaries is None:
            self.__boundaries = []
            for d in self.data:
                self.__boundaries.append(set())
                start: int = 0
                width: int = 2
                """
                We begin with width=2, while the algorithm in the paper begins with width=1.
                It is because this code block assumes that self.branching_entropy is already computed.
                """
                while start < len(d):
                    context = d[start : start + width]
                    if (
                        self.branching_entropy[context]
                        - self.branching_entropy[context[:-1]]
                        > self.threshold
                    ):
                        self.__boundaries[-1].add(start + width)
                    if start + width + 1 < len(d):
                        width = 1 + width
                    else:
                        start = 1 + start
                        width = 2
        return self.__boundaries

    @property
    def segments(self) -> List[Tuple[Tuple[T, ...], ...]]:  # noqa D103
        if self.__segments is None:
            segs: List[List[Tuple[T, ...]]] = []
            for data, boundaries in zip(self.data, self.boundaries):
                segs.append([])
                bot = 0
                for top in sorted(boundaries | {len(data)}):
                    word = data[bot:top]
                    bot = top
                    segs[-1].append(word)
            self.__segments = [tuple(x) for x in segs]
        return self.__segments

    @property
    def segment_ids(self):  # noqa D103
        if self.__segment_ids is None:
            self.__segment_ids = {
                s: i + 1
                for i, s in enumerate(set(itertools.chain.from_iterable(self.segments)))
            }
        return self.__segment_ids

    @property
    def hashed_segments(self):  # noqa D103
        if self.__hashed_segments is None:
            self.__hashed_segments = [
                tuple(self.segment_ids[x] for x in s) for s in self.segments
            ]
        return self.__hashed_segments

    @property
    def random_boundaries(self) -> List[Set[int]]:  # noqa D103
        if self.__random_boundaries is None:
            rng = np.random.default_rng(seed=self.random_seed)
            self.__random_boundaries = [
                set(
                    rng.choice(
                        np.arange(1, len(data), dtype=np.int_), size=len(boundaries)
                    )
                )
                for data, boundaries in zip(self.data, self.boundaries)
            ]
        return self.__random_boundaries

    @property
    def random_segments(self) -> List[Tuple[Tuple[T, ...], ...]]:  # noqa D103
        if self.__random_segments is None:
            segs: List[List[Tuple[T, ...]]] = []
            for data, boundaries in zip(self.data, self.random_boundaries):
                segs.append([])
                bot = 0
                for top in sorted(boundaries | {len(data)}):
                    word = data[bot:top]
                    bot = top
                    segs[-1].append(word)
            self.__random_segments = [tuple(x) for x in segs]
        return self.__random_segments

    @property
    def random_segment_ids(self):  # noqa D103
        if self.__random_segment_ids is None:
            self.__random_segment_ids = {
                s: i + 1
                for i, s in enumerate(
                    set(itertools.chain.from_iterable(self.random_segments))
                )
            }
        return self.__random_segment_ids

    @property
    def hashed_random_segments(self):  # noqa D103
        if self.__hashed_random_segments is None:
            self.__hashed_random_segments = [
                tuple(self.random_segment_ids[x] for x in s)
                for s in self.random_segments
            ]
        return self.__hashed_random_segments

    @property
    def n_boundaries(self) -> List[int]:  # noqa D103
        return [len(b) for b in self.boundaries]

    @property
    def mean_n_boundaries(self) -> float:  # noqa D103
        return np.mean(self.n_boundaries)

    @property
    def vocab_size(self) -> int:  # noqa D103
        return len(self.segment_ids)


def standard_error_of_mean(
    x: Union[npt.NDArray[np.float_], Sequence[float]]
) -> npt.NDArray[np.float_]:  # noqa D103
    x = np.array(x)
    return np.std(x, ddof=1) / np.sqrt(np.size(x))


def generate_random_synthetic_language(
    n_attributes: int,
    n_values: int,
    max_len: int,
    vocab_size: int,
    random_seed: Optional[int],
):  # noqa D103
    attval_to_segment: Dict[Tuple[int, int], Tuple[int, ...]] = {}
    assert max_len >= n_attributes
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()
    segment_len = max_len // n_attributes
    for a in range(n_attributes):
        for v in range(n_values):
            segment = tuple(rng.choice(vocab_size, segment_len))
            attval_to_segment[a, v] = segment
    synthetic_language: List[Tuple[int, ...]] = []
    for values in itertools.product(range(n_values), repeat=n_attributes):
        message = sum((attval_to_segment[a, v] for a, v in enumerate(values)), start=())
        synthetic_language.append(message)

    return synthetic_language, attval_to_segment
