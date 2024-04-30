"""
Contains all the language metrics, and the reward metrics.

Functions for Mutual information and Positional/BagOfWords Disentanglement adapted from
https://proceedings.neurips.cc/paper/2021/hash/c2839bed26321da8b466c80a032e4714-Abstract.html
"""
from typing import Tuple, Union

import editdistance
import numpy as np
from scipy.spatial import distance
from scipy.stats import entropy, spearmanr


def calc_entropy(x: Union[list, np.ndarray]):
    """
    Calculate the entropy of the given input.

    Parameters
    ----------
    x: Union[list, np.ndarray]
        Input to calculate the entropy for.

    Returns
    -------
    entropy: float
        Entropy of the messages.
    """
    x_s = [str(y) for y in x]
    _, count = np.unique(x_s, return_counts=True)
    return entropy(count, base=2)


def topographic_similarity(
    messages: Union[list, np.ndarray], meanings: Union[list, np.ndarray]
) -> Tuple[float, float]:
    """
    Calculate the topographic similarity between the given messages and meanings.

    Parameters
    ----------
    messages: Union[list, np.ndarray]
        Messages to calculate the mi for.
    meanings: Union[list, np.ndarray]
        Meanings to calculate the mi for.

    Returns
    -------
    topsim: np.ndarray
        Topographic similarity score.
    """
    if len(messages[0]) == 0:
        raise ValueError("Empty messages passed!")

    meanings_dist = distance.pdist(meanings, "hamming")
    # Even though they are ints treat as text
    messages_dist = distance.pdist(
        messages,
        lambda x, y: editdistance.eval(x, y) / ((len(x) + len(y)) / 2),
    )
    topsim, pvalue = spearmanr(meanings_dist, messages_dist, nan_policy="raise")
    return topsim, pvalue


def compute_mutual_information(
    messages: Union[list, np.ndarray], meanings: Union[list, np.ndarray]
) -> float:
    """
    Compute mutual information between the given messages and meanings.

    Parameters
    ----------
    messages: Union[list, np.ndarray]
        Messages to calculate the topsim for.
    meanings: Union[list, np.ndarray]
        Meanings to calculate the topsim for.

    Returns
    -------
    mi: np.ndarray
        Mutual information score.
    """
    meaning_entropy = calc_entropy(meanings)
    message_entropy = calc_entropy(messages)
    messages_and_meanings = np.concatenate((meanings, messages), axis=1)
    messages_and_meanings_joint_entropy = calc_entropy(messages_and_meanings)
    return meaning_entropy + message_entropy - messages_and_meanings_joint_entropy


def posdis(
    messages: Union[list, np.ndarray], meanings: Union[list, np.ndarray]
) -> float:
    """
    Compute Positional Disentanglement between the given messages and meanings.

    Parameters
    ----------
    messages: Union[list, np.ndarray]
        Messages to calculate the posdis for.
    meanings: Union[list, np.ndarray]
        Meanings to calculate the posdis for.

    Returns
    -------
    posdis: float
        Posdis score.
    """
    disentanglement_scores = []
    non_constant_positions = 0

    for j in range(len(messages[0])):
        symbols_j = [message[j] for message in messages]
        symbol_mutual_info = []
        symbol_entropy = calc_entropy(symbols_j)
        for i in range(len(meanings[0])):
            concepts_i = [meaning[i] for meaning in meanings]
            mutual_info = compute_mutual_information(
                np.array([concepts_i]).T, np.array([symbols_j]).T
            )
            symbol_mutual_info.append(mutual_info)
        symbol_mutual_info.sort(reverse=True)

        if symbol_entropy > 0:
            disentanglement_score = (
                symbol_mutual_info[0] - symbol_mutual_info[1]
            ) / symbol_entropy
            disentanglement_scores.append(disentanglement_score)
            non_constant_positions += 1
    if non_constant_positions > 0:
        return sum(disentanglement_scores) / non_constant_positions
    else:
        return float("nan")


def bosdis(
    messages: Union[list, np.ndarray], meanings: Union[list, np.ndarray]
) -> float:
    """
    Compute Bag-of-Words Disentanglement between the given messages and meanings.

    Parameters
    ----------
    messages: Union[list, np.ndarray]
        Messages to calculate the bosdis for.
    meanings: Union[list, np.ndarray]
        Meanings to calculate the bosdis for.

    Returns
    -------
    bosdis: float
        Bosdis score.
    """
    character_set = list(c for message in messages for c in message)
    vocab = {char: idx for idx, char in enumerate(character_set)}
    num_symbols = len(vocab)
    bow_message = []
    bow_meaning = []
    for meaning, message in zip(meanings, messages):
        message_bow = [0 for _ in range(num_symbols)]
        for symbol in message:
            message_bow[list(vocab.keys()).index(symbol)] += 1
        message_bow = [str(symbol) for symbol in message_bow]
        bow_message.append(message_bow)
        bow_meaning.append(meaning)
    return posdis(messages=bow_message, meanings=bow_meaning)
