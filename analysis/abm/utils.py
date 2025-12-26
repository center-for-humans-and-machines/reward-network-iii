from __future__ import annotations
import numpy as np
import yaml


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def sample_categorical(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    """
    Vectorized categorical sampler.
    probs: (..., K) with probs along last axis
    returns: (...) integer indices in [0, K-1]
    """
    cdf = np.cumsum(probs, axis=-1)
    u = rng.random(size=probs.shape[:-1])[..., None]
    return (u > cdf).sum(axis=-1)


def compute_rle(bits: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """
    bits: (S, L) with {0,1} and -1 padding after length                              # [S,L]
    lengths: (S,)                                                                    # [S]
    returns: RLE complexity (S,)                                                     # [S]
    """
    S, L = bits.shape
    rle = np.zeros(S, dtype=np.int32)
    for i in range(S):
        ell = int(lengths[i])
        if ell == 0:
            rle[i] = 0
            continue
        s = bits[i, :ell]
        rle[i] = 1 + int(np.sum(s[1:] != s[:-1]))
    return rle


def load_config(file_path: str) -> dict:
    with open(file_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config