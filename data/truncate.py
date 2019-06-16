# coding=utf-8
import numpy as np


def pad_truncate(x, length, pad_value=0):
    """Pad or truncate an array to a specified length.
    Args:
        x (array_like): Input array.
        length (int): Target length.
        pad_value (number): Padding value.
    Returns:
        array_like: The array padded/truncated to the specified length.
    """
    x_len = len(x)
    if x_len > length:
        x = x[:length]
    elif x_len < length:
        padding = np.full((length - x_len,) + x.shape[1:], pad_value)
        x = np.concatenate((x, padding))

    return x


def truncate_features(vector, n_mel=64, chunk_size=128, r_threshold=32):
    """
    Since the original feature vectors are of variable length, this
    function partitions them into chunks of length `chunk_size`. When
    they cannot be partitioned exactly, one of three things can happen:
      * If the length of the vector is less than the chunk size, the
        vector is simply padded with a fill value.
      * If the remainder, ``r``, is less than ``r_threshold``, the edges
        of the vector are truncated so that it can be partitioned.
      * If the remainder, ``r``, is greater than ``r_threshold``, the
        last chunk is the last `chunk_size` frames of the feature vector
        such that it overlaps with the penultimate chunk.
    Args:
        vector (str): A feature vector.
        n_mel (int): n_mel when the vector was generated
        chunk_size (int): Size of a chunk.
        r_threshold (int): Threshold for ``r`` (see above).
    Returns:
        np.ndarray: Array of feature vectors.
        list: Number of chunks for each audio clip.
    """
    chunks = []
    n_chunks = []
    feat = vector
    # Reshape flat array to original shape
    feat = np.reshape(feat, (-1, n_mel))

    # Split feature vector into chunks along time axis
    q = len(feat) // chunk_size
    r = len(feat) % chunk_size
    if not q:
        split = [pad_truncate(feat, chunk_size,
                                    pad_value=np.min(feat))]
    else:
        r = len(feat) % chunk_size
        off = r // 2 if r < r_threshold else 0
        split = np.split(feat[off:q * chunk_size + off], q)
        if r >= r_threshold:
            split.append(feat[-chunk_size:])

    n_chunks.append(len(split))
    chunks += split

    return np.array(chunks), n_chunks