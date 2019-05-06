# coding=utf-8
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import normalize
from tqdm import tqdm


def show_duration_distribution(path):
    """
    展示所有样本的时长分布

    :param path:
    :return:
    """
    durations = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            y, sr = librosa.load(path, sr=None)
            durations.append(y.shape[0]/sr)

    plt.hist(durations, bins='auto', edgecolor='black')
    plt.title('Duration distribution(Total %d)' % len(durations))
    plt.xlabel('Duration')
    plt.ylabel('Number')
    plt.show()


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


def load_features(vectors, n_mel=64, chunk_size=128, r_threshold=32):
    """Load feature vectors from the specified HDF5 file.
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
        path (str): Path to the HDF5 file.
        chunk_size (int): Size of a chunk.
        r_threshold (int): Threshold for ``r`` (see above).
    Returns:
        np.ndarray: Array of feature vectors.
        list: Number of chunks for each audio clip.
    """
    chunks = []
    n_chunks = []
    feats = vectors
    for i, feat in enumerate(tqdm(feats)):
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


class LogmelExtractor(object):
    """Feature extractor for logmel representations.
    A logmel feature vector is a spectrogram representation that has
    been scaled using a Mel filterbank and a log nonlinearity.
    Args:
        sample_rate (number): Target resampling rate.
        n_window (int): Number of bins in each spectrogram frame.
        hop_length (int): Number of samples between frames.
        n_mels (int): Number of Mel bands.
    Attributes:
        sample_rate (number): Target resampling rate.
        n_window (int): Number of bins in each spectrogram frame.
        hop_length (int): Number of samples between frames.
        mel_fb (np.ndarray): Mel fitlerbank matrix.
    """

    def __init__(self,
                 sample_rate=32000,
                 n_window=1024,
                 hop_length=512,
                 n_mels=64,
                 ):
        self.sample_rate = sample_rate
        self.n_window = n_window
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Create Mel filterbank matrix
        self.mel_fb = librosa.filters.mel(sr=sample_rate,
                                          n_fft=n_window,
                                          n_mels=n_mels,
                                          )

    def output_shape(self, clip_duration):
        """Determine the shape of a logmel feature vector.
        Args:
            clip_duration (number): Duration of the input time-series
                signal given in seconds.
        Returns:
            tuple: The shape of a logmel feature vector.
        """
        n_samples = clip_duration * self.sample_rate
        n_frames = n_samples // self.hop_length + 1
        return n_frames, self.mel_fb.shape[0]

    def extract(self, x, sample_rate=None):
        """Transform the given signal into a logmel feature vector.
        Args:
            x (np.ndarray): Input time-series signal.
            sample_rate (number): New sampling rate of signal.
        Returns:
            np.ndarray: The logmel feature vector.
        """
        # Resample to target sampling rate
        if sample_rate is not None:
            x = librosa.resample(x, sample_rate, self.sample_rate)

        # Compute short-time Fourier transform
        D = librosa.stft(x, n_fft=self.n_window, hop_length=self.hop_length)
        # Transform to Mel frequency scale
        S = np.dot(self.mel_fb, np.abs(D) ** 2).T
        # Apply log nonlinearity and return as float32
        return librosa.power_to_db(S, ref=np.max, top_db=None)


if __name__ == '__main__':
    # show_duration_distribution('./data')
    y, sr = librosa.load('./data/00c91dfc_0.wav', sr=None)
    extractor = LogmelExtractor(sample_rate=32000, n_window=1024, hop_length=512, n_mels=64)
    melgram = extractor.extract(y)
    chunks, n_chunk = load_features(melgram)
    for i in range(5):
        chunk = normalize(chunks[i])

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(chunk,
                                 y_axis='mel',
                                 x_axis='s',
                                 sr=extractor.sample_rate,
                                 hop_length=extractor.hop_length)
        plt.colorbar(format='%+2.5f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        plt.show()
