# coding=utf-8
import librosa
import shutil
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import os
import csv
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

TRAIN_CURATED_DIR = 'train_curated'
TRAIN_CURATED_NON_SILENCE_DIR = TRAIN_CURATED_DIR + '_non_silence'

TRAIN_NOISY_ALL_DIR = 'train_noisy_all'
TRAIN_NOISY_DIR = 'train_noisy'
TRAIN_NOISY_NON_SILENCE_DIR = TRAIN_NOISY_DIR + '_non_silence'

TRAIN_CURATED_LABEL_PATH = 'train_curated.csv'
# TRAIN_CURATED_NUMPY_PATH = 'train_curated_np.npz'
TRAIN_CURATED_NUMPY_DIR = 'train_curated_np'

TRAIN_NOISY_LABEL_PATH = 'train_noisy.csv'
# TRAIN_NOISY_NUMPY_PATH = 'train_noisy_np.npz'
TRAIN_NOISY_NUMPY_DIR = 'train_noisy_np'

TRAIN_NUMPY_DIR = 'train'

TRAIN_CURATED_NON_SILENT_SIZE = 7942
TRAIN_NOISY_NON_SILENCE_SIZE = 5055
TRAIN_SIZE = 57310

TEST_DIR = 'test'
TEST_NON_SILENCE_DIR = TEST_DIR + '_non_silence'
TEST_NUMPY_PATH = 'test_np'

SAMPLE_PATH = 'sample_submission.csv'

img_height = 128
img_width = 64

classes = ['Accelerating_and_revving_and_vroom', 'Accordion', 'Acoustic_guitar', 'Applause', 'Bark',
            'Bass_drum', 'Bass_guitar', 'Bathtub_(filling_or_washing)', 'Bicycle_bell', 'Burping_and_eructation',
            'Bus', 'Buzz', 'Car_passing_by', 'Cheering', 'Chewing_and_mastication', 'Child_speech_and_kid_speaking',
            'Chink_and_clink', 'Chirp_and_tweet', 'Church_bell', 'Clapping', 'Computer_keyboard', 'Crackle', 'Cricket',
            'Crowd', 'Cupboard_open_or_close', 'Cutlery_and_silverware', 'Dishes_and_pots_and_pans',
            'Drawer_open_or_close', 'Drip', 'Electric_guitar', 'Fart', 'Female_singing',
            'Female_speech_and_woman_speaking', 'Fill_(with_liquid)', 'Finger_snapping', 'Frying_(food)',
            'Gasp', 'Glockenspiel', 'Gong', 'Gurgling', 'Harmonica', 'Hi-hat', 'Hiss', 'Keys_jangling', 'Knock',
            'Male_singing', 'Male_speech_and_man_speaking', 'Marimba_and_xylophone', 'Mechanical_fan',
            'Meow', 'Microwave_oven', 'Motorcycle', 'Printer', 'Purr', 'Race_car_and_auto_racing', 'Raindrop',
            'Run', 'Scissors', 'Screaming', 'Shatter', 'Sigh', 'Sink_(filling_or_washing)', 'Skateboard', 'Slam',
            'Sneeze', 'Squeak', 'Stream', 'Strum', 'Tap', 'Tick-tock', 'Toilet_flush',
            'Traffic_noise_and_roadway_noise', 'Trickle_and_dribble', 'Walk_and_footsteps', 'Water_tap_and_faucet',
            'Waves_and_surf', 'Whispering', 'Writing', 'Yell', 'Zipper_(clothing)']
class_num = len(classes)
class2id = dict(zip(classes, range(class_num)))


def show_duration_distribution(path, ratio):
    """
    展示所有样本的时长分布

    :param path:
    :return:
    """
    durations = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            y, sr = librosa.load(file_path, sr=None)
            durations.append(y.shape[0]/sr)

    plt.hist(durations, bins='auto', edgecolor='black')
    plt.title('Duration distribution(Total %d)' % len(durations))
    plt.xlabel('Duration')
    plt.ylabel('Number')
    plt.show()

    durations = sorted(durations)
    index = int(len(durations)*ratio)
    print('The {} percentile is {}\'s'.format(ratio, durations[index]))
    # 80%分位数为12.8s
    # 70%为8.52s


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


'''
def convert_wav_to_fixed_length_melgram_image(wav_dir, output_dir, extractor):
    """
    把所有的.wav文件转换为定长的log-melgram图片，存到output_dir指定的目录下

    :param wav_dir: The dir containing all the .wav files.
    :param output_dir: Output dir for image file.
    :param extractor: Instance of LogmelExtractor
    :return:
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for dirpath, dirnames, filenames in os.walk(wav_dir):
        for fname in tqdm(filenames):
            x, sr = librosa.load(os.path.join(dirpath, fname), sr=None)
            melgram = extractor.extract(x)
            # melgram = normalize(melgram, axis=0)
            chunks, n_chunk = truncate_features(melgram, n_mel=extractor.n_mels)
            for i, chunk in enumerate(chunks):
                # 把chunk通过matplotlib转为log-melspectrogram图片
                dpi = 100
                fig = plt.figure(figsize=(img_height/dpi, img_width/dpi), dpi=dpi, frameon=False)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                librosa.display.specshow(chunk,
                                         x_axis='mel',
                                         y_axis='s',
                                         sr=extractor.sample_rate,
                                         hop_length=extractor.hop_length)
                plt.tight_layout()
                # plt.show()
                chunk_name = '{}_{}.jpg'.format(fname[:-4], i)
                fig.savefig(output_dir + '/' + chunk_name)
                plt.close()
'''


def convert_wav_to_fixed_length_melgram_npz(wav_dir, output_dir, label_path, extractor):
    """
    把所有的.wav文件转换为定长的log-melgram数组，将数组和one hot标签存入.npz

    :param wav_dir: The dir containing all the .wav files.
    :param output_path: Output path for .npz file.
    :param label_path: .csv file containing fnames and labels
    :param extractor: Instance of LogmelExtractor
    :return: 若testing=True，返回fnames和log-melgram的两个数组
    若False，返回labels和log-melgram
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 读取标签，转换成{fname, label}的键值对
    fname2label = {}
    with open(label_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname2label[row['fname']] = row['labels']

    feature_vectors = []
    labels = []
    fnames = []

    for dirpath, dirnames, filenames in os.walk(wav_dir):
        for fname in tqdm(filenames):
            x, sr = librosa.load(os.path.join(dirpath, fname), sr=None)
            melgram = extractor.extract(x, sample_rate=sr)
            # melgram = normalize(melgram)
            # TODO: 加入delta信息，样本形状为[height, width, 2]
            chunks, n_chunk = truncate_features(melgram, n_mel=extractor.n_mels, chunk_size=img_height)
            for i, chunk in enumerate(chunks):
                # feature_vectors.append(chunk)
                # 处理多个标签的情况
                label = fname2label[fname].split(',')
                one_hot_label = to_one_hot(label)
                chunk_name = '{}_{}'.format(fname[:-4], i)
                output_path = os.path.join(output_dir, chunk_name)
                np.savez(output_path, labels=one_hot_label, log_melgram=chunk)
                # labels.append(one_hot_label)
                # fnames.append(chunk_name)

    print('Save numpy arrays to {}'.format(output_dir))


def convert_test_wav_to_fixed_length_melgram_npz(wav_dir, output_dir, extractor):
    """
    把测试集的.wav文件转换为定长的log-melgram数组和fname，存入.npz

    :param wav_dir: The dir containing all the .wav files.
    :param output_path: Output path for .npz file.
    :param extractor: Instance of LogmelExtractor
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for dirpath, dirnames, filenames in os.walk(wav_dir):
        for fname in tqdm(filenames):
            x, sr = librosa.load(os.path.join(dirpath, fname), sr=None)
            melgram = extractor.extract(x, sample_rate=sr)
            # melgram = normalize(melgram)
            # TODO: 加入delta信息，样本形状为[height, width, 2]
            chunks, n_chunk = truncate_features(melgram, n_mel=extractor.n_mels, chunk_size=img_height)
            for i, chunk in enumerate(chunks):
                chunk_name = '{}_{}'.format(fname[:-4], i)
                output_path = os.path.join(output_dir, chunk_name)
                np.savez(output_path, fname=chunk_name, log_melgram=chunk)

    print('Save numpy arrays to {}'.format(output_dir))


def pick_out_files(wav_dir, output_dir, k_fold):
    """
    给train_noisy分层提取出1/k_fold个样本

    :param wav_dir:
    :param output_dir:
    :param k_fold:
    :return:
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    fnames = []
    labels = []
    fname2label = {}

    with open('train_noisy_all.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fnames.append(row['fname'])
            labels.append(row['labels'])
            fname2label[row['fname']] = row['labels']

    skf = StratifiedKFold(n_splits=k_fold, shuffle=True)

    indices = skf.split(fnames, labels)
    fnames = np.asarray(fnames)
    labels = np.asarray(labels)

    # 只取一个fold
    train_indices, valid_indices = next(indices)
    fnames, labels = fnames[valid_indices], labels[valid_indices]

    for (fname, label) in tqdm(list(zip(fnames, labels))):
        file_path = os.path.join(wav_dir, fname)
        shutil.copy(file_path, output_dir)

    # 生成fname, labels文件
    f = open(TRAIN_NOISY_LABEL_PATH, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['fname', 'labels'])
    for dirpath, dirnames, filenames in os.walk(output_dir):
        for fname in tqdm(filenames):
            writer.writerow([fname, fname2label[fname]])
    f.close()
    return


def to_one_hot(labels):
    """
    生成one_hot标签

    :param labels: 文本标签的list
    :return:
    """
    one_hot_label = np.zeros(shape=[class_num], dtype=np.int32)
    for l in labels:
        one_hot_label[class2id[l]] = 1
    return one_hot_label


def count_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        count = 0
        while True:
            line = f.readline()
            if line == '':
                break
            count += 1
    return count


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

    def extract(self, x, sample_rate):
        """Transform the given signal into a logmel feature vector.
        Args:
            x (np.ndarray): Input time-series signal.
            sample_rate (number): Sampling rate of signal.
        Returns:
            np.ndarray: The logmel feature vector.
        """
        # Resample to target sampling rate
        x = librosa.resample(x, sample_rate, self.sample_rate)

        # Compute short-time Fourier transform
        D = librosa.stft(x, n_fft=self.n_window, hop_length=self.hop_length)
        # Transform to Mel frequency scale
        S = np.dot(self.mel_fb, np.abs(D) ** 2).T
        # Apply log nonlinearity and return as float32
        return librosa.power_to_db(S, ref=np.max, top_db=None)


if __name__ == '__main__':
    extractor = LogmelExtractor(sample_rate=32000, n_window=1024, hop_length=512, n_mels=64)
    convert_wav_to_fixed_length_melgram_npz(TRAIN_CURATED_NON_SILENCE_DIR, TRAIN_CURATED_NUMPY_DIR,
    TRAIN_CURATED_LABEL_PATH, extractor)
    # convert_test_wav_to_fixed_length_melgram_npz(TEST_NON_SILENCE_DIR, TEST_NUMPY_PATH, extractor)
    # pick_out_files(TRAIN_NOISY_ALL_DIR, TRAIN_NOISY_DIR, 4)

