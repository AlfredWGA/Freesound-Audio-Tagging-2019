# coding=utf-8
import librosa
import shutil
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import os
import csv
import zipfile
from sklearn.preprocessing import normalize
from data.truncate import *
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

TRAIN_CURATED_DIR = 'train_curated'

TRAIN_NOISY_DIR = 'train_noisy'

TRAIN_CURATED_LABEL_PATH = 'train_curated.csv'
TRAIN_CURATED_NUMPY_DIR = 'train_curated_np'

TRAIN_NOISY_LABEL_PATH = 'trn_noisy_best50s.csv'
TRAIN_NOISY_NUMPY_DIR = 'train_noisy_np'

TRAIN_NUMPY_DIR = 'train_np'

TRAIN_CURATED_NON_SILENT_SIZE = 7942
TRAIN_NOISY_NON_SILENCE_SIZE = 5055
TRAIN_SIZE = 57310

TEST_DIR = 'test'
TEST_NUMPY_DIR = 'test_np'

SAMPLE_PATH = '../data/sample_submission.csv'

sample_data = pd.read_csv(SAMPLE_PATH)
classes = sample_data.columns.values.tolist()[1:]

class_num = len(classes)
class2id = dict(zip(classes, range(class_num)))


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
        # x = librosa.resample(x, sample_rate, self.sample_rate)
#
        # # Compute short-time Fourier transform
        # D = librosa.stft(x, n_fft=self.n_window, hop_length=self.hop_length)
        # # Transform to Mel frequency scale
        # S = np.dot(self.mel_fb, np.abs(D) ** 2).T
        # # Apply log nonlinearity and return as float32
        # return librosa.power_to_db(S, ref=np.max, top_db=None)
        spectrogram = librosa.feature.melspectrogram(x,
                                                 sr=self.sample_rate,
                                                 n_mels=self.n_mels,
                                                 hop_length=self.hop_length,
                                                 n_fft=self.n_window,
                                                 fmin=20)
        spectrogram = librosa.power_to_db(spectrogram)
        spectrogram = spectrogram.astype(np.float32)
        return spectrogram


duration = 2    # 2s
n_mels = 128
img_height = n_mels
extractor = LogmelExtractor(sample_rate=44100, n_window=n_mels*20, hop_length=347*duration, n_mels=n_mels)
img_width = extractor.output_shape(duration)[0]

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


'''
def save_array_as_image(melgram, output_path, extractor):
    # 把chunk通过matplotlib转为log-melspectrogram图片
    dpi = 100
    fig = plt.figure(figsize=(img_width / dpi, img_height / dpi), dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    librosa.display.specshow(melgram,
                             x_axis='mel',
                             y_axis='s',
                             sr=extractor.sample_rate,
                             hop_length=extractor.hop_length)
    plt.tight_layout()
    # plt.show()
    fig.savefig(output_path)
    plt.close()
'''

'''
def convert_npz_to_fixed_length_melgram_image(wav_dir, output_dir, extractor):
    """
    把所有的.npz文件转换为定长的log-melgram图片，存到output_dir指定的目录下

    :param npz_dir: The dir containing all the .npz files.
    :param output_dir: Output dir for image file.
    :param extractor: Instance of LogmelExtractor
    :return:
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for dirpath, dirnames, filenames in os.walk(wav_dir):
        for fname in tqdm(filenames):
            x, sr = librosa.load(os.path.join(dirpath, fname), sr=None)
            melgram = extractor.extract(x, sample_rate=sr)
            chunks, n_chunk = truncate_features(melgram, n_mel=extractor.n_mels, chunk_size=img_height)
            for i, chunk in enumerate(chunks):
                chunk_name = '{}_{}'.format(fname[:-4], i)
                output_path = os.path.join(output_dir, chunk_name)
                save_array_as_image(chunk, output_path, extractor)
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

    for dirpath, dirnames, filenames in os.walk(wav_dir):
        for fname in tqdm(filenames):
            x, sr = librosa.load(os.path.join(dirpath, fname), sr=None)
            melgram = extractor.extract(x, sample_rate=sr)
            chunks, n_chunk = truncate_features(melgram, n_mel=extractor.n_mels, chunk_size=img_width,
                                                r_threshold=int(img_width / 3))
            for i, chunk in enumerate(chunks):
                # 把melgram转为[duration, n_mel]的形状
                chunk_expanded = np.expand_dims(np.transpose(chunk), -1)
                # 处理多个标签的情况
                label = fname2label[fname].split(',')
                one_hot_label = to_one_hot(label)
                chunk_name = '{}_{}'.format(fname[:-4], i)
                output_path = os.path.join(output_dir, chunk_name)
                # feature_vectors.append(chunk_expanded)
                # labels.append(one_hot_label)
                np.savez(output_path, label=one_hot_label, log_melgram=chunk_expanded)
    # return np.asarray(feature_vectors), np.asarray(labels)


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
            chunks, n_chunk = truncate_features(melgram, n_mel=extractor.n_mels, chunk_size=img_width,
                                                r_threshold=int(img_width / 3))
            for i, chunk in enumerate(chunks):
                chunk_expanded = np.expand_dims(np.transpose(chunk), -1)
                chunk_name = '{}_{}'.format(fname[:-4], i)
                output_path = os.path.join(output_dir, chunk_name)
                np.savez(output_path, fname=chunk_name, log_melgram=chunk_expanded)

    print('Save numpy arrays to {}'.format(output_dir))


def extract_file_from_zip(file_list_path, zip_path, output_dir):
    file_list = []
    with open(file_list_path, 'r') as f:
        reader = csv.DictReader(f)
        for item in reader:
            file_list.append(item['fname'])

    with zipfile.ZipFile(zip_path, 'r') as f:
        for file in tqdm(file_list):
            f.extract(file, output_dir)


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


if __name__ == '__main__':
    convert_wav_to_fixed_length_melgram_npz(TRAIN_NOISY_DIR, TRAIN_NOISY_NUMPY_DIR, TRAIN_NOISY_LABEL_PATH, extractor)
