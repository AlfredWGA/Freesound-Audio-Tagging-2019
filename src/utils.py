import numpy as np
import os
from data import feature


def load_fnames(datapath):
    data = []
    for root, path, file_names in os.walk(datapath):
        for filename in file_names:
            data.append(os.path.join(root, filename))
    return np.asarray(data)


def process_from_file(file_list):
    x = []
    y = []
    for file in file_list:
        f = np.load(file)
        x.append(f["log_melgram"])
        y.append(f["labels"])
    return np.asarray(x), np.asarray(y)


'''
Generate the batches from files
Parameters
----------
data_path: data path where the generate load from
batch_size: batch size of each training
shuffle: random shuffle when every training period

return
----------
xx: the data to be input to the net. shape:(batch_size, data_shape)
yy: the ground truth of the data. shape:(batch_size, data_shape)

'''


def generate_arrays_from_file(fname_list, batch_size=64, shuffle=True):
    idx = np.arange(len(fname_list))
    if shuffle:
        np.random.shuffle(idx)
    batches = [idx[range(batch_size * i, min(len(fname_list), batch_size * (i + 1)))] for i in
               range(len(fname_list) // batch_size + 1)]
    while True:
        for i in batches:
            xx, yy = process_from_file(fname_list[i])
            yield (xx, yy)


if __name__ == '__main__':
    fname_list = load_fnames('../data/' + feature.TRAIN_NUMPY_DIR)
    batch = generate_arrays_from_file(fname_list)
    print(next(batch))