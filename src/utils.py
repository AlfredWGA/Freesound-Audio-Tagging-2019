import numpy as np
import os


def load_file(datapath):
    data = []
    for root, path, file_names in os.walk(datapath):
        for filename in file_names:
            data.append(os.path.join(root, filename))
    return data


def process_from_file(data_list):
    x = []
    y = []
    for file in data_list:
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


def generate_arrays_from_file(data_path, batch_size=64, shuffle=True):
    data, label = load_file(data_path)
    idx = np.arange(len(data))
    if shuffle:
        np.random.shuffle(idx)
    batches = [idx[range(batch_size * i, min(len(data), batch_size * (i + 1)))] for i in
               range(len(data) // batch_size + 1)]
    while True:
        for i in batches:
            xx, yy = process_from_file(data[i])
            yield (xx, yy)
