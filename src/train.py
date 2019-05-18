import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
import time
from sklearn import metrics
from keras import backend as K
from keras.models import load_model
import csv
from sklearn.model_selection import StratifiedKFold
from model import *
from data import feature
from metrics import *
import os


def train(train=True):
    if not os.path.exists("logs"):
        os.mkdir("logs")
    if not os.path.exists("../result"):
        os.mkdir("../result")
    if not os.path.exists("../model"):
        os.mkdir("../model")

    raw_data = np.load("../data/" + feature.TRAIN_CURATED_NUMPY_PATH)
    data = raw_data["log_melgram"]  # (25670, 128, 64)
    label = raw_data["labels"]  # (25670, 80)
    config = K.tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = K.tf.Session(config=config)

    Fname = 'Audio_'
    Time = Fname + str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    tensorboard = TensorBoard(log_dir='./logs/' + Time, histogram_freq=0, write_graph=False, write_images=False,
                              embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    # meta_train = np.zeros(shape=(len(data), 80))
    # meta_test = np.zeros(shape=(len(label), 80))
    y = np.zeros((len(label), 1))
    skf = StratifiedKFold(n_splits=5, random_state=4, shuffle=True)
    lwlrap = 0
    for i, (tr_ind, te_ind) in enumerate(skf.split(data, y)):
        print('FOLD: {}'.format(str(i)))
        print(len(te_ind), len(tr_ind))
        X_train, X_train_label = data[tr_ind], label[tr_ind]
        X_val, X_val_label = data[te_ind], label[te_ind]
        model = simple_cnn()
        print(model.summary())
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=[tf_wrapped_lwlrap_sklearn])
        model_save_path = '../model/model_weight_raw_cnn_{}.h5'.format(str(i))
        if not train:
            model.load_weights(model_save_path)
            print(model.evaluate(X_val, X_val_label))
        else:
            ear = EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode='min', baseline=None,
                                restore_best_weights=True)
            checkpoint = ModelCheckpoint(model_save_path, save_best_only=True,
                                         save_weights_only=True)
            history = model.fit(X_train, X_train_label,
                                batch_size=128,
                                epochs=100,
                                shuffle=True,
                                validation_data=(X_val, X_val_label), callbacks=[ear, checkpoint, tensorboard])

            lwlrap += history.history["val_tf_wrapped_lwlrap_sklearn"][-1]
        K.clear_session()
    print(lwlrap / 5.0)


def test():
    test_data = np.load("../data/" + feature.TRAIN_CURATED_NUMPY_PATH)
    data = test_data["log_melgram"]  # (25670, 128, 64)
    fnames = test_data["fnames"]  # (25670, 80)
    config = K.tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = K.tf.Session(config=config)

    for dirpath, dirnames, filenames in os.walk('../model'):
        for filename in filenames:
            model_save_path = os.path.join(dirpath, filename)
            model = load_model(model_save_path)



if __name__ == "__main__":
    train(train=True)
