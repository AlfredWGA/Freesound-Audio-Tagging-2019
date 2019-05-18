import numpy as np
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
import os
from collections import defaultdict


def train(train=True):
    if not os.path.exists("../logs"):
        os.mkdir("../logs")
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
        model_save_path = '../model/model_cnn_{}.h5'.format(str(i))
        if not train:
            model.load_weights(model_save_path)
            print(model.evaluate(X_val, X_val_label))
        else:
            ear = EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode='min', baseline=None,
                                restore_best_weights=True)
            checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, save_weights_only=False)
            history = model.fit(X_train, X_train_label,
                                batch_size=128,
                                epochs=100,
                                shuffle=True,
                                validation_data=(X_val, X_val_label), callbacks=[ear, checkpoint, tensorboard])

            lwlrap += history.history["val_tf_wrapped_lwlrap_sklearn"][-1]
        K.clear_session()
    print(lwlrap / 5.0)


def test():
    test_data = np.load("../data/" + feature.TEST_NUMPY_PATH)
    data = test_data["log_melgram"]  # (None, 128, 64)
    fnames = test_data["fnames"]  # (None)
    config = K.tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = K.tf.Session(config=config)

    # scores给每个模型开一个dict，key是fname，value是[score，times]的二元组
    scores = [{}] * 5
    model = simple_VGG()

    for score in scores:
        for dirpath, dirnames, filenames in os.walk('../model'):
            for i in range(len(filenames)):
                model_save_path = os.path.join(dirpath, filenames[i])
                model.load_weights(model_save_path)
                pred = model.predict(data)

                for i in range(len(fnames)):
                    fname = fnames[i][:8] + '.wav'
                    if fname not in score:
                        score[fname] = [np.zeros(shape=[feature.class_num]), 0]
                    score[fname][0] += pred[i]
                    score[fname][1] += 1

    # key是fname，value是score
    final_scores = {}
    # 将5个模型的结果相加求均值
    for score in scores:
        for fname in score.keys():
            score[fname][0] /= score[fname][1]
            if fname not in final_scores:
                final_scores[fname] = np.zeros([feature.class_num])
            final_scores[fname] += score[fname][0]

    for fname in final_scores.keys():
        final_scores[fname] /= 5

    wf = open('submission.csv', 'w', encoding='utf-8', newline='')
    writer = csv.writer(wf)
    writer.writerow(['fname'] + feature.classes)

    sf = open('../data/' + feature.SAMPLE_PATH, 'r')
    reader = csv.reader(sf)
    next(reader)
    for line in reader:
        fname = line[0]
        writer.writerow([fname] + final_scores[fname].tolist())

    sf.close()
    wf.close()

if __name__ == "__main__":
    train(train=True)
    # test()