import numpy as np
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
import time
from keras import backend as K
from keras.models import load_model
import csv
from sklearn.model_selection import StratifiedKFold, KFold
from model import *
import os
from metrics import tf_wrapped_lwlrap_sklearn
from utils import *
from keras.utils import plot_model
from data import feature
import pandas as pd
from tqdm import tqdm


models = {'simple_vgg': simple_vgg, 'simple_inception': simple_inception,
          'simple_crnn': simple_crnn, 'xception': xception,
          'inceptionv3': inceptionv3}


def train(model_type):
    if not os.path.exists("../logs"):
        os.mkdir("../logs")
    if not os.path.exists("../result"):
        os.mkdir("../result")
    if not os.path.exists("../model"):
        os.mkdir("../model")

    config = K.tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = K.tf.Session(config=config)
    set_session(session)

    Fname = 'Audio_'
    Time = Fname + str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    tensorboard = TensorBoard(log_dir='../logs/' + Time, histogram_freq=0, write_graph=False, write_images=False,
                              embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    kf = KFold(n_splits=5, shuffle=True)
    fname_list = np.concatenate([load_fnames('../data/' + feature.TRAIN_CURATED_NUMPY_DIR),
                 + load_fnames('../data/' + feature.TRAIN_NOISY_NUMPY_DIR)], axis=0)

    batch_size = 128
    lwlrap = 0.0

    for i, (tr_ind, te_ind) in enumerate(kf.split(fname_list)):
        print('FOLD: {}'.format(str(i)))
        train_size = len(tr_ind)
        test_size = len(te_ind)
        print('Training: {}, validation: {}'.format(len(tr_ind), len(te_ind)))
        fname_train = fname_list[tr_ind]
        fname_test = fname_list[te_ind]
        batch_train = generate_arrays_from_file(fname_train, batch_size, shuffle=True)
        batch_test = generate_arrays_from_file(fname_test, batch_size, shuffle=True)

        model_save_path = '../model/model_{}_{}.h5'.format(model_type,str(i))

        model = models[model_type]()
        print(model.summary())

        if not train:
            # model = load_model(model_save_path, {'tf_wrapped_lwlrap_sklearn': tf_wrapped_lwlrap_sklearn})
            # print(model.evaluate(X_val, X_val_label))
            pass
        else:
            ear = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min', baseline=None,
                                restore_best_weights=True)
            checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, save_weights_only=False)
            # history = model.fit(X_train, X_train_label,
            #                     batch_size=128,
            #                     epochs=100,
            #                     shuffle=True,
            #                     validation_data=(X_val, X_val_label), callbacks=[ear, checkpoint, tensorboard])
            history = model.fit_generator(batch_train,
                                          steps_per_epoch=train_size//batch_size,
                                          epochs=100,
                                          callbacks=[ear, checkpoint, tensorboard],
                                          validation_data=batch_test,
                                          validation_steps=test_size//batch_size)
            lwlrap += history.history["val_tf_wrapped_lwlrap_sklearn"][-1]
        K.clear_session()
    print(lwlrap / 5.0)


def test():
    config = K.tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = K.tf.Session(config=config)

    fname_list = load_fnames('../data/' + feature.TEST_NUMPY_DIR)

    # scores给每个模型开一个dict，key是fname，value是[score，times]的二元组
    scores = [{} for x in range(5)]

    model = xception()
    for dirpath, dirnames, filenames in os.walk('../model'):
        for i in range(len(filenames)):
            model_save_path = os.path.join(dirpath, filenames[i])
            model.load_weights(model_save_path)
            for fname in tqdm(fname_list):
                data = np.load(fname)
                short_fname = os.path.split(fname)[1]
                short_fname = short_fname[:8] + '.wav'
                feature_vector = np.expand_dims(data['log_melgram'], axis=0)

                pred = model.predict(feature_vector)
                if fname not in scores[i]:
                    scores[i][short_fname] = [np.zeros(shape=[feature.class_num]), 0.0]
                scores[i][short_fname][0] += pred[0]
                scores[i][short_fname][1] += 1

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

    print(len(final_scores.keys()))
    sf = open('../data/' + feature.SAMPLE_PATH, 'r')
    reader = csv.reader(sf)
    header = next(reader)
    test_fnames = [line[0] for line in reader]
    sf.close()
    print(len(test_fnames))

    for key in test_fnames:
        if key not in final_scores:
            print(key)

    result = [[test_fname] + final_scores[test_fname].tolist() for test_fname in test_fnames]
    submission = pd.DataFrame(result, columns=header)
    submission.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    train('xception')
    # test()
