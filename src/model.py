from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape, LSTM, RNN, CuDNNLSTM, \
    SimpleRNNCell, SpatialDropout1D, Add, Maximum, Conv2D
from keras.layers import Conv1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D, concatenate, AveragePooling1D, \
    ConvLSTM2D, BatchNormalization, MaxPool2D
from keras.regularizers import l1, l2, l1_l2
from keras import backend as K
import sys
sys.path.append('..')

from data import feature
from metrics import *

def simple_cnn():
    main_input = Input(shape=(feature.img_height, feature.img_width))
    conv = Conv1D(64, 3, padding="same", activation="relu")(main_input)
    conv = Conv1D(64, 3, padding="same", activation="relu")(conv)
    F = Flatten()(conv)
    fc = Dense(128)(F)
    main_output = Dense(feature.class_num, activation="softmax")(fc)
    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[tf_wrapped_lwlrap_sklearn])
    return model


def simple_VGG():
    main_input = Input(shape=(feature.img_height, feature.img_width))
    expanded_input = Lambda(lambda x: K.expand_dims(x, -1))(main_input)
    expanded_input = BatchNormalization()(expanded_input)

    conv = Conv2D(64, 3, padding='same', activation="relu")(expanded_input)
    conv = Conv2D(64, 3, padding='same', activation="relu")(conv)

    pool_1 = MaxPool2D(2, padding='same')(conv)

    conv = Conv2D(128, 3, padding='same', activation="relu")(pool_1)
    conv = Conv2D(128, 3, padding='same', activation="relu")(conv)

    pool_2 = MaxPool2D(2, padding='same')(conv)

    flatten = Flatten()(pool_2)
    fc = Dense(1024, activation='relu',
               kernel_regularizer=l1_l2(),
               bias_regularizer=l1_l2()
               )(flatten)
    fc = Dropout(0.5)(fc)

    main_output = Dense(feature.class_num, activation="softmax")(fc)
    main_output = Dropout(0.5)(main_output)

    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[tf_wrapped_lwlrap_sklearn])
    return model
