from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape, LSTM, RNN, CuDNNLSTM, \
    SimpleRNNCell, SpatialDropout1D, Add, Maximum, Conv2D
from keras.layers import Conv1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D, concatenate, AveragePooling1D, \
    ConvLSTM2D, BatchNormalization, MaxPool2D, Concatenate
from keras.regularizers import l1, l2, l1_l2
from keras.optimizers import Adam, SGD
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

    conv = Conv2D(32, 3, padding='same')(expanded_input)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(32, 3, padding='same')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    pool_1 = MaxPool2D(2, padding='same')(conv)

    conv = Conv2D(64, 3, padding='same')(pool_1)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(64, 3, padding='same')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    pool_2 = MaxPool2D(2, padding='same')(conv)

    conv = Conv2D(128, 3, padding='same')(pool_2)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(128, 3, padding='same')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    pool_3 = MaxPool2D(2, padding='same')(conv)

    conv = Conv2D(256, 3, padding='same')(pool_3)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(256, 3, padding='same')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    pool_4 = MaxPool2D(2, padding='same')(conv)

    flatten = Flatten()(pool_4)
    fc = Dense(512, activation='relu',
               kernel_regularizer=l2(),
               bias_regularizer=l2()
               )(flatten)
    fc = Dropout(0.5)(fc)

    main_output = Dense(feature.class_num, activation="softmax")(fc)
    main_output = Dropout(0.5)(main_output)

    model = Model(inputs=main_input, outputs=main_output)
    optimizer = Adam(lr=5e-4)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=[tf_wrapped_lwlrap_sklearn])
    return model


def InceptionBlock(x):
    out_1 = Conv2D(64, 1, padding='same')(x)

    out_2 = Conv2D(96, 1, padding='same')(x)
    out_2 = Conv2D(128, 3, padding='same')(out_2)

    out_3 = Conv2D(16, 1, padding='same')(x)
    out_3 = Conv2D(32, 5, padding='same')(out_3)

    out_4 = MaxPool2D(3, strides=1, padding='same')(x)
    out_4 = Conv2D(32, 1, padding='same')(out_4)

    output = Concatenate()([out_1, out_2, out_3, out_4])

    return output


def simple_Inception():
    main_input = Input(shape=(feature.img_height, feature.img_width))
    expanded_input = Lambda(lambda x: K.expand_dims(x, -1))(main_input)

    expanded_input = BatchNormalization()(expanded_input)
    conv = Conv2D(32, 7, strides=2, padding='same')(expanded_input)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    pool_1 = MaxPool2D(3, padding='same')(conv)

    conv = Conv2D(64, 3, padding='same')(pool_1)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    pool_2 = MaxPool2D(3, padding='same')(conv)

    inception = InceptionBlock(pool_2)

    pool_3 = MaxPool2D(3, padding='same')(inception)

    flatten = Flatten()(pool_3)
    fc = Dense(256, activation='relu',
               kernel_regularizer=l2(),
               bias_regularizer=l2()
               )(flatten)
    fc = Dense(512, activation='relu',
               kernel_regularizer=l2(),
               bias_regularizer=l2()
               )(fc)
    fc = Dropout(0.5)(fc)

    main_output = Dense(feature.class_num, activation="softmax")(fc)
    main_output = Dropout(0.5)(main_output)

    model = Model(inputs=main_input, outputs=main_output)
    optimizer = Adam(lr=5e-4)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=[tf_wrapped_lwlrap_sklearn])
    return model


