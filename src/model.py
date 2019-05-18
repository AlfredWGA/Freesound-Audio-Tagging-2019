from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape, LSTM, RNN, CuDNNLSTM, \
    SimpleRNNCell, SpatialDropout1D, Add, Maximum, Conv2D
from keras.layers import Conv1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D, concatenate, AveragePooling1D, \
    ConvLSTM2D


def simple_cnn():
    main_input = Input(shape=(128, 64))
    conv = Conv1D(64, 3, padding="same", activation="relu")(main_input)
    conv = Conv1D(64, 3, padding="same", activation="relu")(conv)
    F = Flatten()(conv)
    fc = Dense(128)(F)
    main_output = Dense(80, activation="softmax")(fc)
    model = Model(inputs=main_input, outputs=main_output)
    return model
