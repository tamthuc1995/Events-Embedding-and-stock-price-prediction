#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:01:36 2017

@author: red-sky
"""

import sys
import numpy as np
np.random.seed(280295)
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers, optimizers


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true[:, 1] * y_pred[:, 1], 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true[:, 1], 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true[:, 1] * y_pred[:, 1], 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred[:, 1], 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def fbeta_score(y_true, y_pred):

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = 1 ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def main(dataX_path, dataY_path, result_path,
         n_epoch, input_dim, days):

    # load data
    np.random.seed(2204)
    X = np.load(dataX_path)
    Y = np.load(dataY_path)

    # build Model
    model = Sequential()
    model.add(Conv1D(128, 1, activation='relu', input_shape=(days, input_dim)))
    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dropout(0.8))
    model.add(Dense(2, activation='softmax'))

    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy', recall, precision, fbeta_score])

    # model Compile
    model_name = result_path+'model2_price_move_predict.hdf5'
    checkpointer = ModelCheckpoint(filepath=model_name,
                                   monitor='val_fbeta_score',
                                   verbose=2, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=2)

    outmodel = open(result_path+'model2_price_move_predict.json', 'w')
    outmodel.write(model.to_json())
    outmodel.close()

    # process Training
    model.fit(X, Y, batch_size=32, verbose=2,
              validation_split=0.1, epochs=n_epoch,
              callbacks=[checkpointer])


if __name__ == "__main__":
    dataX = sys.argv[1]
    dataY = sys.argv[2]
    model_path = sys.argv[3]
    n_epoch = int(sys.argv[4])
    input_dim = int(sys.argv[5])
    days = int(sys.argv[6])
    main(dataX, dataY, model_path, n_epoch, input_dim, days)
