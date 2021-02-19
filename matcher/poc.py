import matplotlib
import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.metrics import *
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from SimilarityMatrix import MatrixBuilder

matrix_builder = MatrixBuilder()

DATA_DIR = '../data'

dataset_path = os.path.join(DATA_DIR, 'computers_train', 'computers_train_small.json.gz')
df = pd.read_json(dataset_path, compression='gzip', lines=True)

neg, pos = np.bincount(df['label'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))

weight_for_0 = (1 / neg)*(total)/2.0
weight_for_1 = (1 / pos)*(total)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))


df = df.sample(frac=1).reset_index(drop=True)


def preprocess_X_y(dataframe, feature, target):
    label = dataframe.loc[:, target].tolist()
    title_left = dataframe.loc[:, f'{feature}_left'].tolist()
    title_right = dataframe.loc[:, f'{feature}_right'].tolist()
    similarity_matrices = matrix_builder.build_matrices(title_left, title_right)
    X = similarity_matrices
    y = tf.convert_to_tensor(label)
    return X, y


def build_dataset(dataframe, feature='title', target='label'):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    df_train, df_val = None, None
    for train_idx, val_idx in splitter.split(dataframe, dataframe[target]):
        df_train = dataframe.loc[train_idx, :]
        df_val = dataframe.loc[val_idx, :]
    X_train, y_train = preprocess_X_y(df_train, feature, target)
    X_val, y_val = preprocess_X_y(df_val, feature, target)
    return X_train, y_train, X_val, y_val


X_train, y_train, X_val, y_val = build_dataset(df, 'title', 'label')
exam_matrix = tf.squeeze(X_train[0])
plt.matshow(exam_matrix)
plt.show()


model = Sequential()
model.add(Conv2D(64, 3, activation='selu', padding='same', input_shape=X_train[0].shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(2))

model.add(Conv2D(128, 3, activation='selu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Conv2D(128, 3, activation='selu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.05))
model.add(MaxPooling2D(2))

model.add(Conv2D(256, 3, activation='selu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Conv2D(256, 3, activation='selu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.05))
model.add(MaxPooling2D(2))

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dropout(0.15))
model.add(Dense(32, activation='selu', kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=5e-5), metrics=[
    TruePositives(name='tp'),
    FalsePositives(name='fp'),
    TrueNegatives(name='tn'),
    FalseNegatives(name='fn'),
    BinaryAccuracy(name='accuracy'),
    Precision(name='precision'),
    Recall(name='recall'),
    AUC(name='auc'),
])
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=1,
    patience=50,
    restore_best_weights=True)

datagen = ImageDataGenerator(
    height_shift_range=0.2, width_shift_range=0.2,
)
datagen.fit(X_train)
batch_size = 32
h = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
              steps_per_epoch=len(X_train) / batch_size, epochs=500,
              validation_data=(X_val, y_val),
              class_weight=class_weight,
              callbacks=[early_stopping])

matplotlib.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_metrics(history):
    metrics = ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color=colors[1], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()


plot_metrics(h)
plt.show()
