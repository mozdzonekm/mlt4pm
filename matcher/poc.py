import matplotlib
import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
from SimilarityMatrix import MatrixBuilder

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.metrics import *

matrix_builder = MatrixBuilder()

DATA_DIR = '../data'

dataset_path = os.path.join(DATA_DIR, 'computers_train', 'computers_train_medium.json.gz')
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
label = df.loc[:, 'label'].tolist()
title_left = df.loc[:, 'title_left'].tolist()
title_right = df.loc[:, 'title_right'].tolist()

similarity_matrices = matrix_builder.build_matrices(title_left, title_right)

exam_matrix = tf.squeeze(similarity_matrices[0])
plt.matshow(exam_matrix)
plt.show()

print(title_left[0])
print(title_right[0])
print('similar' if label[0] == 1 else 'different')

model = Sequential()
model.add(Conv2D(64, 3, activation='selu', input_shape=similarity_matrices[0].shape))
model.add(MaxPooling2D(2))
model.add(Conv2D(64, 3, activation='selu'))
model.add(Dropout(0.05))
model.add(MaxPooling2D(2))

model.add(Conv2D(64, 3, activation='selu'))
model.add(Dropout(0.15))
model.add(MaxPooling2D(2))

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(16, activation='selu'))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
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
    monitor='val_auc',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

X = similarity_matrices
y = tf.convert_to_tensor(label)

h = model.fit(X, y, batch_size=64, epochs=100, validation_split=0.2,
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
                 color=colors[0], linestyle="--", label='Val')
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