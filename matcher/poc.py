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

dataset_path = os.path.join(DATA_DIR, 'computers_train', 'computers_train_small.json.gz')
df = pd.read_json(dataset_path, compression='gzip', lines=True)
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
model.add(Conv2D(64, 3, activation='relu', input_shape=similarity_matrices[0].shape))
model.add(MaxPooling2D(2))
model.add(Conv2D(64, 3, activation='relu'))
model.add(Conv2D(64, 3, activation='relu'))
model.add(MaxPooling2D(2))
model.add(Conv2D(64, 3, activation='relu'))
model.add(Conv2D(64, 3, activation='relu'))
model.add(MaxPooling2D(2))
model.add(Conv2D(64, 3, activation='relu'))
model.add(Conv2D(64, 3, activation='relu'))
model.add(MaxPooling2D(2))
model.add(Flatten())
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

X = similarity_matrices
y = tf.convert_to_tensor(label)

h = model.fit(X, y, batch_size=16, epochs=10, validation_split=0.2)

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