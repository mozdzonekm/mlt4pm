import pandas as pd
import os

import tensorflow as tf
from transformers import TFRobertaModel, RobertaTokenizer

import matplotlib.pyplot as plt


DATA_DIR = '../data'

dataset_path = os.path.join(DATA_DIR, 'computers_train', 'computers_train_small.json.gz')
df = pd.read_json(dataset_path, compression='gzip', lines=True)
label = df.loc[:3, 'label'].tolist()
title_left = df.loc[:3, 'title_left'].tolist()
title_right = df.loc[:3, 'title_right'].tolist()

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = TFRobertaModel.from_pretrained('roberta-base')

inputs_left = tokenizer(title_left, return_tensors='tf', padding=True)
outputs_left = model(inputs_left)
rep_left = outputs_left[0]

inputs_right = tokenizer(title_right, return_tensors='tf', padding=True)
outputs_right = model(inputs_right)
rep_right = outputs_right[0]

similarity_matrices = [tf.tensordot(L, tf.transpose(R), axes=1) for L, R in zip(rep_left, rep_right)]

print(tf.shape(similarity_matrices[0]))
plt.matshow(similarity_matrices[0])
plt.show()

print(title_left[0])
print(title_right[0])
print('similar' if label[0] == 1 else 'different')


