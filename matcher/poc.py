import pandas as pd
import os

import tensorflow as tf
from transformers import TFRobertaModel, RobertaTokenizer

import matplotlib.pyplot as plt


DATA_DIR = '../data'

dataset_path = os.path.join(DATA_DIR, 'computers_train', 'computers_train_small.json.gz')
df = pd.read_json(dataset_path, compression='gzip', lines=True)
IDS = 1350
label = df.loc[IDS, 'label']
title_left = df.loc[IDS, 'title_left']
title_right = df.loc[IDS, 'title_right']

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = TFRobertaModel.from_pretrained('roberta-base')

inputs = tokenizer([title_left, title_right], return_tensors='tf', padding=True)

outputs = model(inputs)
representation = outputs[0]

similarity_matrix = tf.tensordot(representation[0], tf.transpose(representation[1]), axes=1)

plt.matshow(similarity_matrix)
plt.show()

print(title_left)
print(title_right)
print('similar' if label == 1 else 'different')


