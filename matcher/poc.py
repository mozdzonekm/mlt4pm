import pandas as pd
import os

import tensorflow as tf
from transformers import TFRobertaModel, RobertaTokenizer

import matplotlib.pyplot as plt


class MatrixBuilder:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = TFRobertaModel.from_pretrained('roberta-base')

    def build_matrices(self, input_left, input_right):
        rep_left = self._get_internal_representation(input_left)
        rep_right = self._get_internal_representation(input_right)
        similarity_matrices = [self._build_sim_matrix_from_rep(L, R) for L, R in zip(rep_left, rep_right)]
        return similarity_matrices

    def _get_internal_representation(self, text_input):
        inputs_ids = self.tokenizer(text_input, return_tensors='tf', padding=True)
        model_outputs = self.model(inputs_ids)
        internal_rep = model_outputs[0]
        return internal_rep

    @staticmethod
    def _build_sim_matrix_from_rep(L, R):
        return tf.tensordot(L, tf.transpose(R), axes=1)

    
matrix_builder = MatrixBuilder()

DATA_DIR = '../data'

dataset_path = os.path.join(DATA_DIR, 'computers_train', 'computers_train_small.json.gz')
df = pd.read_json(dataset_path, compression='gzip', lines=True)
label = df.loc[:3, 'label'].tolist()
title_left = df.loc[:3, 'title_left'].tolist()
title_right = df.loc[:3, 'title_right'].tolist()

similarity_matrices = matrix_builder.build_matrices(title_left, title_right)

plt.matshow(similarity_matrices[0])
plt.show()

print(title_left[0])
print(title_right[0])
print('similar' if label[0] == 1 else 'different')


