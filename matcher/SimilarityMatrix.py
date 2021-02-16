import tensorflow as tf
from transformers import TFRobertaModel, RobertaTokenizer


class MatrixBuilder:
    BATCH_SIZE = 64

    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = TFRobertaModel.from_pretrained('roberta-base')

    def build_matrices(self, input_left, input_right):
        similarity_matrices = []
        for in_L, in_R in self._batch_iterator(input_left, input_right):
            rep_left = self._get_internal_representation(in_L)
            rep_right = self._get_internal_representation(in_R)
            similarity_matrices += [self._build_sim_matrix_from_rep(L, R) for L, R in zip(rep_left, rep_right)]
        return similarity_matrices

    def _batch_iterator(self, input_left, input_right):
        assert len(input_left) == len(input_right)
        dataset_len = len(input_left)
        i = 0
        while (i + 1) * self.BATCH_SIZE < dataset_len:
            beg = i * self.BATCH_SIZE
            end = beg + self.BATCH_SIZE
            yield input_left[beg:end], input_right[beg:end]
            i += 1
        yield input_left[i * self.BATCH_SIZE:], input_right[i * self.BATCH_SIZE:]

    def _get_internal_representation(self, text_input):
        inputs_ids = self.tokenizer(text_input, return_tensors='tf', padding=True)
        model_outputs = self.model(inputs_ids)
        internal_rep = model_outputs.last_hidden_state
        return internal_rep

    @staticmethod
    def _build_sim_matrix_from_rep(L, R):
        return tf.matmul(L, R, transpose_b=True)
