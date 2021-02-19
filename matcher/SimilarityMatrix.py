import tensorflow as tf
from transformers import TFRobertaModel, RobertaTokenizer
from tqdm import tqdm


class MatrixBuilder:
    BATCH_SIZE = 64
    MATRIX_SIZE = (64, 64)

    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        self.model = TFRobertaModel.from_pretrained('roberta-base')

    def build_matrices(self, input_left, input_right):
        dataset_size = len(input_left)
        ta_metrices = []
        batch_count = int(dataset_size / self.BATCH_SIZE) + 1
        for in_L, in_R in tqdm(self._batch_iterator(input_left, input_right), total=batch_count):
            rep_left = self._get_internal_representation(in_L)
            rep_right = self._get_internal_representation(in_R)
            ta_metrices += [self._build_sim_matrix_from_rep(L, R) for L, R in zip(rep_left, rep_right)]
        ta_metrices = tf.convert_to_tensor(ta_metrices)
        return ta_metrices

    def _batch_iterator(self, input_left, input_right):
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

    def _build_sim_matrix_from_rep(self, L, R):
        rep = tf.matmul(L, R, transpose_b=True)
        rep = tf.expand_dims(rep, axis=-1)
        rep = tf.image.per_image_standardization(rep)
        rep = tf.image.resize_with_crop_or_pad(rep, self.MATRIX_SIZE[0], self.MATRIX_SIZE[1])
        return rep

