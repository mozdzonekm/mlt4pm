import pandas as pd
import os
import matplotlib.pyplot as plt
from SimilarityMatrix import MatrixBuilder

matrix_builder = MatrixBuilder()

DATA_DIR = '../data'

dataset_path = os.path.join(DATA_DIR, 'computers_train', 'computers_train_small.json.gz')
df = pd.read_json(dataset_path, compression='gzip', lines=True)
label = df.loc[:, 'label'].tolist()
title_left = df.loc[:, 'title_left'].tolist()
title_right = df.loc[:, 'title_right'].tolist()

similarity_matrices = matrix_builder.build_matrices(title_left, title_right)

plt.matshow(similarity_matrices[0])
plt.show()

print(title_left[0])
print(title_right[0])
print('similar' if label[0] == 1 else 'different')


