# -*- coding: utf-8 -*-
from Bio import SeqIO
import biovec
import numpy as np
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('glove/vectors.txt', binary=False)
#model = gensim.models.KeyedVectors.load_word2vec_format('glove/vectors.bin', binary=True)
y = []
fasta_sequences = SeqIO.parse(open('data/deeploc_data.fasta'), 'fasta')
for fasta in fasta_sequences:
    name, class0 = fasta.id, fasta.description.split()[1]
    class0 = class0.split('-')[0]
    y.append(class0)

def transform_to_matrix(padding_size=340, vec_size=260):
    res = []
    f = open('data/deeploc.txt')
    for i, line in enumerate(f):
        line = line.split(' ')
        matrix = []
        for m in range(padding_size):
            try:
                matrix.append(model[line[m]].tolist())
            except:
                matrix.append([0] * vec_size)
        res.append(matrix)
    return np.array(res)


def expansion(x):

    a = []
    for i in range(len(x)):
        for n in range(3):
            a.append(i)
    aa = np_utils.to_categorical(le.fit_transform(a))
    return np.dot(aa, x)


x = transform_to_matrix()
print('part one finished')
print(x.shape)

le = LabelEncoder()
y = np_utils.to_categorical(le.fit_transform(y))
y_test = expansion(y)
y = np.array(y_test)
print(y.shape)


import h5py

f = h5py.File('data/new_glove_340_260.hdf5')
d1 = f.create_dataset('x', data=x)
d2 = f.create_dataset('y', data=y)

