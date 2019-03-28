# -*- coding: utf-8 -*-
from Bio import SeqIO
import numpy as np
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
# acid_letters = ['0', 'A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y']
acid_letters = ['0', 'A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'B', 'J', 'O', 'U', 'X', 'Z']
non_amino_letters = ['B', 'J', 'O', 'U', 'X', 'Z']
from itertools import product
acid_letters_trigrams = list(product(*[acid_letters] * 3))


y = []
fasta_sequences = SeqIO.parse(open('deeploc_data.fasta'), 'fasta')
for fasta in fasta_sequences:
    name, class0 = fasta.id, fasta.description.split()[1]
    class0 = class0.split('-')[0]
    y.append(class0)

with open("deeploc.txt", encoding="utf8") as f, open("deeploc_num.txt", "w", encoding="utf8") as f2:
    for line in f:
        line = line.split()
        x = [acid_letters_trigrams.index(tuple(list(c))) for c in line]
        x = str(x)
        x = x.replace('[','')
        x = x.replace(']','')
        x = x.replace(',','')
        f2.write(x)
        f2.write('\n')


