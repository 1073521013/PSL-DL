# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import pandas as pd
from collections import OrderedDict
from Bio import SeqIO


max_sequence_size = 1204
fasta_sequences = SeqIO.parse(open('uniprot.fasta'), 'fasta')
fasta_data = OrderedDict()

fasta_data['sequence'] = []

s = []
for fasta in fasta_sequences:
    s.append(len(fasta.seq))
    if len(fasta.seq) >= 100:
        sequence = list(str(fasta.seq))
        if len(sequence) >= max_sequence_size:
            a = int((len(sequence) - max_sequence_size))
            for i in list(range(a)):
                sequence.pop(601)
            fasta_data['sequence'].append((sequence))
        else:
            b = int((max_sequence_size - len(sequence)))
            #        if b <= 800:
            for i in list(range(b)):
                sequence.insert(int((len(sequence))), '0')
            fasta_data['sequence'].append((sequence))

print(len(fasta_data['sequence']))
train_full = (pd.DataFrame.from_dict(fasta_data))
train_x = train_full['sequence']
train_x.to_csv('data/train_x.csv', index=False)
train_x = pd.read_csv('data/train_x.csv', header=None)


def csv_list(data):
    k = []
    for m in list(range(len(data))):
        a = data[m]
        x = []
        for i in a:
            for n in list(range(6020)):
                if (n + 3) % 5 == 0:
                    x.append(i[n])
                pass
        #    x=''.join(x)
        k.append(x)
    k = np.array(k)
    return k



# print(list(test_x.values))
x_train = csv_list(list(train_x.values))


acid_letters = ['0', 'A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y']
non_amino_letters = ['B', 'J', 'O', 'U', 'X', 'Z']
from itertools import product, permutations
# acid_letters_trigrams=list(permutations(acid_letters, 3))
acid_letters_trigrams = list(product(*[acid_letters] * 3))


def filter_21(aa):
    for n in range(len(aa)):
        x = aa[n, :]
        for i in range(1204):
            x[i] = x[i].replace('X', '0')
            x[i] = x[i].replace('U', '0')
            x[i] = x[i].replace('O', '0')
            x[i] = x[i].replace('B', 'N')
            x[i] = x[i].replace('Z', 'Q')
            x[i] = x[i].replace('J', 'L')
    return aa

with open('uniprot.txt','w') as f:
    bb=filter_21(x_train)
    for n in range(len(bb)):
        x = bb[n, :]
        trigrams1 = [(x[i], x[i + 1], x[i + 2]) for i in range(1202) if i % 3 == 0]
        buffer = ' '.join(trigrams1) + '\n'
        f.write(buffer)
        trigrams2 = [(x[i + 1], x[i + 2], x[i + 3]) for i in range(1201) if i % 3 == 0]
        buffer = ' '.join(trigrams2) + '\n'
        f.write(buffer)
        trigrams3 = [(x[i + 2], x[i + 3], x[i + 4]) for i in range(1200) if i % 3 == 0]
        buffer = ' '.join(trigrams3) + '\n'
        f.write(buffer)





