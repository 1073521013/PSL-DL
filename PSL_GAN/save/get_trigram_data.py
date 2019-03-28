# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import pandas as pd

acid_letters = ['0', 'A', 'C', 'E', 'D', 'G', 'F','I', 'H', 'K',
                'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y']
non_amino_letters = ['B', 'J', 'O', 'U', 'X', 'Z']
from itertools import product, permutations
#acid_letters_trigrams=list(product(*[acid_letters]*3))
acid_letters_trigrams=list(product(*[acid_letters]*3))
print(len(acid_letters_trigrams)) ####9261
with open('deeploc.txt') as f , open('human.txt','w') as w:
    for i,line in enumerate(f):
        if i<=10004:
            line = line.strip('\n')
            line = line.split(" ")
            x=[]
            max_sequence_size = 100
            sequences = []
            if len(line) >= max_sequence_size:
                a=int((len(line)-max_sequence_size))
                for _ in list(range(a)):
                    line.pop(51)
                # sequences.append(line)
            else:
                b=int((max_sequence_size-len(line)))
                for _ in list(range(b)):
                    line.insert(int((len(line))),'000')
                # sequences.append(line)

            for q,t in enumerate(line):
		t = ''.join(t)
		t = t.replace('B','')
		t = t.replace('J','')
		t = t.replace('O','')
		t = t.replace('U','')
		t = t.replace('X','')
		t = t.replace('Z','')
		if len(tuple(t))==3:
                    t = acid_letters_trigrams.index(tuple(t))
                    x.append(str(t)+' ')
            x = (''.join(x))
	    x = x.rstrip()
	    if len(x.split())==100:
                w.write(x)
                w.write("\n")
	    else:
                print(len(x.split()))
                print(x)
        
    #     sequence = list(str(fasta.seq))
    #     if len(sequence) >= max_sequence_size:
    #         a=int((len(sequence)-max_sequence_size))
    #         for i in list(range(a)):
    #             sequence.pop(601)
    #         fasta_data['sequence'].append((sequence))
    #     else:
    #         b=int((max_sequence_size-len(sequence)))
    # #        if b <= 800:
    #         for i in list(range(b)):
    #             sequence.insert(int((len(sequence))),'0')
    #         fasta_data['sequence'].append((sequence))







