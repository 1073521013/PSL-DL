# -*- coding: utf-8 -*-
import numpy as np
acid_letters = ['0', 'A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'B', 'J', 'O', 'U', 'X', 'Z']
non_amino_letters = ['B', 'J', 'O', 'U', 'X', 'Z']
from itertools import product
acid = list(product(*[acid_letters] * 3))


with open("uniprot.txt", encoding="utf8") as f, open("uniprot_num.txt", "w", encoding="utf8") as f2:
    for line in f:
        line = line.split()
        x = [acid.index(tuple(list(c))) for c in line]
        x = str(x)
        x = x.replace('[','')
        x = x.replace(']','')
        x = x.replace(',','')
        f2.write(x)
        f2.write('\n')


