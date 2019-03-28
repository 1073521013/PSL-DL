import xlrd
import pandas as pd
import numpy as np
from collections import OrderedDict
data=xlrd.open_workbook('save/thaliana.xlsx')
table=data.sheets()[0]

x=table.nrows
max_sequence_size = 500

with open("save/human.txt","w") as f:
    for n in range(10000):
        sequences = []
        col = table.row_values(n + 1)
        sequence = list(col[1])
        if len(sequence) >= max_sequence_size:
            a = int((len(sequence) - max_sequence_size))
            for i in list(range(a)):
                sequence.pop(251)
            sequences.append(sequence)
        else:
            b = int((max_sequence_size - len(sequence)))
            for i in list(range(b)):
                sequence.insert(int((len(sequence))), '0')
            sequences.append(sequence)
        # print("".join(str(i+" ") for i in sequences[0]))
        f.write("".join((str(ord(i))+" ") for i in sequences[0]))
        f.write("\n")
