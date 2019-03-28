import pandas as pd
import numpy as np
from modlamp.datasets import load_AMPvsUniProt
data0 = load_AMPvsUniProt()
print(data0.target[:-2600])
data = data0.sequences
max_sequence_size = 50

with open("save/amp.txt", "w") as f:
    for n in range(2600):
        sequences = []
        sequence = data[n]
        sequence = sequence.replace('X', '0')
        sequence = sequence.replace('U', '0')
        sequence = sequence.replace('O', '0')
        sequence = sequence.replace('B', 'N')
        sequence = sequence.replace('Z', 'Q')
        sequence = sequence.replace('J', 'L')
        sequence = list(sequence)
	#print(sequence)
        if len(sequence) >= max_sequence_size:
            a = int((len(sequence) - max_sequence_size))
            for i in list(range(a)):
                sequence.pop(26)
            sequences.append(sequence)
        else:
            b = int((max_sequence_size - len(sequence)))
            for i in list(range(b)):
                sequence.insert(int((len(sequence))), '0')
            sequences.append(sequence)
        # print("".join(str(i+" ") for i in sequences[0]))
        f.write("".join((str(ord(i))+" ") for i in sequences[0]))
        f.write("\n")


with open("save/uni.txt", "w") as f:
    for n in range(2600,5200):
	sequences = []
        sequence = data[n]
        sequence = sequence.replace('X', '0')
        sequence = sequence.replace('U', '0')
        sequence = sequence.replace('O', '0')
        sequence = sequence.replace('B', 'N')
        sequence = sequence.replace('Z', 'Q')
        sequence = sequence.replace('J', 'L')
        sequence = list(sequence)
        if len(sequence) >= max_sequence_size:
            a = int((len(sequence) - max_sequence_size))
            for i in list(range(a)):
                sequence.pop(26)
            sequences.append(sequence)
        else:
            b = int((max_sequence_size - len(sequence)))
            for i in list(range(b)):
                sequence.insert(int((len(sequence))), '0')
            sequences.append(sequence)
        # print("".join(str(i+" ") for i in sequences[0]))
        f.write("".join((str(ord(i))+" ") for i in sequences[0]))
        f.write("\n")
