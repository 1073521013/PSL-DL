from Bio import SeqIO
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
acid_letters = ['0', 'A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y']
non_amino_letters = ['B', 'J', 'O', 'U', 'X', 'Z']
max_sequence_size=100
x1 = []
x2 = []
fasta_sequences = SeqIO.parse(open('deeploc_data.fasta'), 'fasta')
for fasta in fasta_sequences:
    sequences = []
    name, class0 = fasta.id, fasta.description.split()[1]
    class0 = class0.split('-')[0]
    if 'Extracellular' in class0:
        #y.append(class0)
        sequence = str(fasta.seq)
        sequence = sequence.replace('X', '0')
        sequence = sequence.replace('U', '0')
        sequence = sequence.replace('O', '0')
        sequence = sequence.replace('B', 'N')
        sequence = sequence.replace('Z', 'Q')
        sequence = sequence.replace('J', 'L')
        sequence = list(sequence)
        if len(sequence)<300:
            if len(sequence) >= max_sequence_size:
                a = int((len(sequence) - max_sequence_size))
                for i in list(range(a)):
                    sequence.pop(51)
                sequences.append(sequence)
            else:
                b = int((max_sequence_size - len(sequence)))
                for i in list(range(b)):
                    sequence.insert(int((len(sequence))), '0')
                sequences.append(sequence)
            t=[str(i)+' ' for i in sequences[0]]
            x1.append(''.join(t))

        # t="".join((str(ord(i))+" ") for i in sequences[0])
        # g.write(t.strip())
        # g.write("\n")
    elif  'Nucleus' in class0:
        #y.append(class0)
        sequence = str(fasta.seq)
        sequence = sequence.replace('X', '0')
        sequence = sequence.replace('U', '0')
        sequence = sequence.replace('O', '0')
        sequence = sequence.replace('B', 'N')
        sequence = sequence.replace('Z', 'Q')
        sequence = sequence.replace('J', 'L')
        sequence = list(sequence)
        if len(sequence)<300:
            if len(sequence) >= max_sequence_size:
                a = int((len(sequence) - max_sequence_size))
                for i in list(range(a)):
                    sequence.pop(51)
                sequences.append(sequence)
            else:
                b = int((max_sequence_size - len(sequence)))
                for i in list(range(b)):
                    sequence.insert(int((len(sequence))), '0')
                sequences.append(sequence)
            t = [str(i) + ' ' for i in sequences[0]]
            x2.append(''.join(t))

tokenizer = Tokenizer(num_words=21,lower=False)
tokenizer.fit_on_texts(x1)
data = tokenizer.texts_to_sequences(x1)
print(len(data))
word_index = tokenizer.word_index
print(word_index)
data = pad_sequences(data, maxlen=100,padding='post')
np.save("w2i.npy", word_index)
with open('Extracellular.txt', "w") as f:
    for i in data:
        line = [str(x) + ' ' for x in i]
        f.write("".join(line))
        f.write('\n')


tokenizer = Tokenizer(num_words=21,lower=False)
tokenizer.fit_on_texts(x2)
data = tokenizer.texts_to_sequences(x2)
print(len(data))
word_index = tokenizer.word_index
print(word_index)
data = pad_sequences(data, maxlen=100,padding='post')
np.save("x2_w2i.npy", word_index)
with open('Nucleus.txt', "w") as f:
    for i in data:
        line = [str(x) + ' ' for x in i]
        f.write("".join(line))
        f.write('\n')
