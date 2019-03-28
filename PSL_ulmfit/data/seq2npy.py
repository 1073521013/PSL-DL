import numpy as np
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
with open ('deeploc_num.txt') as f:
    x=[]
    for line in f:
        line = line.strip()
        line = line.split(' ')
        x.append(line)

y = []
fasta_sequences = SeqIO.parse(open('deeploc_data.fasta'), 'fasta')
for fasta in fasta_sequences:
    name, class0 = fasta.id, fasta.description.split()[1]
    class0 = class0.split('-')[0]
    y.append(class0)
le = LabelEncoder()
y0 = le.fit_transform(y)
y = []
for t in [3 * [i] for i in y0]:
    for j in t:
        y.append(j)

n = len(x)
trn_index=[]
val_index=[]
index = np.arange(n)
np.random.seed(seed=123456)
np.random.shuffle(index)
print(index)
x=np.array(x)
y=np.array(y)
x = x[index]
y = y[index]
for i in range(n):
    if i%4 ==0:
        val_index.append(i)
    else :
        trn_index.append(i)
print(len(x))
tok_trn = x[trn_index]
tok_val = x[val_index]
lbl_trn = y[trn_index]
lbl_val = y[val_index]

np.save('wiki/ch/tmp/tok_trn.npy',tok_trn)
np.save('wiki/ch/tmp/tok_val.npy',tok_val)
np.save('wiki/ch/tmp/lbl_trn.npy',lbl_trn)
np.save('wiki/ch/tmp/lbl_val.npy',lbl_val)
