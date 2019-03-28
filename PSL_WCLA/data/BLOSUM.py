# -*- coding: utf-8 -*-
#import numpy as np
#import pandas as pd  
#txt0 = np.loadtxt('data/BLOSUM.txt',dtype=bytes,skiprows=1).astype(str)
#txtDF0 = pd.DataFrame(txt0).astype(str)
#del txtDF0[0]
#txtDF= pd.DataFrame(np.array(txtDF0)).astype(str) 
#txtDF.to_csv('data/BLOSUM.csv',header = None,index = None)  
#columns=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','B','Z','X','0']
# -*- coding: utf-8 -*-
from Bio import SeqIO
import pandas as pd
from collections import OrderedDict
from keras.utils import np_utils
import numpy as np
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('data/BLOSUM.csv',header =None)
data=np.array(data)

max_sequence_size=1000
fasta_sequences = SeqIO.parse(open('data/deeploc_data.fasta'),'fasta')
fasta_data = OrderedDict()
fasta_data['class'] = []
fasta_data['test0'] = []
fasta_data['sequence'] = []
for fasta in fasta_sequences:
    name,class0 = fasta.id,fasta.description.split()[1]
    class0 = class0.split('-')[0]
    test0 = fasta.description.find('test')
    fasta_data['test0'].append(test0)
    fasta_data['class'].append(class0)
    
    sequence = list(str(fasta.seq))
    if len(sequence) >= max_sequence_size:
        a=int((len(sequence)-max_sequence_size))
        for i in list(range(a)):
            sequence.pop(501)
        fasta_data['sequence'].append((sequence))
    else:
        b=int((max_sequence_size-len(sequence)))
#        if b <= 800:
        for i in list(range(b)):
            sequence.insert(int((len(sequence))),'0')
        fasta_data['sequence'].append((sequence))

print(len(fasta_data['sequence']))
train_full=(pd.DataFrame.from_dict(fasta_data))          
        
train_0 = train_full.ix[train_full.test0==-1]
test_0 = train_full.ix[train_full.test0!=-1]
train = train_0.drop('test0', axis=1)
test = test_0.drop('test0', axis=1)

train_x = train['sequence']
test_x = test['sequence']
train_y = train['class']
test_y = test['class']

train_x.to_csv('data/train_x.csv', index=False)
test_x.to_csv('data/test_x.csv', index=False)
train_y.to_csv('data/train_y.csv', index=False)
test_y.to_csv('data/test_y.csv', index=False)

train_x = pd.read_csv('data/train_x.csv',header =None)
test_x = pd.read_csv('data/test_x.csv',header =None)
train_y = pd.read_csv('data/train_y.csv',header =None)
test_y = pd.read_csv('data/test_y.csv',header =None)
def csv_list(data):
    k=[]
    for m in list(range(len(data))):
        a=data[m]
        x=[]
        for i in a :
            for n in list(range(5000)):
                if (n+3)%5==0:
                    x.append(i[n])
                pass
        k.append(x)
    k=np.array(k)
    return k
#print(list(test_x.values))
x_test = csv_list(list(test_x.values))
x_train = csv_list(list(train_x.values))
le = LabelEncoder()
y_train = np_utils.to_categorical(le.fit_transform(train_y))
y_test = np_utils.to_categorical(le.fit_transform(test_y))

#
def filter_21(aa):
    for n in range(len(aa)):
        x = aa[n,:]
        for i in range(1000):
            x[i] = x[i].replace('X', '0')
            x[i] = x[i].replace('U', '0')
            x[i] = x[i].replace('B', '0')
            x[i] = x[i].replace('Z', '0')            
            x[i] = x[i].replace('O', '0')
            x[i] = x[i].replace('J', '0')
    return aa

x_test = filter_21(x_test)
x_train = filter_21(x_train)

print(x_test.shape)
print(x_train.shape)
columns=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','B','Z','X','0']
#columns=['C','S','T','P','A','G','N','D','E','Q','H','R','K','M','I','L','V','F','Y','W','0']
####mask
def two2three(x):
    xx=[]
    for i,m in enumerate(x):
        k=[]
        for j,n in enumerate(m):
            x=columns.index(n)
            k.append(data[x])
        xx.append(k)
    return np.array(xx)

x_train=two2three(x_train)
x_test=two2three(x_test)

x_test = np.array(x_test)
x_train = np.array(x_train)
print(x_train.shape)
print(x_test.shape)

import h5py
f=h5py.File('data/BLOSUM.hdf5')
#spec_dtype = h5py.special_dtype(vlen=np.dtype('float32'))
d1=f.create_dataset('x_train',data=x_train)
d2=f.create_dataset('x_test',data=x_test)
d3=f.create_dataset('y_train',data=y_train)
d4=f.create_dataset('y_test',data=y_test)



