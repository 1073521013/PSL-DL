# -*- coding: utf-8 -*-
from Bio import SeqIO
import pandas as pd
from collections import OrderedDict
from keras.utils import np_utils
import numpy as np
from sklearn.preprocessing import LabelEncoder
import csv
data=[]
with open("data/3grams.csv","r") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        for x in line:
            data.append(x.split('\t'))

data=np.array(data)
acid=list(data[:,0])
datas=np.array(data[:,1:],dtype=np.float32)
x=[]
for i,m in enumerate(acid):
    m=tuple(m)
    x.append(m)
    
acid_letters_trigrams=x

max_sequence_size=902
fasta_sequences = SeqIO.parse(open('data/deeploc_data.fasta'),'fasta')
fasta_data = OrderedDict()
fasta_data['class'] = []
fasta_data['test0'] = []
fasta_data['sequence'] = []


for fasta in fasta_sequences:
    if len(fasta.seq)>=50:
        name,class0 = fasta.id,fasta.description.split()[1]
        class0 = class0.split('-')[0]
        test0 = fasta.description.find('test')
        fasta_data['test0'].append(test0)
        fasta_data['class'].append(class0)
        
        sequence = list(str(fasta.seq))
        if len(sequence) >= max_sequence_size:
            a=int((len(sequence)-max_sequence_size))
            for i in list(range(a)):
                sequence.pop(451)
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
            for n in list(range(4510)):
                if (n+3)%5==0:
                    x.append(i[n])
                pass
    #    x=''.join(x)       
        k.append(x)
    k=np.array(k)
    return k

def expansion(x):
    a=[]
    for i in range(len(x)):
        for n in range(3):
            a.append(i)
    aa=np_utils.to_categorical(le.fit_transform(a))
    return np.dot(aa,x)
#print(list(test_x.values))
x_test = csv_list(list(test_x.values))
x_train = csv_list(list(train_x.values))
le = LabelEncoder()
Y_train = np_utils.to_categorical(le.fit_transform(train_y))
Y_test = np_utils.to_categorical(le.fit_transform(test_y))
y_train = expansion(Y_train)
y_test = expansion(Y_test)

def filter_21(aa):
    for n in range(len(aa)):
        x = aa[n,:]
        for i in range(902):
            x[i] = x[i].replace('U', 'N')
            x[i] = x[i].replace('O', 'Q')
            x[i] = x[i].replace('B', 'N')
            x[i] = x[i].replace('Z', 'Q')
            x[i] = x[i].replace('J', 'L')
            x[i] = x[i].replace('X', 'L')
    return aa

def str_nums(bb):   
     X_all1 = []
     X_all2 = []
     X_all3 = []
#    mask=[]
#    mask1 = np.ones((len(bb),300))
#    mask2 = np.ones((len(bb),300))
#    mask3 = np.ones((len(bb),300))
    #X= list(x)
     for n in range(len(bb)):
        x = bb[n,:]
        trigrams1 = [(x[i], x[i+1], x[i+2]) for i in range(900) if i%3==0]
        trigrams2 = [(x[i+1], x[i+2], x[i+3]) for i in range(899) if i%3==0]
        trigrams3 = [(x[i+2], x[i+3], x[i+4]) for i in range(898) if i%3==0]
        #trigrams = np.vstack((trigrams1,trigrams2,trigrams3))
        x_1=[]
        for i,m in enumerate(trigrams1):
            if m in acid_letters_trigrams:
                x_1.append(acid_letters_trigrams.index(m))
            else:
                x_1.append(-1)
#                mask1[n][i]=0
        x_2=[]
        for i,m in enumerate(trigrams2):
            if m in acid_letters_trigrams:
                x_2.append(acid_letters_trigrams.index(m))
            else:
                x_2.append(-1)      
#                mask2[n][i]=0
        x_3=[]
        for i,m in enumerate(trigrams3):
            if m in acid_letters_trigrams:
                x_3.append(acid_letters_trigrams.index(m))
            else:
                x_3.append(-1)
#               mask3[n][i]=0
        X_all1.append(np.array(x_1))
        X_all2.append(np.array(x_2))
        X_all3.append(np.array(x_3))
#        mask.append(mask1)
#        mask.append(mask2)
#        mask.append(mask3)
#    X_all =  np.array(X_all)
    #mask = np.array(mask)
     return X_all1,X_all2,X_all3
x_test = filter_21(x_test)
x_train = filter_21(x_train)
x_test1,x_test2,x_test3 = str_nums(x_test)
x_train1,x_train2,x_train3 = str_nums(x_train)


#print(mask_test)
#mask_test = np.array(mask_test)
#mask_train = np.array(mask_train)
#print(mask_test.shape)
#print(mask_train.shape)
#print(x_test.shape)
#print(x_train.shape)
####mask
def two2three(x):
    xx=[]
    for i,m in enumerate(x):
        k=[]
        for j,n in enumerate(m):
            if n!=-1:
                k.append(datas[n])
            else:
                k.append(np.zeros((100,)))
        xx.append(k)
    return np.array(xx)

x_train1=two2three(x_train1)
x_test1=two2three(x_test1)
x_train2=two2three(x_train2)
x_test2=two2three(x_test2)
x_train3=two2three(x_train3)
x_test3=two2three(x_test3)

import h5py
f=h5py.File('data/gensim100_1.hdf5')
#spec_dtype = h5py.special_dtype(vlen=np.dtype('float32'))
d0=f.create_dataset('Y_train',data=Y_train)
d1=f.create_dataset('x_train1',data=x_train1)
d2=f.create_dataset('x_test1',data=x_test1)
d3=f.create_dataset('x_train2',data=x_train2)
d4=f.create_dataset('x_test2',data=x_test2)
d5=f.create_dataset('x_train3',data=x_train3)
d6=f.create_dataset('x_test3',data=x_test3)
d7=f.create_dataset('y_train',data=y_train)
d8=f.create_dataset('y_test',data=y_test)
#d5=f.create_dataset('mask_train',data=mask_train)
#d6=f.create_dataset('mask_test',data=mask_test)

