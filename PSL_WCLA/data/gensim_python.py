import numpy as np
import h5py
f=h5py.File('data/gensim100.hdf5','r')
x_train=f['x_train']
x_test=f['x_test']
y_train=f['y_train']
y_test=f['y_test']
y_train = np.array(y_train)
x_train = np.array(x_train)
y_test = np.array(y_test)
x_test = np.array(x_test)


indices0 = np.arange(x_test.shape[0])
np.random.shuffle(indices0)
x_test = x_test[indices0]
y_test = y_test[indices0]


#x_test = x_test0[:3000,:,:]
#y_test = y_test0[:3000,:]
#
#x_val = x_test0[3000:,:,:]
#y_val = y_test0[3000:,:]
#x_train = np.concatenate((x_train,x_val))
#y_train = np.concatenate((y_train,y_val))

indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]
print(x_train.shape)
print(y_train.shape)
## In[2]:
from keras.layers import Input, Dense, Dropout,BatchNormalization
from keras.layers import LSTM,concatenate,Conv1D,Bidirectional
from keras.models import  Model,Sequential
from keras_attention import Position_Embedding,Attention
from keras.layers.pooling import GlobalMaxPooling1D
from keras import regularizers
def build_model():
    inputs = Input(shape=(500,100))
    #embedding = Position_Embedding()(embedding)
    #embedding0=SpatialDropout1D(0.2)(embedding0)
    x1 = Conv1D(30, 1, activation='relu', strides=1, padding='same')(inputs)
    x2 = Conv1D(30, 5, activation='relu', strides=1, padding='same')(inputs)
    x3 = Conv1D(30, 9, activation='relu', strides=1, padding='same')(inputs)
    x4 = Conv1D(30, 15, activation='relu', strides=1, padding='same')(inputs)
    x = concatenate([x1, x2,x3,x4])
    #b = GlobalMaxPooling1D()(x)
    a = Bidirectional(LSTM(256, return_sequences=True,dropout=0.3))(x)
    a = Attention()(a)
    #x = concatenate([a,b])
    x = BatchNormalization()(a)
    x = Dropout(0.5)(x)
    x = Dense(200,kernel_regularizer=regularizers.l2(0.01), init="glorot_normal",activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(10,kernel_regularizer=regularizers.l2(0.01), init="glorot_normal",activation='softmax')(x)
    
    model = Model(inputs, output)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#def build_model():
#    model = Sequential()
#    model.add(LSTM(100, return_sequences=True,input_shape=(500, 100)))
#    model.add(Bidirectional(LSTM(100, return_sequences=True,dropout=0.3)))
#    model.add(Attention())
#    model.add(Dense(output_dim=200, input_dim=200,kernel_regularizer=regularizers.l2(0.01), init="glorot_normal",activation='relu'))
#    model.add(BatchNormalization())
#    model.add(Dropout(0.6))
#    model.add(Dense(100, init='glorot_normal', kernel_regularizer=regularizers.l2(0.01),activation='relu'))
#    model.add(BatchNormalization())
#    model.add(Dropout(0.4))
#    model.add(Dense(10,  activation='softmax'))
#    # Compile model
#    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#    model.summary()
#    return model
model=build_model()       
model.fit(x_train, y_train,epochs=200,batch_size=256,shuffle='batch',validation_split=0.05)
score = model.evaluate(x_test, y_test,batch_size=256)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save_weights('weights/predict100_.hdf5')

