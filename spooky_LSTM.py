#!/usr/bin/env python 
from __future__ import print_function 
import os 
import numpy as np 
np.random.seed(2017)

from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical 
from keras.layers import Dense,Input,Flatten
from keras.layers import Conv1D, MaxPooling1D,Embedding,GlobalMaxPooling1D
from keras.layers import Activation,LSTM,SpatialDropout1D
from keras.models import Model 
from keras.optimizers import *
from keras.models import Sequential 
from keras.layers import Merge 
from keras import regularizers
from keras.callbacks import EarlyStopping
import sys

import numpy as np
import pandas as pd

MAX_SEQUENCE_LENGTH = 1000 #max words in a document
MAX_NB_WORDS = 400000 #max words in a dictionary X_SEQUENCE_LENGTH
VALIDATION_SPLIT = 0.2 #percentage of test data 

print('Processing text dataset')

input_df = pd.read_csv("/home/lcai/s2/Halloween/input/train.csv")
author_mapping_dict = {'EAP':0,'HPL':1,'MWS':2}
train_y = input_df['author'].map(author_mapping_dict)
#finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(np.asarray(input_df['text']))
sequences = tokenizer.texts_to_sequences(np.asarray(input_df["text"]))

word_index = tokenizer.word_index
print('Found %s unique tokens in trainning data.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(train_y))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
#x_train = data
#y_train = labels
##prepare test data
test_df = pd.read_csv("/home/lcai/s2/Halloween/input/test.csv")
test_id = test_df['id'].values
tokenizer.fit_on_texts(np.asarray(test_df['text']))
test_sequences = tokenizer.texts_to_sequences(np.asarray(test_df["text"]))
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
x_val = test_data

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_train_val = data[-num_validation_samples:]
y_train_val = labels[-num_validation_samples:]

print('Training model.')

model = Sequential()

model.add(Embedding(MAX_NB_WORDS,128))
model.add(SpatialDropout1D(0.2))

model.add(Conv1D(activation='relu',padding='same',filters=64, kernel_size=5))
model.add(MaxPooling1D(pool_size=4))
###LSTM layer 
model.add(LSTM(128,recurrent_dropout=0.2,dropout=0.2))
model.add(Dense(3,activation='softmax',kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.001)))
#model.add(Activation('sigmoid'))
model.add(Activation('relu'))
    
model.summary()

#try using different optimizers and different optimizer configs
earlyStopping= EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')    
model.compile(loss='categorical_crossentropy',
                #optimizer='Adadelta',
                optimizer='adam',
                metrics=['accuracy'])

model.fit(x_train,y_train,
        validation_data=(x_train_val,y_train_val),
        batch_size=128,
        epochs=10,
        callbacks=[earlyStopping],
        )
#        validation_data=(x_val,y_val))

score = model.evaluate(x_train,y_train,verbose=0)
print('train score:',score[0])
print('train accuracy:',score[1])
'''
score = model.evaluate(x_val,y_val,verbose=0)
print('Test score:',score[0])
print('Test accuracy:',score[1])
'''
classes = model.predict(x_val)
print(classes[1])
out_df = pd.DataFrame(classes)
out_df.columns = ['EAP', 'HPL', 'MWS']
out_df.insert(0, 'id', test_id)
out_df.to_csv("Conv1D_LSTM.csv", index=False)
