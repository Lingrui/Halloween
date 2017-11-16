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
from keras.models import Model 
from keras.optimizers import *
from keras.models import Sequential 
from keras.layers import Merge 
import sys

import numpy as np
import pandas as pd

BASE_DIR = '/home/lcai/s2/Keras_IMDB'  
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR,'20_newsgroup')
MAX_SEQUENCE_LENGTH = 1000 #max words in a document
#MAX_NB_WORDS = 20000 #max words in a dictionary 
MAX_NB_WORDS = 400000 #max words in a dictionary 
EMBEDDING_DIM = 100 #dimention of word vector 
VALIDATION_SPLIT = 0.2 #percentage of test data 

# first, build index mapping words in the embeddings set
# to their embedding vector
print('Indexing word vectors')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR,'glove.6B.100d.txt'))
for line in f: 
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:],dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' %len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')
input_df = pd.read_csv("/home/lcai/s2/Halloween/input/train.csv")
author_mapping_dict = {'EAP':0,'HPL':1,'MWS':2}
train_y = input_df['author'].map(author_mapping_dict)
#finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(np.asarray(input_df['text']))
sequences = tokenizer.texts_to_sequences(np.asarray(input_df["text"]))

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(train_y))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
print("x_train:",x_train[1])
y_train = labels[:-num_validation_samples]
print("y_train:",y_train[1])
x_val = data[-num_validation_samples:]
print("x_val:",x_val[1])
y_val = labels[-num_validation_samples:]
print("y_val:",y_val[1])

print('Preparing embedding matrix.')

num_words = min(MAX_NB_WORDS,len(word_index))
embedding_matrix = np.zeros((num_words,EMBEDDING_DIM))
j = 0     
for word, i in word_index.items():
#if i >= MAX_NB_WORDS:
    if i >= num_words:
        j+=1
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
    #words not found in embedding index will be all-zeros
        embedding_matrix[i] = embedding_vector

#load pre-trained word embeddings into an Embedding layer
#note that we set trainable = Flase so as to keep the embeddings fixed

print("j=",j)
print("num_words =" , num_words)
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length = MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')
'''
#train a 1D convnet with global maxpooling 
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,),dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['acc'])
'''
#train a 1D convnet with global maxpoolinnb_wordsg
#lsft model 
model_left = Sequential()
model_left.add(embedding_layer)
model_left.add(Conv1D(128,5,activation='relu'))
model_left.add(MaxPooling1D(5))
model_left.add(Conv1D(128,5,activation='relu'))
model_left.add(MaxPooling1D(5))
model_left.add(Conv1D(128,5,activation='relu'))
model_left.add(MaxPooling1D(35))
model_left.add(Flatten())

#right model 
model_right = Sequential()
model_right.add(embedding_layer)
model_right.add(Conv1D(128,4,activation='relu'))
model_right.add(MaxPooling1D(4))
model_right.add(Conv1D(128,4,activation='relu'))
model_right.add(MaxPooling1D(4))
model_right.add(Conv1D(128,4,activation='relu'))
model_right.add(MaxPooling1D(58))
model_right.add(Flatten())

#third model 
model_third = Sequential()
model_third.add(embedding_layer)
model_third.add(Conv1D(128,6,activation='relu'))
model_third.add(MaxPooling1D(6))
model_third.add(Conv1D(128,6,activation='relu'))
model_third.add(MaxPooling1D(6))
model_third.add(Conv1D(128,6,activation='relu'))
model_third.add(MaxPooling1D(21))
model_third.add(Flatten())

merged = Merge([model_left,model_right,model_third],mode='concat')
model = Sequential()
model.add(merged)
model.add(Dense(128,activation='relu'))
#model.add(Dense(len(train_y),activation='softmax'))
model.add(Dense(3,activation='softmax'))

model.compile(loss='categorical_crossentropy',
                optimizer='Adadelta',
                metrics=['accuracy'])
model.fit(x_train,y_train,
        batch_size=128,
        epochs=10,
        )
#        validation_data=(x_val,y_val))

score = model.evaluate(x_train,y_train,verbose=0)
print('train score:',score[0])
print('train accuracy:',score[1])
score = model.evaluate(x_val,y_val,verbose=0)
print('Test score:',score[0])
print('Test accuracy:',score[1])

classes = model.predict(x_val)
print(classes[1])
