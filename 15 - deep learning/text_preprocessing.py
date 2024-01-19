import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#just a simple test

sentences =  [
    'I like eggs and ham.',
    'I love chocolate and bunnies.',
    'I hate onions.'
]

MAX_VOCAB_SIZE = 2000
tokenizer = Tokenizer(num_words= MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

print(sequences)

tokenizer.word_index

data = pad_sequences(sequences)

print(data)

MAX_SEQUENCE_LENGTH = 5
data = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGTH)
print(data)

data = pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH, padding ='post')
print(data)

## adicionar um preenchimento em excesso

data = pad_sequences(sequences, maxlen=6)
print(data)

#truncation - caber no comprimento máximo especificado , início da sequência foi cortado

data = pad_sequences(sequences, maxlen= 4 )
print(data)

## Spam Detection CNN - sequences

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

df = pd.read_csv('spam.csv',encoding='ISO-8859-1')

df.head()

#drop unnecessary columns

df = df.drop(['Unnamed: 2', "Unnamed: 3", "Unnamed: 4"],axis = 1 )

df.head()

df.columns = ['labels', 'data']

df.head()

# create binary labels
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1 })
Y = df['b_labels'].values

#split up the data
df_train, df_test, Ytrain, Ytest = train_test_split(df['data'], Y, test_size=0.33)

#Convert sentences to sequences
MAX_VOCAB_SIZE = 2000
tokenizer = Tokenizer(num_words = MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(df_train)
sequences_train = tokenizer.texts_to_sequences(df_train)
sequences_test = tokenizer.texts_to_sequences(df_test)

# get word --> integer mapping
word2idx = tokenizer.word_index
V = len(word2idx)
print('Found %s unique tokens.' % V)

# pad sequences so that we get a N x T matrix
data_train = pad_sequences(sequences_train)
print('Shape of data train tensor:', data_train.shape)

#get sequence length
T = data_train.shape[1]

data_test = pad_sequences(sequences_test,maxlen=T)
print('Shape of data test tensor:', data_test.shape)

# create the model

D = 20
i = Input(shape=(T,))
x = Embedding(V+1,D)(i)
x = Conv1D(32, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(64, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128,3, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(1,activation='sigmoid')(x)

model = Model(i, x)

#compile and fit
model.compile(
    loss='binary_crossentropy',
    optimizer = 'adam',
    metrics=['accuracy']
)

print('training model ...')
r = model.fit(
    data_train,
    Ytrain,
    epochs = 5 ,

    validation_data =(data_test,Ytest)
    )

import matplotlib.pyplot as plt
plt.plot(r.history['loss'],label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

plt.plot(r.history['accuracy'],label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()