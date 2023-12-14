import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Stop console from printing tensorflow info message

import csv
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import zero_one_loss
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Bidirectional, Dense, Conv1D, MaxPooling1D


def preprocess_text(text):
    text = text.lower()                                 # makes text lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)  # removes puncuations and numbers
    text = text.replace('\n', ' ').strip()              # removes newlines and whitespaces
    return text


def build_model(X):
    model = Sequential()                                                                # Allows us to add multiple ways to process the data
    model.add(Embedding(3000, 100, input_length=X))   
    model.add(Conv1D(64, 5, activation='relu'))                               
    model.add(MaxPooling1D(pool_size=5))
    model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))             # LSTM but it takes into account words in front and behind current word. Have been messing around with the numbers 
    model.add(Dense(6, activation='sigmoid')) # 6 labels                                # Experimenting with NN layer where each label returns 1 or 0 (sigmoid or relu)
    model.compile(loss='binary_crossentropy', optimizer='adam')                         # Trying 'binary_crossentropy' and 'mse' for loss, Trying 'sgd' and 'adam' for optimizer
    return model

def main():
    # Load training data
    train = pd.read_csv('train.csv')
    train['comment_text'] = train['comment_text'].apply(preprocess_text)

    # Training Y Values
    TrY = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

    # Training X Values
    tokenizer = Tokenizer(num_words=3000)                               # Creates a tokenizer that will only consider the top n most frequent words. 
    tokenizer.fit_on_texts(train['comment_text'].values)                # Analyzes the text and builds an internal vocabulary based on word frequency
    TrX = tokenizer.texts_to_sequences(train['comment_text'].values)    # Turn words into sequences of integers. Each integer represents the index of a specific word in the vocabulary.
    TrX = pad_sequences(TrX, maxlen=100)                                # Ensures all sequences in the dataset have the same length

    # Test Y Values
    testlabels = pd.read_csv('NewTestLabels.csv')
    TeY = testlabels[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

    # Test X Values
    test = pd.read_csv('NewTest.csv')
    test['comment_text'] = test['comment_text'].apply(preprocess_text)
    tokenizer = Tokenizer(num_words=3000)                               # Same as above Tokenizer
    tokenizer.fit_on_texts(test['comment_text'].values)
    TeX = tokenizer.texts_to_sequences(test['comment_text'].values)
    TeX = pad_sequences(TeX, maxlen=100)
    
    model = build_model(TrX.shape[1])
    model.fit(TrX,TrY)

    #Predictions return numbers between 1 and 0. just a relu operation so that zero_one_loss can calculate errors
    Trpred = model.predict(TrX)
    Trpred = (Trpred > .5).astype(int)
    print(f'Training Error rate: {zero_one_loss(Trpred, TrY)}')
    
    Tepred = model.predict(TeX)
    Tepred = (Tepred > .5).astype(int)
    print(f'Testing Error rate: {zero_one_loss(Tepred, TeY)}')

if __name__ == "__main__":
    main()