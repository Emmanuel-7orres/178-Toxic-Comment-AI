import csv
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import zero_one_loss
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Bidirectional, Dense


def preprocess_text(text):
    text = text.lower()                                 # makes text lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)  # removes puncuations and numbers
    text = text.replace('\n', ' ').strip()              # removes newlines and whitespaces
    return text

# All of this is VERY experimental, its EXTREMELY slow to train and predict
def build_model(X):
    model = Sequential()                                                                # Allows us to add multiple ways to process the data
    model.add(Embedding(5000, 100, input_length=X))                                     # Fixed errors when trying Bidirectional
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
    tokenizer = Tokenizer(num_words=5000)                               # Creates a tokenizer that will only consider the top n most frequent words. 
    tokenizer.fit_on_texts(train['comment_text'].values)                # Analyzes the text and builds an internal vocabulary based on word frequency
    TrX = tokenizer.texts_to_sequences(train['comment_text'].values)    # Turn words into sequences of integers. Each integer represents the index of a specific word in the vocabulary.
    TrX = pad_sequences(TrX, maxlen=100)                                # Ensures all sequences in the dataset have the same length

    # Test Y Values
    testlabels = pd.read_csv('test_labels.csv')
    TeY = testlabels[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

    # Test X Values
    test = pd.read_csv('test.csv')
    test['comment_text'] = test['comment_text'].apply(preprocess_text)
    tokenizer = Tokenizer(num_words=5000)                               # Same as above Tokenizer
    tokenizer.fit_on_texts(test['comment_text'].values)
    TeX = tokenizer.texts_to_sequences(test['comment_text'].values)
    TeX = pad_sequences(TrX, maxlen=100)
    
    model = build_model(TrX.shape[1])
    model.fit(TrX,TrY)

    print(f'Training Error rate: {zero_one_loss(model.predict(TrX), TrY)}')
    print(f'Testing Error rate: {zero_one_loss(model.predict(TeX), TeY)}')

if __name__ == "__main__":
    main()