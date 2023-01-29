import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import xlsxwriter as xlw
import os
import html
import re
import seaborn as sb
import tensorflow as tf
import keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras import layers
from keras.backend import clear_session
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import json
import gzip
import math

# read gz file
# dataset found here http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz
data = pd.read_json("C:\\Users\\kelvi\\Downloads\\reviews_Musical_Instruments_5.json.gz", lines=True)
output_folder = 'C:\\Users\\kelvi\\Desktop\\'
# select the relevant categories for text and how we want to try and train NLP ml
data = data[['reviewText', 'overall']]


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

# apply a text cleaning function to column reviewText
def clean(text):
    # convert html escapes to characters
    text = html.unescape(text)
    # tags like <tab>
    text = re.sub(r'<[^<>]*>', ' ', text)
    # markdown URLs
    text = re.sub(r'\[([^\[\]]*)\]\([^\(\)]*\)', r'\1', text)
    # text in code or brackets
    text = re.sub(r'\[[^\[\]]*\]', ' ', text)
    # standalone sequences of specials
    text = re.sub(r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)', ' ', text)
    # standalone sequences of hyphens
    text = re.sub(r'(?:^|\s)[\-=\+]{2,}(?:\s|$)', ' ', text)
    # sequences of whitespaces
    text = re.sub(r'\s+', ' ', text)
    # make lower case
    text = text.lower()
    
    return text.strip()

data['reviewText'] = data['reviewText'].apply(clean)

# make a custom function which says whether a rating is positive or negative

def review_rating(value):
    if value['overall'] > 3: # a rating of 4 stars or more is viewed as "positive"
        return 1
    elif value['overall'] < 3:
        return 0
    else:
        return 2
    
data['type'] = data.apply(review_rating,axis=1)

# we'll remove the 3 star removes as they're not very helpful to say without investigation as to whether it is positive or negative
data = data[data['type']!=2]

# let's get a breakdown of our dataset so we know how it is comprised
type_counts = data['type'].value_counts()

# this is quite one-sided, let's remove most of the positive to make the dataset more even

data_pos = data[data['type']==1]
data_neg = data[data['type']==0]

data_pos = data_pos[1:800]

data = pd.concat([data_pos,data_neg],axis=0)

class_names=sorted(pd.Series(data['type']).drop_duplicates().tolist())

data_y = data['type']
data_x = data['reviewText']

X_test1 = data['reviewText']

sentences_train, sentences_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.25, random_state=1000)

# get data ready for keras model

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)
X_test1 = tokenizer.texts_to_sequences(X_test1)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 200

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
X_test1 = pad_sequences(X_test1, padding='post', maxlen=maxlen)

clear_session()

embedding_dim = 50

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=maxlen))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train,
                    epochs=15,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)

def value_output(value):
    if value>0.5:
        return 1
    else:
        return 0

# get data predictions for actual test data
preds = model.predict(X_test1)
preds = pd.DataFrame(preds)
preds.rename(columns={0:'preds'},inplace=True)
preds['preds'] = preds['preds'].apply(value_output)


# compare to default data
compar = data.reset_index()
compar = compar.drop(columns=['index'])
compar['preds'] = preds['preds']

# export this to excel
with pd.ExcelWriter(output_folder+"File.xlsx", engine = "xlsxwriter") as writer:
   #do analysis
   compar.to_excel(writer, sheet_name='Test', startcol=0, startrow=0, index=False)
writer.save()