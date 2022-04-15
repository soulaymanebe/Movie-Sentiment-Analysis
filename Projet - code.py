import re  # import the package re

import nltk  # import the package nltk
import numpy as np  # import the package numpy as np
np.random.seed(42)
import pandas as pd  # import the package pandas as pd
from nltk.corpus import \
    stopwords  # import the stopwords from  the package nltk.corpus
from numpy import array  # import array from numpy
from sklearn.model_selection import \
    train_test_split  # import train_test_split from the package sklearn.model_selection
from tensorflow import keras  # import keras from tensorflow
from tensorflow.keras.layers import \
    LSTM  # import LSTM from the package keras.preprocessing.layers
from tensorflow.keras.layers import \
    Conv1D  # import Conv1Dfrom the package keras.preprocessing.layers
from tensorflow.keras.layers import \
    Embedding  # import Embedding from the package keras.preprocessing.layers
from tensorflow.keras.layers import \
    Flatten  # import Flatten from the package keras.preprocessing.layers
from tensorflow.keras.layers import \
    GlobalMaxPooling1D  # import GlobalMaxPooling1D from the package keras.preprocessing.layers
from tensorflow.keras.layers import (  # import Activation,Dropout,dense from the package keras.preprocessing.layers
    Activation, Dense, Dropout)
from tensorflow.keras.models import \
    Sequential  # import sequentiel from the package keras.preprocessing.models
from tensorflow.keras.preprocessing.sequence import \
    pad_sequences  # import pad_sequences from the package keras.preprocessing.sequence
from tensorflow.keras.preprocessing.text import \
    Tokenizer  # import Tokeniz from the package tensorflow.keras.preprocessing.text
from tensorflow.keras.preprocessing.text import \
    one_hot  # import one_hot from the package keras.preprocessing.text

 # import and analyze our dataset.

movie_reviews = pd.read_csv("IMDB Dataset.csv")#read_csv() method of the pandas library to read the CSV file containing our dataset

movie_reviews.isnull().values.any()#we check if the dataset contains any NULL value or not

movie_reviews.shape#print the shape of our dataset.

movie_reviews.head() #print the first 5 rows of the dataset using the head() method
movie_reviews["review"][3]
import seaborn as sns

sns.countplot(x='sentiment', data=movie_reviews)

def preprocess_text(sen): #method the first step is to remove the HTML tags
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence
   
TAG_RE = re.compile(r'<[^>]+>') 

def remove_tags(text):  # function simply replaces anything between opening and closing <> with an empty space
    return TAG_RE.sub('', text)

# preprocessing our reviews and  storing  them in a new list  
X = [] 
sentences = list(movie_reviews['review'])
for sen in sentences:
    X.append(preprocess_text(sen))
X[3] # see the fourth review

y = movie_reviews['sentiment']

y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y))) #replacing "positive" with digit 1 and negative with digit 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)# divides our data into 80% for the training set and 20% for the testing set

#Preparing the Embedding Layer To converts our textual data into numeric data 
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)#Tokenizer class from the keras.preprocessing.text module to create a word-to-index dictionary

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1 #setting the size of the vocabulary 

maxlen = 100 #setting the maximum size of each list to 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

from numpy import array  # import of package array from numpy
from numpy import asarray  # import of package asarray from numpy
from numpy import zeros  # import of package zeros from numpy

#use GloVe embeddings to create our feature matrix
embeddings_dictionary = dict() # create a dictionnary 
glove_file = open('glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 100))#embedding_matrix will contain 92547 rows 
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector



#Text Classification with Simple Neural Network
model = Sequential() #using the model sequentiel 
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False) #we create our embedding layer
model.add(embedding_layer)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

#use of the adam optimizer
#binary_crossentropy as our loss function
#accuracy as metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc']) 

print(model.summary())#print the summary of our mode
#use the fit method to train our neural network
#The validation_split of 0.2 means that 20% of the training data is used to find the training accuracy of the algorithm
history = model.fit(X_train, y_train, batch_size=128, epochs=1, verbose=1, validation_split=0.2)
# evaluate the model
score = model.evaluate(X_test, y_test, verbose=1)
#checking the test accuracy and loss
print("Test Score:", score[0])
print("Test Accuracy:", score[1]) 
#to plot the loss and accuracy differences for training and test sets
import matplotlib.pyplot as plt

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()


#Text Classification with a Convolutional Neural Network

model = Sequential() #we create a sequential model

embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)#embedding layer
model.add(embedding_layer)

model.add(Conv1D(128, 5, activation='relu'))#creation a one-dimensional convolutional layer with 128 kernels
model.add(GlobalMaxPooling1D())# addition of a global max pooling layer to reduce feature size
model.add(Dense(1, activation='sigmoid'))#activation with sigmoid function 
#use of the adam optimizer
#binary_crossentropy as our loss function
#accuracy as metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
# train and test our model 
history = model.fit(X_train, y_train, batch_size=128, epochs=1, verbose=1, validation_split=0.2)

score = model.evaluate(X_test, y_test, verbose=1)
#checking the test accuracy and loss
print("Test Score:", score[0])
print("Test Accuracy:", score[1])
#to plot the loss and accuracy differences for training and test sets
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()

#Text Classification with Recurrent Neural Network (LSTM)

model = Sequential() #initializing a sequential model
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False) #creation of the embedding layer
model.add(embedding_layer)
model.add(LSTM(128)) # create an LSTM layer with 128 neurons

model.add(Dense(1, activation='sigmoid'))#activation with sigmoid function
#use of the adam optimizer
#binary_crossentropy as our loss function
#accuracy as metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
# train and test our model 
history = model.fit(X_train, y_train, batch_size=128, epochs=1, verbose=1, validation_split=0.2)

score = model.evaluate(X_test, y_test, verbose=1)
#checking the test accuracy and loss
print("Test Score:", score[0])
print("Test Accuracy:", score[1])
#to plot the loss and accuracy differences for training and test sets
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

#Making Predictions on Single Instance

instance = X[57] # random select of review from the corpus
print(instance)
# convert this review into numeric form
instance = tokenizer.texts_to_sequences(instance) # text_to_sequences method will convert the sentence into its numeric counter part
# pad our input sequence
flat_list = []
for sublist in instance:
    for item in sublist:
        flat_list.append(item)

flat_list = [flat_list]

instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)

model.predict(instance)# predict method of our model and pass it our processed input sequence
