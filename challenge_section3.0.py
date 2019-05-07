# -*- coding: utf-8 -*-


## Author: Elizabeth Trippe
""" This code is based on code written as a tutorial as part of a manuscript by 
James Zou, Mikael Huss, Abubakar Abid, Pejman Mohammadi, Ali Torkamani & Amalio Telentil titled  A Primer on Deep Learning in Genomics (Nature Genetics, 2018) by 

Original file is located at
    https://colab.research.google.com/drive/17E4h5aAOioh5DiTo7MZg4hpL6Z_0FyWr

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from Bio import SeqIO

# import the fasta sequences
records = list(SeqIO.parse("hiv-db-BC-2.fasta", "fasta"))
sequences_raw = []
max_seq_size = 0
for rec in records:
    #print(rec.seq)
    #print(type(rec.seq))
    sequences_raw.append(str(rec.seq))
    if len(rec.seq) > max_seq_size:
        max_seq_size = len(rec.seq)

print("max = ", max_seq_size)

sequences = []
for sequence in sequences_raw:
    seq_temp = sequence
    counter = len(sequence)
    while counter < max_seq_size:
        seq_temp = seq_temp + '-'
        counter = counter + 1
    #print(len(seq_temp))
    seq_temp = seq_temp + 'ACGTURYKMSWBDHVN-'
    sequences.append(seq_temp)



# Let's print the first few sequences.
pd.DataFrame(sequences, index=np.arange(1, len(sequences)+1), 
             columns=['Sequences']).head()

"""Organize data
"""

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.sparse import csr
# The LabelEncoder encodes a sequence of bases as a sequence of integers.
integer_encoder = LabelEncoder()  
# The OneHotEncoder converts an array of integers to a sparse matrix where 
# each row corresponds to one possible value of each feature.
one_hot_encoder = OneHotEncoder(categories='auto')#,handle_unknown="ignore")
#one_hot_encoder = OneHotEncoder(categories=['A','C','G','T','-']),handle_unknown="ignore")
input_features = []

for sequence in sequences:
  integer_encoded = integer_encoder.fit_transform(list(sequence))
  integer_encoded = np.array(integer_encoded).reshape(-1, 1)
  one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
  one_hot_encoded = csr.csr_matrix(one_hot_encoded)
  if one_hot_encoded.shape[1] != 17:
      print(one_hot_encoded.shape[1])
      print(sequence)
  input_features.append(one_hot_encoded.toarray())
                      
np.set_printoptions(threshold=40)
input_features = np.stack(input_features)
print("Example sequence\n-----------------------")
print('DNA Sequence #1:\n',sequences[0][:10],'...',sequences[0][-10:])
print('One hot encoding of Sequence #1:\n',input_features[0].T)

"""Add labels
"""

labels = []
for rec in records:
    if rec.id[0] == 'B':
        labels.append(1)
    else:
        labels.append(0)

#labels = list(filter(None, labels))  # removes empty sequences

one_hot_encoder = OneHotEncoder(categories='auto')
labels = np.array(labels).reshape(-1, 1)
input_labels = one_hot_encoder.fit_transform(labels).toarray()

print('Labels:\n',labels.T)
print('One-hot encoded labels:\n',input_labels.T)

"""split the data into training and test sets.."""

from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(
    input_features, input_labels, test_size=0.25, random_state=42)

""" Select the Architecture and Train using CNN

![alt text](https://github.com/abidlabs/deep-learning-genomics-primer/blob/master/Screenshot%20from%202018-08-01%2020-31-49.png?raw=true)

"""

from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential

model = Sequential()
#model.add(Conv1D(filters=32, kernel_size=12, 
    #             input_shape=(train_features.shape[1], 4)))
model.add(Conv1D(filters=32, kernel_size=12, 
                input_shape=(train_features.shape[1], 17)))   
model.add(MaxPooling1D(pool_size=4))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['binary_accuracy'])
model.summary()

"""divide intor training and validation set"""

history = model.fit(train_features, train_labels, 
                    epochs=50, verbose=0, validation_split=0.25)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

"""Similarly, """

plt.figure()
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

"""## 3. Evaluate
"""

from sklearn.metrics import confusion_matrix
import itertools

predicted_labels = model.predict(np.stack(test_features))
cm = confusion_matrix(np.argmax(test_labels, axis=1), 
                      np.argmax(predicted_labels, axis=1))
print('Confusion matrix:\n',cm)

cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]

plt.imshow(cm, cmap=plt.cm.Blues)
plt.title('Normalized confusion matrix')
plt.colorbar()
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.xticks([0, 1]); plt.yticks([0, 1])
plt.grid('off')
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], '.2f'),
             horizontalalignment='center',
             color='white' if cm[i, j] > 0.5 else 'black')


"""
# Original GitHub Repository

If you found this tutorial helpful, kindly star the [associated GitHub repo](https://github.com/abidlabs/deep-learning-genomics-primer/blob/master/A_Primer_on_Deep_Learning_in_Genomics_Public.ipynb) so that it is more visible to others as well!
"""