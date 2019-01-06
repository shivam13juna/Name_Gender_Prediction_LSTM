#!/usr/bin/env python
# coding: utf-8

# In[63]:


from __future__ import print_function

from sklearn.preprocessing import OneHotEncoder
from keras.layers.core import Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.models import Model, load_model
from keras.layers import Dense, Embedding, Bidirectional, Input, TimeDistributed
from keras.layers import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
import pandas as pd
import numpy as np
import sys
import os


# In[2]:


#parameters
maxlen = 30
labels = 2


# In[3]:


czech = pd.read_excel('czech.xlsx', encoding='latin',header = None)
czech.columns = ['Name', 'm_or_f']
czech['namelen'] = [len(str(i)) for i in czech['Name']]


# In[4]:


cname = czech['Name']
collect = []


for i in range(len(cname)):
    collect.extend(list(str(cname[i]).lower()))
# collect.extend(['END'])
collect = set(collect)

czech.head()


# In[5]:


data_set = pd.read_csv("gender_data.csv",header=None)
data_set.columns = ['name','m_or_f']
data_set['namelen']= [len(str(i)) for i in data_set['name']]
data_set1 = data_set[(data_set['namelen'] >= 2) ]


# In[6]:


data_set1.groupby('m_or_f')['name'].count()


# In[7]:


names = data_set['name']
gender = data_set['m_or_f']
vocab = set(' '.join([str(i) for i in names]))
vocab.add('END')
vocab = vocab.union(collect)
len_vocab = len(vocab)


# In[8]:


# print(vocab)

# print("vocab length is ",len_vocab)
# print ("length of data_set is ",len(data_set1))


# In[9]:


char_index = dict((c, i) for i, c in enumerate(vocab))


# In[10]:


# print(char_index)


# In[37]:


#train test split
msk = np.random.rand(len(data_set1)) < 0.9
train = data_set1[msk]
test = data_set1[~msk]     


# In[38]:


def set_flag(i):
    tmp = np.zeros(len_vocab);
    tmp[i] = 1
    return(tmp)


# In[39]:


set_flag(3)


# #### modify the code above to also convert each index to one-hot encoded representation

# In[40]:


#take data_set upto max and truncate rest
#encode to vector space(one hot encoding)
#padd 'END' to shorter sequences
#also convert each index to one-hot encoding
train_X = []
train_Y = []
trunc_train_name = [str(i)[0:maxlen] for i in train.name]
for i in trunc_train_name:
    tmp = [set_flag(char_index[j]) for j in str(i)]
    for k in range(0,maxlen - len(str(i))):
        tmp.append(set_flag(char_index["END"]))
    train_X.append(tmp)
for i in train.m_or_f:
    if i == 'm':
        train_Y.append([1,0])
    else:
        train_Y.append([0,1])
    
train_X=np.asarray(train_X)
train_Y=np.asarray(train_Y)


# In[41]:


test_X = []
test_Y = []
trunc_test_name = [str(i)[0:maxlen] for i in test.name]
for i in trunc_test_name:
    tmp = [set_flag(char_index[j]) for j in str(i)]
    for k in range(0,maxlen - len(str(i))):
        tmp.append(set_flag(char_index["END"]))
    test_X.append(tmp)
for i in test.m_or_f:
    if i == 'm':
        test_Y.append([1,0])
    else:
        test_Y.append([0,1])
    
test_X = np.asarray(test_X)
test_Y = np.asarray(test_Y)


# In[42]:


print(np.asarray(test_X).shape)
print(np.asarray(test_Y).shape)


# In[44]:


vtrain_x = []
vtrain_y = []

train_name = [str(i) for i in czech.Name]
for i in train_name:
    tmp = [set_flag(char_index[j]) for j in str(i.lower())]
    for k in range(0, maxlen - len(str(i))):
        tmp.append(set_flag(char_index['END']))
    vtrain_x.append(tmp)
for i in czech.m_or_f:
    if i == 'm':
        vtrain_y.append([1,0])
    else:
        vtrain_y.append([0,1])
vtrain_x = np.asarray(vtrain_x)
vtrain_y = np.asarray(vtrain_y)


# #### build model in keras ( a stacked LSTM model with many-to-one arch ) here 30 sequence and 2 output each for one category(m/f)

# In[45]:


#build the model: 2 stacked LSTM
# print('Build model...')
# input_bilstm=Input(shape = (maxlen,len_vocab))
# bi_one = Bidirectional(LSTM(512, return_sequences=True))(input_bilstm)
# drop1 = Dropout(0.2)(bi_one)
# bi_two = Bidirectional(LSTM(512, return_sequences=False))(drop1)
# drop2 = Dropout(0.2)(bi_two)
# output = Dense(2, activation='softmax')(drop2)
# model = Model(input_bilstm, output)


# optimizer = optimizers.adam(lr = 0.01)
# model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])


# early_stopping = EarlyStopping(monitor='val_acc',patience=10, verbose=1)
# model_checkpoint = ModelCheckpoint('Martin_program.hdf5',monitor='val_acc',save_best_only=True, verbose=1)
# reduce_lr = ReduceLROnPlateau(monitor='val_acc',factor=0.5, patience=5, min_lr=0.0001, verbose=1)
# print('Model Built')
print("Weights are being imported from previously trained model..")
model = load_model('Martin_program.hdf5')
print("Weights imported")

# batch_size=500
# model.fit(train_X, train_Y,
#           batch_size=batch_size,
#           epochs=50,
#           callbacks=[ model_checkpoint,reduce_lr,early_stopping],
#           validation_data=(vtrain_x, vtrain_y),
#           verbose = 2
#          )
# model.save('Martin_program.h5')
# In[47]:


score, acc = model.evaluate(vtrain_x, vtrain_y)
print('Test accuracy:', acc)


# In[75]:


name=sys.argv[1]


# name = ["Shivam"]
X=[]
trunc_name = [i[0:maxlen] for i in name]
for i in trunc_name:
    tmp = [set_flag(char_index[j]) for j in str(i.lower())]
    for k in range(0,maxlen - len(str(i))):
        tmp.append(set_flag(char_index["END"]))
    X.append(tmp)
pred=model.predict(np.asarray(X))
pred = pred.round()[0]
if pred[0] == 1.0:
    print( name[0], ".... is name of a male")
else:
    print(name[0], ".... is name of a female")

#### Lets train more, clearly some very simple female names it doesnt get right like mentioned above (inspite it exists in training data)out = pd.DataFrame(prob_m)
# out['name'] = test.name.reset_index()['name']

# out['m_or_f']=test.m_or_f.reset_index()['m_or_f']out.head(10)
# out.columns = ['prob_m','name','actual']
# out.head(10)
# out.to_csv("gender_pred_out.csv")