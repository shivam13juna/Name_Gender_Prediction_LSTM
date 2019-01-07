
from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model, load_model, model_from_json
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, Input, TimeDistributed
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import optimizers
import pandas as pd
from sklearn.metrics import r2_score
import numpy as np
import sys
import h5py
import pickle
import os



#parameters
maxlen = 30 #This is the maximum length of characters in names we're gonna deal with
labels = 2  #Two labels, that is male or female


#Reading Czech Names
czech = pd.read_excel('czech.xlsx', encoding='latin',header = None)
czech.columns = ['Name', 'male_or_female']
czech['namelen'] = [len(str(i)) for i in czech['Name']]

cname = czech['Name']
collect = [] #Gonna create a set of all characters in a name, for one hot encoding


for i in range(len(cname)):
    collect.extend(list(str(cname[i]).lower()))
collect = set(collect)

#Now we're importing our main Data Set
data_set = pd.read_csv("gender_data.csv",header=None)
data_set.columns = ['name','male_or_female']
data_set['namelen']= [len(str(i)) for i in data_set['name']]
data_set1 = data_set[(data_set['namelen'] >= 2) ]



data_set1.groupby('male_or_female')['name'].count()

names = data_set['name']
gender = data_set['male_or_female']
vocab = set(' '.join([str(i) for i in names]))
vocab.add('END')
vocab = vocab.union(collect)
len_vocab = len(vocab)



# print(vocab)    #If you want, this cell can be executed to see all the characters in our vocab
# print("vocab length is ",len_vocab)
# print ("length of data_set is ",len(data_set1))



char_index = dict((c, i) for i, c in enumerate(vocab))

# #We've to execute this first time we're training, as we need to have same set of key-vlaue paid all the time, 
# I wasted an entire day because I didn't take this measuer, 
# checkpoints restored after training model just won't give the same result as everytime the order of key-value pair assigned were different. 


# with open('char_index.txt', 'wb') as handle:  
#     pickle.dump(char_index, handle)

with open('char_index.txt', 'rb') as handle:
    char_index = pickle.loads(handle.read())


# print(char_index)



#train test split
msk = np.random.rand(len(data_set1)) < 0.9
train = data_set1[msk]
test = data_set1[~msk]     


# In[13]:


def set_flag(i): #For creating one hot encoding
    tmp = np.zeros(len_vocab); 
    tmp[i] = 1
    return(tmp)



# #### modify the code above to also convert each index to one-hot encoded representation

#take data_set upto max and truncate rest
#encode to vector space(one hot encoding)
#padd 'END' to shorter sequences
#also convert each index to one-hot encoding
train_x = []
train_y = []
trunc_train_name = [str(i)[0:maxlen] for i in train.name]
for i in trunc_train_name:
    tmp = [set_flag(char_index[j]) for j in str(i)]
    for k in range(0,maxlen - len(str(i))):
        tmp.append(set_flag(char_index["END"]))
    train_x.append(tmp)
for i in train.male_or_female:
    if i == 'm':
        train_y.append([1,0])
    else:
        train_y.append([0,1])
    
train_x=np.asarray(train_x)
train_y=np.asarray(train_y)


# In[16]:


test_x = []
test_y = []
trunc_test_name = [str(i)[0:maxlen] for i in test.name]
for i in trunc_test_name:
    tmp = [set_flag(char_index[j]) for j in str(i)]
    for k in range(0,maxlen - len(str(i))):
        tmp.append(set_flag(char_index["END"]))
    test_x.append(tmp)
for i in test.male_or_female:
    if i == 'm':
        test_y.append([1,0])
    else:
        test_y.append([0,1])
    
test_x = np.asarray(test_x)
test_y = np.asarray(test_y)



# print(np.asarray(test_x).shape)
# print(np.asarray(test_y).shape)




vtrain_x = []
vtrain_y = []

train_name = [str(i) for i in czech.Name]
for i in train_name:
    tmp = [set_flag(char_index[j]) for j in str(i.lower())]
    for k in range(0, maxlen - len(str(i))):
        tmp.append(set_flag(char_index['END']))
    vtrain_x.append(tmp)
for i in czech.male_or_female:
    if i == 'm':
        vtrain_y.append([1,0])
    else:
        vtrain_y.append([0,1])
vtrain_x = np.asarray(vtrain_x)
vtrain_y = np.asarray(vtrain_y)


# #### build model in keras ( a stacked LSTM model with many-to-one arch ) here 30 sequence and 2 output each for one category(m/f)

#We've already built the model, we don't need to rebuild it, we can just restore checkpoints



# #build the model: 2 stacked LSTM
# print('Build model...')
# input_bilstm=Input(shape = (maxlen,len_vocab))
# bi_one = Bidirectional(LSTM(512, return_sequences=True))(input_bilstm)
# drop1 = Dropout(0.2)(bi_one)
# bi_two = Bidirectional(LSTM(512, return_sequences=False))(drop1)
# drop2 = Dropout(0.2)(bi_two)
# output = Dense(2, activation='softmax')(drop2)
# model = Model(input_bilstm, output)


# optimizer = optimizers.Adam(lr = 0.01)
# model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])


# checkpoint_path = 'tmp/model.ckpt'
# early_stopping = EarlyStopping(monitor='val_acc',patience=10, verbose=1)
# model_checkpoint = ModelCheckpoint(checkpoint_path,monitor='val_acc',save_best_only=True, verbose=1)
# reduce_lr = ReduceLROnPlateau(monitor='val_acc',factor=0.5, patience=5, min_lr=0.0001, verbose=1)
# print('Model Built')


# While training we trained for like 20 epochs, with batch_size of 500


# batch_size=500
# model.fit(train_x, train_y,
#           batch_size=batch_size,
#           epochs=20,
#           callbacks=[model_checkpoint,reduce_lr,early_stopping],
#           validation_data=(vtrain_x, vtrain_y),
#           verbose = 2
#          )

#Loading a pre-trained model
print("Loading Model")
new_model = load_model('tmp/model.ckpt')
print("Model Loaded")



#Evaluating it on Czech Names
score, acc = new_model.evaluate(vtrain_x, vtrain_y)
print("Accuracy we're getting on Czech Names is...", acc * 100,"%")



names = sys.argv[1:]
for i in names:
    
    name=[i]

    X=[]
    trunc_name = [i[0:maxlen] for i in name]
    for i in trunc_name:
        tmp = [set_flag(char_index[j]) for j in str(i.lower())]
        for k in range(0,maxlen - len(str(i))):
            tmp.append(set_flag(char_index["END"]))
        X.append(tmp)
    pred=new_model.predict(np.asarray(X))
    pred = pred.round()[0]
    if pred[0] == 1.0:
        print( name[0], ".... is name of a male")
    else:
        print(name[0], ".... is name of a female")



