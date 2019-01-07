

from __future__ import print_function

import pandas as pd
from sklearn.metrics import r2_score
import numpy as np
import sys
import os
import tensorflow as tf
from sklearn.metrics import accuracy_score#, log_loss, f1_score
# tf.enable_eager_execution()

from tensorflow.nn.rnn_cell import LSTMStateTuple
from tensorflow.keras.layers import TimeDistributed
from tensorflow.contrib.rnn import OutputProjectionWrapper

import torch

from torch.utils.data import TensorDataset, DataLoader
import pickle



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


# ### Making BatchLoader

#Creating Tensor Datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

#Now creating Data Loaders
batch = 600

train_loader = DataLoader(train_data, shuffle = True, batch_size = batch)
test_loader = DataLoader(test_data, shuffle = True, batch_size = batch)

dataiter = iter(train_loader)

sample_x, sample_y = dataiter.next()

print("Sample input size of one instance from iter is: ", sample_x.size()) #Batch size, Sequence Length, Feature Dimension
print("Sample target size of that same one instance from iter is", sample_y.size())


# #### build model in Tensorflow ( a stacked LSTM model with many-to-one arch ) here 30 sequence and 2 output each for one category(m/f)



''' Parameters '''
tf.reset_default_graph()
no_units=524 

features=train_x.shape[-1] #30
# new_phoneme=new_phoneme.reshape((-1,1,features))


# ### Making Placeholders


train = tf.placeholder(dtype = tf.float32, shape = [None, None , features], name = 'input')
target = tf.placeholder(dtype = tf.int32, shape = [None, 2], name = 'label')



LSTM_fw = [tf.nn.rnn_cell.LSTMCell(num_units=no_units, initializer=tf.keras.initializers.glorot_normal(), state_is_tuple=True)]
LSTM_bw = [tf.nn.rnn_cell.LSTMCell(num_units=no_units, initializer=tf.keras.initializers.glorot_normal(), state_is_tuple=True)]

for i in range(1,3):

    LSTM_fw.append(tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(num_units=no_units, initializer=tf.keras.initializers.glorot_normal(), state_is_tuple=True)]))
    LSTM_bw.append(tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(num_units=no_units, initializer=tf.keras.initializers.glorot_normal(), state_is_tuple=True)]))





LSTM_outputs, LSTM_fw_state, LSTM_bw_state = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                                                                           cells_fw = LSTM_fw,
                                                                           cells_bw = LSTM_bw,
                                                                           inputs = train,
                                                                           dtype = tf.float32
                                                                          )



# LSTM_outputs = tf.concat((LSTM_fw_output, LSTM_bw_output), 2)

LSTM_state_c = tf.concat((LSTM_fw_state[-1][0].c, LSTM_bw_state[-1][0].c), 1)

LSTM_state_h = tf.concat((LSTM_fw_state[-1][0].h, LSTM_bw_state[-1][0].h), 1)

LSTM_final_state = LSTMStateTuple(
    c=LSTM_state_c,
    h=LSTM_state_h
)

output=tf.layers.Dense(2)(LSTM_state_h)

#Defining Loss function
losss= tf.losses.softmax_cross_entropy(target, output)
trainop =  tf.train.AdamOptimizer(learning_rate=0.001).minimize(losss)



# losx=[]
saver = tf.train.Saver()
# # with tf.device('/gp):
# # with tf.device('/gpu:0'):
# with tf.Session() as sess:
    
#     sess.run(tf.global_variables_initializer())
    
#     if(tf.train.checkpoint_exists("tmp/model.ckpt")):
#         saver.restore(sess, "tmp/model.ckpt")
#         print("Model Restored")

#     for i in range(15):
#         for batch_j, target_j in train_loader:

#             loss,_=sess.run([losss,trainop],feed_dict={train: batch_j,
#                                                        target : target_j})
#             losx.append([loss])
#         print("Current epoch going is...", i," and current loss is...",loss)
#     save_path = saver.save(sess, "tmp/model_tensorflow.ckpt")
#     print("Model saved in path: %s" % save_path)

    




#Checking accuracy
accc = []
# with tf.device('/gpu:0'):
with tf.Session() as sess:

#         sess.run(tf.global_variables_initializer())
    if(tf.train.checkpoint_exists("tmp/model_tensorflow.ckpt")):
        saver.restore(sess, "tmp/model_tensorflow.ckpt")
        print("Model Restored")
    for batch_j, target_j in test_loader:
        out = sess.run([output], feed_dict={train: batch_j})
        out = np.squeeze(out)
#             out = np.round(out)
#         print("This is",sum([np.argmax(out[i])==np.argmax(target_j[i].numpy()) for i in range(target_j.shape[0])])/1000
        metric = accuracy_score(np.argmax(target_j.numpy(), axis= 1),  np.argmax(np.squeeze(out), axis = 1))
        print("real score",metric )

#         acc = tf.keras.backend.categorical_crossentropy(new_ta, out)
        something = metric
        something = np.squeeze(something)
        accc.append(np.mean(something))
    print("Final mean is: ",np.mean(accc),'\n\n\n')




with tf.Session() as sess:
    if(tf.train.checkpoint_exists("tmp/model_tensorflow.ckpt")):
        saver.restore(sess, "tmp/model_tensorflow.ckpt")
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

#         sess.run(tf.global_variables_initializer())
    
            out = sess.run([output], feed_dict={train: np.asarray(X)})
            out = np.squeeze(out)
                
        # out = np.argmax(out, axis=1)
            out = np.array(out)
            out = np.argmax(out)
            # for i in range(out.shape[0]):
            if out == 0:
                print(name[0], "....it's Male")
            else:
                print(name[0],"....It's Female")



