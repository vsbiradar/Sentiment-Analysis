import cPickle
import numpy as np
import json
import tensorflow as tf
import gensim
import re
import os
import nltk
import string
import time
from nltk import word_tokenize
from nltk.corpus import stopwords
import cPickle as pickle

f = open('snli_1.0/snli_1.0_train.jsonl','r')
starttime = time.time()
labels = []
train1 = []
train2 = []
labels_dict = {'neutral':0,'entailment':1,'contradiction':2}
for line in f:
    if json.loads(line)['gold_label'] not in labels_dict:
        labels.append('neutral')
    else:
        labels.append(json.loads(line)['gold_label'])
    train1.append(json.loads(line)['sentence1'])
    train2.append(json.loads(line)['sentence2'])

train1list = []
train2list = []
stopwords = set(nltk.corpus.stopwords.words('english'))
print "done loading files"
for line in train1:
    sent = str(line).lower()
    sent.translate(None, string.punctuation)
    linestr = [word for word in sent.split() if word not in stopwords]
    #linestr = [word for word in linestr if word.isalpha()]
    train1list.append(linestr)
for line in train2:
    sent = str(line).lower()
    sent.translate(None, string.punctuation)
    linestr = [word for word in sent.split() if word not in stopwords]
    #linestr = [word for word in linestr if word.isalpha()]
    train2list.append(linestr)
    
max_len = 0
for line in train1list+train2list:
    size = len(line)
    if(size>max_len):
        max_len = size
        


# testing file processing ----------------------------------
f = open('snli_1.0/snli_1.0_test.jsonl','r')
testlabels = []
test1 = []
test2 = []
for line in f:
    if json.loads(line)['gold_label'] not in labels_dict:
        testlabels.append('neutral')
    else:
        testlabels.append(json.loads(line)['gold_label'])
    test1.append(json.loads(line)['sentence1'])
    
    test2.append(json.loads(line)['sentence2'])
f.close()

test1list = []
test2list = []

for line in test1:
    sent = str(line).lower()
    sent.translate(None, string.punctuation)
    linestr = [word for word in sent.split() if word not in stopwords]
    #linestr = [word for word in linestr if word.isalpha()]
    test1list.append(linestr)
for line in test2:
    sent = str(line).lower()
    sent.translate(None, string.punctuation)    
    linestr = [word for word in sent.split() if word not in stopwords]
    #linestr = [word for word in linestr if word.isalpha()]
    test2list.append(linestr)
#---------------------------------------------------------------------

train = train1list+train2list+test1list+test2list
#vocabulary = {}
#id2word = {}
#-----------------------------------------------------------------------------

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

vocabulary = model.vocab

train1 = []

for line in train1list:
    dmtr = np.zeros((max_len,300))
    for i,word in enumerate(line):
        if word in vocabulary:
            dmtr[i] = model.wv[word]
        else:
            dmtr[i] = np.random.random((300))
    train1.append(dmtr)
    
train1 = np.array(train1)

print "train1 shape:",train1.shape

train2 = []

for line in train2list:
    dmtr = np.zeros((max_len,300))
    for i,word in enumerate(line):
        if word in vocabulary:
            dmtr[i] = model.wv[word]
        else:
            dmtr[i] = np.random.random((300))
    train2.append(dmtr)
    
train2 = np.array(train2)

print " train2 shape:",train2.shape

# index=0
# for line in train:
#     for word in line:
#         if word not in vocabulary:
            
#             vocabulary[word]=index
#             id2word[index] = word
#             index = index + 1
            
# print len(vocabulary)
dimensionv = 100



padd = len(vocabulary)
train1ids = []
train2ids = []
for line in train1list:
    id1 = [vocabulary[word] for word in line]
    for i in range(0,maxlen-len(line)):
        id1.append(padd)
    train1ids.append(id1)
train1ids = np.array(train1ids)

for line in train2list:
    id1 = [vocabulary[word] for word in line]
    for i in range(0,maxlen-len(line)):
        id1.append(padd)
    train2ids.append(id1)
train2ids = np.array(train2ids)

print "trainid shape:",train1ids.shape
#trainids = np.concatenate((train1ids,train2ids),1)


#getting lables 
labels_dict = {'neutral':0,'entailment':1,'contradiction':2}
trainlabels = []
for word in labels:
    ab = [0,0,0]
    ab[labels_dict[word]] = 1
    trainlabels.append(ab)

trainlabels = np.array(trainlabels)

# testing data and labels ---------------------------


test1ids = []
test2ids = []
for line in test1list:
    id1 = [vocabulary[word] for word in line]
    for i in range(0,maxlen-len(line)):
        id1.append(padd)
    test1ids.append(id1)
test1ids = np.array(test1ids)

for line in test2list:
    id1 = [vocabulary[word] for word in line]
    for i in range(0,maxlen-len(line)):
        id1.append(padd)
    test2ids.append(id1)
test2ids = np.array(test2ids)

testlabelsids = []
for word in testlabels:
    testlabelsids.append(labels_dict[word])

testlabelsids = np.array(testlabelsids)

print "preocessed test file"

# model --------------------------------------------
W= tf.Variable(W)
ids1 = tf.placeholder(tf.int32, [None,maxlen])
keep_factor = tf.placeholder(tf.float64)
ids1 = tf.reshape(ids1, [-1,maxlen])
ids2 = tf.placeholder(tf.int32, [None,maxlen])
ids2 = tf.reshape(ids2, [-1,maxlen])
embeddings1 = tf.nn.embedding_lookup(W, ids1)
vector1 = tf.divide(tf.reduce_sum(embeddings1, 1),maxlen)
vector1 = tf.reshape(vector1, [-1,dimensionv])
#output1 = tf.layers.dense(inputs=vector1, units=10,activation=tf.nn.relu)

embeddings2 = tf.nn.embedding_lookup(W, ids2)
vector2 = tf.divide(tf.reduce_sum(embeddings2, 1),maxlen)
vector2 = tf.reshape(vector2, [-1,dimensionv])
#output2 = tf.layers.dense(inputs=vector2, units=10, activation=tf.nn.relu)

output = tf.concat((vector1,vector2),1)
#hid1 = tf.layers.dense(inputs=output, units=700, activation=tf.nn.relu)
#hid1drop = tf.nn.dropout(hid1,keep_factor)

#hid2 = tf.layers.dense(inputs = hid1drop, units=700, activation=tf.nn.relu)

out = tf.layers.dense(inputs=output, units=3)
#testing ------------------------
predict = tf.nn.softmax(out,1)
predict = tf.reshape(predict, [-1,3])
result = tf.argmax(predict,1)
#--------------------------------
y = tf.placeholder(tf.float32, [None,3])
loss = tf.losses.softmax_cross_entropy(y,out)
train_step = tf.train.AdamOptimizer(0.05).minimize(loss)

os.environ["CUDA_VISIBLE_DEVICES"]="2"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
#sess = tf.InteractiveSession()
init_op = tf.global_variables_initializer()
sess.run(init_op)

# training ------------------------------------------------------

for epoch in range(0,10):
    for i in range(0,550016,64):
        batch1 = train1ids[i:i+64]
        batch2 = train2ids[i:i+64]
        
        batchlabels = trainlabels[i:i+64]
        sess.run(train_step, {ids1:batch1,ids2:batch2, y:batchlabels, keep_factor:0.8})
    print epoch




    correct = 0
    for i in range(0,10000,500):
        batch1 = test1ids[i:i+500]
        batch2 = test2ids[i:i+500]
        batchtestlabel = testlabelsids[i:i+500]
        predictions = sess.run(result, {ids1:batch1, ids2:batch2, keep_factor:1})
        batchtestlabel = np.array(batchtestlabel)
        predictions = np.array(predictions)
        correct = correct + sum(predictions==batchtestlabel)
    print "accuracy:",correct

endtime = time.time()
print "time required is :",endtime- starttime
# testing ------------------------------------------------

'''
print "accuracy",correct
f = open('accuracy2','w+')
f.write("%s\n"%correct)
f.close()
'''