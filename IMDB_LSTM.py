
# Accuracy is 84.98 with 10 epochs
import numpy as np
import os
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
import cPickle as pickle
stop = set(stopwords.words('english'))
path = 'aclImdb/train/pos/'
f = open('imdbtrainpos.txt','w+')
count=0
reviews = []
for filename in os.listdir(path):
    f2 = open(path+filename)
    sent = f2.read()
    sent.translate(None, string.punctuation)
    sentence = [word for word in sent.split() if word not in stop and len(word)>1]
    reviews.append(sentence)
    count=count+1
    f2.close()
pickle.dump(reviews,f)
f.close()
f = open('imdbtrainneg.txt','w+')
reviews = []
path = 'aclImdb/train/neg/'
for filename in os.listdir(path):
    f2 = open(path+filename)
    sent = f2.read()
    sent.translate(None, string.punctuation)
    sentence = [word for word in sent.split() if word not in stop and len(word)>1]
    reviews.append(sentence)
    count=count+1
    f2.close()
    
print count
pickle.dump(reviews,f)
f.close()

# testing data

path = 'aclImdb/test/pos/'
f = open('imdbtest.txt','w+')
count=0
reviews = []
for filename in os.listdir(path):
    f2 = open(path+filename)
    sent = f2.read()
    sent.translate(None, string.punctuation)
    sentence = [word for word in sent.split() if word not in stop and len(word)>1]
    reviews.append(sentence)
    count=count+1
    f2.close()
#f = open('imdbtrainneg.txt','w+')
#reviews = []
path = 'aclImdb/test/neg/'
for filename in os.listdir(path):
    f2 = open(path+filename)
    sent = f2.read()
    sent.translate(None, string.punctuation)
    sentence = [word for word in sent.split() if word not in stop and len(word)>1]
    reviews.append(sentence)
    count=count+1
    f2.close()
    
print count
pickle.dump(reviews,f)
f.close()


imdbtrain = pickle.load(open('imdbtrain.txt'))
imdbtrain=np.array(imdbtrain)
imdbtrain.shape

imdbtest = pickle.load(open('imdbtest.txt'))
imdbtest = np.array(imdbtest)

print imdbtest.shape

#train = postrain+negtrain
import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
print "model loaded"
vocabulary = model.vocabulary

max_len = 500


train = []

for line in imdbtrain:
    dmtr = np.zeros((max_len,300))
    for i,word in enumerate(line):
        if i>=500:
            break;
        if word in vocabulary:
            dmtr[i] = model.wv[word]
        else:
            dmtr[i] = np.random.random((300))
    train.append(dmtr)
    
train = np.array(train)

print "train shape:",train.shape


test = []

for line in imdbtest:
    dmtr = np.zeros((max_len,300))
    for i,word in enumerate(line):
        if i>=500:
            break;
        if word in vocabulary:
            dmtr[i] = model.wv[word]
        else:
            dmtr[i] = np.random.random((300))
    test.append(dmtr)
    
test = np.array(test)

print "test shape:",test.shape

import os
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
import tensorflow as tf
place = tf.placeholder(tf.float32,[None,max_len,300])

num_units = 50
num_layers = 1
dropout = 0.0

def length(sequence):
  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32)
  return length

def last_relevant(output, length):
  batch_size = tf.shape(output)[0]
  max_length = tf.shape(output)[1]
  out_size = int(output.get_shape()[2])
  index = tf.range(0, batch_size) * max_length + (length - 1)
  flat = tf.reshape(output, [-1, out_size])
  relevant = tf.gather(flat, index)
  return relevant



cells = []
for _ in range(num_layers):
  cell = tf.contrib.rnn.LSTMCell(num_units)  # Or LSTMCell(num_units)
  cell = tf.contrib.rnn.DropoutWrapper(
      cell, output_keep_prob=1.0 - dropout)
  cells.append(cell)
cell = tf.contrib.rnn.MultiRNNCell(cells)


out1, state = tf.nn.dynamic_rnn(cell, place, dtype=tf.float32,sequence_length=length(place))
last = last_relevant(out1,length(place))
vector = tf.reshape(last, [-1,50])
output = tf.layers.dense(inputs=vector, units=2)


forprediction = tf.reshape(output, [-1,2])
predict = tf.argmax(forprediction,1)
y = tf.placeholder(tf.float32, [None,2])
loss = tf.losses.softmax_cross_entropy(y,output)
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

import numpy
trainlabels = [0,1]*12500
trainlabel = numpy.concatenate((numpy.array(trainlabels).reshape(-1,2), numpy.array([1,0]*12500).reshape(-1,2)))
print trainlabel.shape

a = np.random.permutation(range(25000))
train = train[a]
trainlabel = trainlabel[a]
# training model

for epoch in range(10):
    for k in range(0,25000,256):
        max_sample = min(k+256,25000)
        batch = train[k:max_sample]
        yval = trainlabel[k:max_sample].reshape(-1,2)
        sess.run(train_step, {place:batch, y:yval})
        print epoch,k


# testing model
sum1 = 0
for k in range(0,12500,256):
    max_sample = min(k+256,12500)
    batch = test[k:max_sample]
    label1 = sess.run(predict, {place:batch})
    sum1 += sum(label1)
    

for k in range(12500,25000,256):
    max_sample = min(k+256,25000)
    batch = test[k:max_sample]
    label1 = sess.run(predict, {place:batch})
    sum1 -= sum(label1)
    
sum1 += 12500
print "accuracy : ", sum1/25000.0*100