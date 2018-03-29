#Accuracy with 10 epochs is 84.41

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

#place1 = tf.reshape(place,[-1,max_len,300,1])
out1 = tf.layers.conv1d(place,filters=100,kernel_size=3,padding='valid',activation=tf.nn.relu)
out2 = tf.layers.conv1d(place,filters=100,kernel_size=4,padding='valid',activation=tf.nn.relu)
out3 = tf.layers.conv1d(place,filters=100,kernel_size=5,padding='valid',activation=tf.nn.relu)


out4 = tf.layers.max_pooling1d(out1,pool_size=498,strides=1)
out5 = tf.layers.max_pooling1d(out2,pool_size=497,strides=1)
out6 = tf.layers.max_pooling1d(out3,pool_size=496,strides=1)
print out4.get_shape()

out4 = tf.reshape(out4,[-1,100])
out5 = tf.reshape(out5,[-1,100])
out6 = tf.reshape(out6,[-1,100])

out7 = tf.concat((out4,out5,out6),1)
vector = tf.reshape(out7, [-1,300])
output = tf.layers.dense(inputs=vector, units=100,activation=tf.nn.relu)

output1 = tf.layers.dense(inputs=output,units=2)


forprediction = tf.reshape(output1, [-1,2])
predict = tf.argmax(forprediction,1)
y = tf.placeholder(tf.float32, [None,2])
loss = tf.losses.softmax_cross_entropy(y,output1)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

init_op = tf.global_variables_initializer()
sess.run(init_op)


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