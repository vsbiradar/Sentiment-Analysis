import tensorflow as tf
import gensim
import numpy
import numpy as np
import cPickle as pickle
import os
import logging
from nltk import word_tokenize
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
filepos = open('rt-polaritydata/rt-polarity.pos','r')
fileneg = open('rt-polaritydata/rt-polarity.neg','r')

postrain = []
negtrain = []
posn = 0
for line in filepos:
    posn= posn +1;

    linestr = [word for word in str(line).lower().split() if word not in stop]
    postrain.append(linestr)

negn = 0
for line in fileneg:
    negn = negn + 1;
    linestr = [word for word in str(line).lower().split() if word not in stop]
    negtrain.append(linestr)

filepos.close()
fileneg.close()
print "possitive examples:",posn
print "negative examples :",negn

train = postrain+negtrain
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
print "model loaded"
vocabulary = model.vocab
lengths = []
max_len = 0
for line in train:
	lengths.append(len(line))
max_len = max(lengths)


positivetrain = []

for line in postrain:
    dmtr = np.zeros((max_len,300))
    for i,word in enumerate(line):
        if word in vocabulary:
            dmtr[i] = model.wv[word]
        else:
            dmtr[i] = np.random.random((300))
    positivetrain.append(dmtr)
    
positivetrain = np.array(positivetrain)

print "positive train shape:",positivetrain.shape

negativetrain = []

for line in negtrain:
    dmtr = np.zeros((max_len,300))
    for i,word in enumerate(line):
        if word in vocabulary:
            dmtr[i] = model.wv[word]
        else:
            dmtr[i] = np.random.random((300))
    negativetrain.append(dmtr)
    
negativetrain = np.array(negativetrain)

print "negative train shape:",negativetrain.shape


import tensorflow

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
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


for i in range(0,5):
    trainfold = []
    testfold = []
    trainlabel = []
    testlabel = []
    trainlabel = [10,11]
    for j in range(0,5):
        if i==j:
            testfold.append(positivetrain[i*1066:(i+1)*1066])
            testlabel = [0,1]*1066
            
            testfold.append(negativetrain[i*1066:(i+1)*1066])
            testlabel = numpy.concatenate((numpy.array(testlabel).reshape(-1,2), numpy.array([1,0]*1066).reshape(-1,2)))
        else:
            #print i,j
            trainfold.append(positivetrain[j*1066:(j+1)*1066])
            trainlabel = numpy.concatenate((numpy.array(trainlabel).reshape(-1,2),numpy.array([0,1]*1066).reshape(-1,2)))
            trainfold.append(negativetrain[j*1066:(j+1)*1066])
            trainlabel = numpy.concatenate((numpy.array(trainlabel).reshape(-1,2),numpy.array([1,0]*1066).reshape(-1,2)))
    trainfold = np.vstack(trainfold)
    trainlabel = numpy.delete(trainlabel,0,0) #shape is 8528,42,300
    
    testfold = np.vstack(testfold) #shape is 2132,42,300
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    
    for epoch in range(10):
        for k in range(0,8528,64):
            max_sample = min(k+64,8528)
            batch = trainfold[k:max_sample]
            yval = trainlabel[k:max_sample].reshape(-1,2)
            sess.run(train_step, {place:batch, y:yval})
        print epoch
    correct = 0
    label = sess.run(predict, {place:testfold})
    
    label1 = label[:1066]
    label2 = label[1066:]
    correct = sum(label1)+1066-sum(label2)
    print i," acc: ",correct/2132.0 *100
    