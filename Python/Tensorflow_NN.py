import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import random

voice =pd.read_csv('voice.csv')
voice = pd.DataFrame(voice)

for i in range(6):
    copy = voice
    copy['meanfreq']=copy['meanfreq']+random.gauss(.0001,.001) # add noice to mean freq var
    voice=voice.append(copy,ignore_index=True) # make voice df 2x as big

label = voice.pop("label")

# converts from dataframe to np array
voice=voice.values

# convert train labels to one hots
train_labels = pd.get_dummies(label)
# make np array
train_labels = train_labels.values

x_train,x_test,y_train,y_test = train_test_split(voice,train_labels,test_size=0.2)
# # so no we have predictors and y values, separated into test and train

x_train,x_test,y_train,y_test = np.array(x_train,dtype='float32'), np.array(x_test,dtype='float32'), np.array(y_train,dtype='float32'), np.array(y_test,dtype='float32')

# place holder for inputs. feed in later
x = tf.placeholder("float", [None, 20])

# # take 20 features to 1000 nodes in hidden layer. why? just cuz?
w1 = tf.Variable(tf.random_normal([20, 1000],stddev=.5,name='w1'))
# # add biases for each node
b1 = tf.Variable(tf.zeros([1000]))
# calculate activations 
hidden_output = tf.nn.softmax(tf.matmul(x, w1) + b1)

# bring from 10 nodes to 2 for my output
w2 = tf.Variable(tf.random_normal([1000, 2],stddev=.5,name='w2'))

b2 = tf.Variable(tf.zeros([2]))

# placeholder for correct values 
y_ = tf.placeholder("float", [None,2])
# #implement model. these are predicted ys
y = tf.nn.softmax(tf.matmul(hidden_output, w2) + b2)

loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y, y_, name='xentropy')))
opt = tf.train.AdamOptimizer(learning_rate=0.001)
train_step = opt.minimize(loss, var_list=[w1,b1,w2,b2])

tf_correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
tf_accuracy = tf.reduce_mean(tf.cast(tf_correct_prediction, "float"))

def get_mini_batch(x,y):
    rows=np.random.choice(x.shape[0], 100)
    return x[rows], y[rows]

# start session
sess = tf.Session()
summary_writer = tf.train.SummaryWriter('voices')
#summary_writer = tf.train.SummaryWriter('voices', sess.graph)

# # init all vars
init = tf.initialize_all_variables()
sess.run(init)

ntrials = 10000
for i in range(ntrials):
    # get mini batch
    a,b=get_mini_batch(x_train,y_train)
    # run train step, feeding arrays of 100 rows each time
    _, cost =sess.run([train_step,loss], feed_dict={x: a, y_: b})

result = sess.run(tf_accuracy, feed_dict={x: x_test, 
                                          y_: y_test})

print(50 * "-")
print("Train accuracy: {}".format(result))

result = sess.run(tf_accuracy, feed_dict={x:x_train,
                                          y_ : y_train})
print("Test accuracy: {}".format(result))

