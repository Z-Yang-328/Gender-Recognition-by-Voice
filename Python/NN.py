import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split



#Sigmoid function

def nonlin(x, deriv = False):
    if (deriv == True):
       return x * (1 - x)
    else:
       return 1/(1 + np.exp(-x))

all_data = pd.read_csv('voice.csv')
label = all_data.pop('label')

all_data = all_data.values

label.replace(['male','female'], [1, 0], inplace = True)
label = label.values

train_data, test_data, train_labels, test_labels = train_test_split(all_data, label, test_size = 0.2)

train_data, test_data, train_labels, test_labels = np.array(train_data, dtype = 'float32'), np.array(test_data, dtype = 'float32'),np.array(train_labels, dtype = 'float32'),np.array(test_labels, dtype = 'float32')

#To ensure random number of every loop is the same
np.random.seed(1245)
#Initialize weights
syn0 = 2 * np.random.random((20,1)) - 1
l0 = train_data
for iter in range(100000):
    #Input(l0) with weights(syn0) 
    #Combined by sigmoid function
    l1 = nonlin(np.dot(l0,syn0))
    l1_error = train_labels.T - l1.T
    syn0 += np.dot(l0.T, l1_error.T)

#Training Correct Rate
count_0 = 0
count_1 = 0
for i in l1:
    if 0 in i:
       count_0 += 1
    else:
       count_1 += 1
print("---------------Training-------------------")
print("Result of Training Part")
print("predicted number of female", count_0)
print("predicted number of male", count_1)
print("total number of train data:",count_0 + count_1)
print("---------------Training-------------------")

correct_predic = 0
wrong_predic = 0
#Do not forget to change range when changing training-test ratio
for i in range(0,2534):
    if train_labels[i] == l1[i]:
       correct_predic += 1
    else:
       wrong_predic += 1
    
print("Number of correct prediction", correct_predic)
print("Number of wrong prediction", wrong_predic)
print("---------------Trainging-------------------")

print("Training Correct rate is: ")
print(correct_predic/(wrong_predic+correct_predic))

#Testing part

l0 = test_data
   
l1 = nonlin(np.dot(l0,syn0))

count_0 = 0
count_1 = 0
for i in l1:
    if 0 in i:
       count_0 += 1
    else:
       count_1 += 1
print("---------------Testing-------------------")
print("Result of Testing Part")
print("predicted number of female", count_0)
print("predicted number of male", count_1)
print("total number of train data:",count_0 + count_1)
print("---------------Testing-------------------")

correct_predic = 0
wrong_predic = 0
for i in range(0,634):
    if test_labels[i] == l1[i]:
       correct_predic += 1
    else:
       wrong_predic += 1
    
print("Number of correct prediction", correct_predic)
print("Number of wrong prediction", wrong_predic)
print("---------------Testing-------------------")
print("Testing Correct rate is: ")
print(correct_predic/(wrong_predic + correct_predic))
