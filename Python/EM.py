import pandas as pd
import numpy as np
import math

from sklearn.cross_validation import train_test_split



all_data = pd.read_csv("voice.csv")
label = all_data.pop('label')

all_data = all_data.values

label.replace(['male','female'], [1, 0], inplace = True)
label = label.values

train_data, test_data, train_labels, test_labels = train_test_split(all_data, label, test_size = 0.2)

train_data, test_data, train_labels, test_labels = np.array(train_data, dtype = 'float32'), np.array(test_data, dtype = 'float32'),np.array(train_labels, dtype = 'float32'),np.array(test_labels, dtype = 'float32')


label_1 = np.ones(len(train_labels)/2)
label_2 = np.zeros(len(train_labels)/2)
labels = np.concatenate((label_1,label_2), axis = 0)

iteration = 1000

for iter in range(iteration):
    D1 = np.zeros((1,20))
    D2 = np.zeros((1,20))
    for i in range(0,len(train_labels)):
        if labels[i]==1:
            D1 = np.vstack((D1, train_data[i,:].reshape(1,20)))
        else:
            D2 = np.vstack((D2, train_data[i,:].reshape(1,20)))

    D1 = np.delete(D1,0,0)
    D2 = np.delete(D2,0,0)
    
    sd1 = len(D1)
    sd2 = len(D2)
    mu1 = np.mean(D1, axis = 0)
    mu2 = np.mean(D2, axis = 0)

    dd1 = D1 - ((mu1) * np.ones((sd1,1)))
    dd2 = D2 - ((mu2) * np.ones((sd2,1)))
    
    cov1 = (np.dot(dd1.T, dd1))/sd1 
    cov2 = (np.dot(dd2.T, dd2))/sd2 
    
    
    icov1 = np.linalg.pinv(cov1)
    icov2 = np.linalg.pinv(cov2)


    (sign1,logdet1) = np.linalg.slogdet(cov1)
    (sign2,logdet2) = np.linalg.slogdet(cov2)
    #print(sign1*np.exp(logdet1))
    det1 = sign1 * np.exp(logdet1) + 0.00001
    det2 = sign2 * np.exp(logdet2) + 0.00001
 
    labels = np.zeros((len(train_labels),1))

    for i in range(0, len(train_labels)):
        train_mu1 = (train_data[i,:] - mu1).reshape(1,20)   
        train_mu2 = (train_data[i,:] - mu2).reshape(1,20)
    
        likelihood1 = (1/(2*math.pi*math.sqrt(det1)))*np.dot(np.dot(-0.5*train_mu1, icov1),train_mu1.reshape(20,1))
        likelihood2 = (1/(2*math.pi*math.sqrt(det2)))*np.dot(np.dot(-0.5*train_mu2, icov2),train_mu2.reshape(20,1))
        #print("Likelihood1:", likelihood1, "Likelihood2: ", likelihood2)
        if (likelihood1 * sd1) > (likelihood2 * sd2):
           labels[i] = 1
        else:
           labels[i] = 0


#Training Correct Rate
count_0 = 0
count_1 = 0
for i in range(0, 2534):
    if labels[i] == 0:
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
    if labels[i] == train_labels[i]:
       correct_predic += 1
    else:
       wrong_predic += 1
    
print("Number of correct prediction", correct_predic)
print("Number of wrong prediction", wrong_predic)
print("---------------Trainging-------------------")

print("Training Correct rate is: ")
print(correct_predic/(wrong_predic+correct_predic))


#Testing part

for i in range(0, len(test_labels)):
    train_mu1 = (test_data[i,:] - mu1).reshape(1,20)   
    train_mu2 = (test_data[i,:] - mu2).reshape(1,20)
    
    likelihood1 = (1/(2*math.pi*math.sqrt(det1)))*np.dot(np.dot(-0.5*train_mu1, icov1),train_mu1.reshape(20,1))
    likelihood2 = (1/(2*math.pi*math.sqrt(det2)))*np.dot(np.dot(-0.5*train_mu2, icov2),train_mu2.reshape(20,1))
    #print("Likelihood1:", likelihood1, "Likelihood2: ", likelihood2)
    if (likelihood1 * sd1) > (likelihood2 * sd2):
        labels[i] = 1
    else:
        labels[i] = 0

count_0 = 0
count_1 = 0
for i in range(0, 634):
    if labels[i] == 0:
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
    if test_labels[i] == labels[i]:
       correct_predic += 1
    else:
       wrong_predic += 1
    
print("Number of correct prediction", correct_predic)
print("Number of wrong prediction", wrong_predic)
print("---------------Testing-------------------")
print("Testing Correct rate is: ")
print(correct_predic/(wrong_predic + correct_predic))
