from sklearn import svm
from sklearn.cross_validation import train_test_split

import numpy as np
import pandas as pd

all_data = pd.read_csv('voice.csv')
label = all_data.pop('label')

all_data = all_data.values

label.replace(['male','female'], [1, 0], inplace = True)
label = label.values

train_data, test_data, train_labels, test_labels = train_test_split(all_data, label, test_size = 0.2)

train_data, test_data, train_labels, test_labels = np.array(train_data, dtype = 'float32'), np.array(test_data, dtype = 'float32'),np.array(train_labels, dtype = 'float32'),np.array(test_labels, dtype = 'float32')


clf = svm.LinearSVC()
clf.fit(train_data, train_labels)

#Training Part
predict_labels_train = clf.predict(train_data)
count_0 = 0
count_1 = 0
for i in range(0,2534):
    if predict_labels_train[i] == 0:
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
for i in range(0,2534):
    if train_labels[i] == predict_labels_train[i]:
       correct_predic += 1
    else:
       wrong_predic += 1
    
print("Number of correct prediction", correct_predic)
print("Number of wrong prediction", wrong_predic)
print("---------------Training-------------------")
print("Training Correct rate is: ")
print(correct_predic/(wrong_predic+correct_predic))

predict_labels_test = clf.predict(test_data)

count_0 = 0
count_1 = 0
for i in range(0,634):
    if predict_labels_test[i] == 0:
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
    if test_labels[i] == predict_labels_test[i]:
       correct_predic += 1
    else:
       wrong_predic += 1
    
print("Number of correct prediction", correct_predic)
print("Number of wrong prediction", wrong_predic)
print("---------------Testing-------------------")
print("Correct rate is: ")
print(correct_predic/(wrong_predic + correct_predic))


