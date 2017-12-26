# -*-coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# load train data
train = pd.read_csv('F:/GitHub/kaggle/Digit Recognizer/train.csv')

# split feature and label
feature = train.drop('label', axis=1)
label = train[['label']]

# 设置一个阈值，为0-255之间，大于该值的设置为 255， 小于该值的设置为0
threshold = 100
feature[feature >= threshold] = 255
feature[feature < threshold] = 0

# split train and test data
train_feature, test_feature, train_label, test_label = train_test_split(feature, label, test_size=0.2, random_state=0)

# fit classifier model
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(train_feature, train_label)

# predict test_feature
predict_test = clf.predict(test_feature)

# evaluate score
print(metrics.classification_report(test_label, predict_test))

# predict test data,
test = pd.read_csv('F:/GitHub/kaggle/Digit Recognizer/test.csv')

# 设置一个阈值，为0-255之间，大于该值的设置为 255， 小于该值的设置为0
threshold = 100
test[test >= threshold] = 255
test[test < threshold] = 0

# you need save file as .csv like sample_submission.csv
predict = clf.predict(test)
predict = pd.DataFrame(predict)
predict.to_csv('predict.csv')