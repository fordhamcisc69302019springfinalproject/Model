from sklearn import svm
import numpy as np
import pandas as pd

from sklearn import tree
import Simple_preprocess
import Classifier


train_data, train_label, test_data, test_label = Simple_preprocess.preprocess()
train_data = train_data[0:5]
train_label = train_label[0:5]
print(train_data)






if __name__ == "__main__":
    SVM = svm.SVC(gamma = 0.1)
    classifier = Classifier.make_Classifier(SVM,"SVM")
    classifier.fit(test_data,test_label)
    predict = classifier.predict(test_data)
    print(classifier.accuracy(test_label))
