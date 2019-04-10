import time
from sklearn import metrics
class make_Classifier():
    def __init__(self,classifier,name):
        self.classifier = classifier
        self.y_predict = 0
        self.running_time = 0
        self.name = name

    def fit(self,train_data,train_label):
        start = time.time()
        self.classifier.fit(train_data,train_label)
        end = time.time()
        self.running_time = end-start
        print("Running Time:\t",self.running_time)

    def predict(self,test_data):
        self.y_predict = self.classifier.predict(test_data)

        return self.y_predict

    def confusion_matrix(self,test_label):
        self.confusion_matrix = metrics.confusion_matrix(y_pred=self.y_predict,y_true=test_label)
        return self.confusion_matrix

    def accuracy(self,test_label):
        self.accuracy = metrics.accuracy_score(y_pred=self.y_predict, y_true=test_label)
        return self.accuracy

