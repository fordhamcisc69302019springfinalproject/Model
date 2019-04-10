from sklearn import neighbors

from sklearn import tree
import Simple_preprocess
import Classifier

train_data, train_label, test_data, test_label = Simple_preprocess.preprocess()

if __name__ == "__main__":
    KNN = neighbors.KNeighborsClassifier(n_neighbors = 3)
    classifier = Classifier.make_Classifier(KNN,"KNN")
    classifier.fit(train_data,train_label)
    predict = classifier.predict(test_data)
    print(classifier.accuracy(test_label))
