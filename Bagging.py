from sklearn.ensemble import BaggingClassifier

from sklearn import tree
import Simple_preprocess
import Classifier

train_data, train_label, test_data, test_label = Simple_preprocess.preprocess()



if __name__ == "__main__":
    bagging = BaggingClassifier(tree.DecisionTreeClassifier(),max_samples=0.5,max_features=0.5)
    classifier = Classifier.make_Classifier(bagging,"bagging")
    classifier.fit(train_data,train_label)
    predict = classifier.predict(test_data)
    print(classifier.accuracy(test_label))
