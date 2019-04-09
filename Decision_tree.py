import Simple_preprocess
from sklearn import tree
from sklearn import metrics
import Predictor
if __name__ == "__main__":

    clf = tree.DecisionTreeClassifier()

    y_pred = Predictor.predict(clf)
    print(y_pred)
