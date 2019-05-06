from CISC6930_preprocessing import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
from Simple_preprocess import preprocess
import numpy as np
from sklearn.metrics import accuracy_score
from  sklearn.metrics import *
scoring = make_scorer(accuracy_score)
def logistic_regression(x_train,y_train,penalty,featureselection = False):
    if penalty == "l1":
        lr = LogisticRegression(penalty="l1", solver='liblinear', max_iter=100,C = 2)
    else:
        lr = LogisticRegression(penalty="l2", solver='sag', max_iter=100,C = 4)

    if featureselection:
        lr = RFECV(lr, step=1, cv=10)
    lr.fit(x_train,y_train)
    if featureselection:
        accuracy = lr.score(x_train, y_train)
        print("Feature index:",lr.get_support(True))
    else:
        accuracy = cross_val_score(lr, x_train, y_train,scoring=scoring).mean()

    print("Accuracy :", accuracy)



if __name__ == "__main__":
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
               'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
               'label']
    data = load_data("data/census-income.data.csv",columns)
    data = get_dummy(data,False)
    x_train = data.iloc[:,:-1]
    y_train = data.iloc[:,-1]
    data = load_data("data/census-income.data.csv", columns)
    data = get_dummy(data, False)
    x_test = data.iloc[:,:-1]
    y_test = data.iloc[:,-1]
    lr = LogisticRegression(penalty="l2", solver='sag', max_iter=100, C=4)
    lr.fit(x_train, y_train)
    #accuarcy = cross_val_score(KNN, x_train, y_train).mean()
    #print("k = %d,Accuracy: %f"%(1,accuarcy))
    predict = lr.predict(x_test)
    print(accuracy_score(predict,y_test))

    """
    #L2 penalty
    #lr = LogisticRegression(penalty = "l2",solver='sag',max_iter = 100)
    #L1 penalty
    lr = LogisticRegression(penalty="l1", solver='liblinear', max_iter=100)
    #lr = RFECV(lr, step=1, cv=10)
    #lr = lr.fit(x_train, y_train)
    #print("Accuracy: ",lr.score(x_train, y_train))
    #without feature selection
    lr = lr.fit(x_train,y_train)
    print("Accuracy: ",cross_val_score(lr, x_train, y_train).mean())
    """
    #logistic_regression(x_train,y_train,"l2")



