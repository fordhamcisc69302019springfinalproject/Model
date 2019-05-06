from sklearn import svm
from CISC6930_preprocessing import *
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import accuracy_score


def linear_kernel(c,x_train,y_train):
    SVM = svm.SVC(C = c,kernel="linear")
    #SVM.fit(x_train,y_train)
    #return SVM.score(x_train,y_train)
    return cross_val_score(SVM,x_train,y_train).mean()

def polyps_kernel(c,degree,x_train,y_train):
    SVM = svm.SVC(C = c,degree = degree,kernel="poly")
    return cross_val_score(SVM,x_train,y_train).mean()

def rbf_kernel(c,gamma,x_train,y_train):
    SVM = svm.SVC(C = c,kernel="rbf")
    return cross_val_score(SVM, x_train, y_train).mean()

def sigmoid_kernel(c,gamma,x_train,y_train):
    SVM = svm.SVC(C = c,kernel="sigmoid")
    return cross_val_score(SVM, x_train, y_train).mean()

def precomputed_kernel(c,x_train,y_train):
    SVM = svm.SVC(C = c,kernel="precomputed")
    return cross_val_score(SVM,x_train,y_train).mean()


if __name__ == "__main__":
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
               'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
               'label']
    data = load_data("data/census-income.data.csv",columns)
    data = get_dummy(data,True)
    x_train = data.iloc[:,:-1]
    y_train = data.iloc[:,-1]
    data = load_data("data/census-income.data.csv", columns)
    data = get_dummy(data, True)
    x_test = data.iloc[:,:-1]
    y_test = data.iloc[:,-1]
    SVM = svm.SVC(C = 9,kernel="linear")
    SVM.fit(x_train, y_train)
    accuarcy = cross_val_score(SVM, x_train, y_train).mean()
    print(accuarcy)
    predict = SVM.predict(x_test)
    print(accuracy_score(predict,y_test))


