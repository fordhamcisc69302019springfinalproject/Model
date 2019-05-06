from sklearn import neighbors
from CISC6930_preprocessing import *
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score



if __name__ == "__main__":
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
               'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
               'label']
    data = load_data("data/census-income.data.csv", columns)
    data = get_dummy(data, False)
    x_train = data.iloc[:, :-1]
    y_train = data.iloc[:, -1]
    x_train = x_train.drop(["workclass_3"],axis = 1)
    data = load_data("data/census-income.test", columns)
    data = get_dummy(data, False,False)
    x_test = data.iloc[:,:-1]
    y_test = data.iloc[:,-1]

    KNN = neighbors.KNeighborsClassifier(n_neighbors = 1)
    KNN.fit(x_train, y_train)
    #accuarcy = cross_val_score(KNN, x_train, y_train).mean()
    #print("k = %d,Accuracy: %f"%(1,accuarcy))
    predict = KNN.predict(x_test)
    print(accuracy_score(predict,list(y_test.values)))
