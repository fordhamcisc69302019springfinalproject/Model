from CISC6930_preprocessing import load_data,get_dummy,preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from  sklearn.metrics import *
import numpy as np
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
               'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
               'label']
    data = load_data("data/census-income.data.csv", columns)
    data = preprocess(data)
    #data = get_dummy(data, False)
    x_train = data.iloc[:,:-1]
    y_train = data.iloc[:,-1]
    data = load_data("data/census-income.data.csv", columns)
    #data = get_dummy(data, False)
    x_test = data.iloc[:,:-1]
    y_test = data.iloc[:,-1]
    #x_train,y_train,x_test,y_test = preprocess()
    scores = []
    accuarcy = []
    scoring = make_scorer(accuracy_score)
    max_depth = range(1,15)
    x_train = x_train.drop(["workclass"],axis = 1)
    rf = RandomForestClassifier(n_estimators = 500,max_depth=5,max_features = "auto",oob_score = True,warm_start = True)
    print(x_train)
    rf.fit(x_train,y_train)
    print(cross_val_score(rf,x_train,y_train,scoring=scoring).mean())
    #predict = rf.predict(x_test)
    #print(predict)
    #print(accuracy_score(predict,y_test))



    """
    x_train,y_train = get_data()
    print(x_train.shape)
    x_train = np.delete(x_train, 0, axis=1)
    print(x_train.shape)
    rf = RandomForestClassifier(n_estimators = 500,max_features = "auto",oob_score = True,warm_start = True)
    accuracy = cross_val_score(rf, x_train, y_train).mean()
    print(accuracy)
    """