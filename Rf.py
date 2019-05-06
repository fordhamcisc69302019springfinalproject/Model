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
    data = preprocess(data, False)
    x_train = data.iloc[:, :-1]
    y_train = data.iloc[:, -1]
    data = load_data("data/census-income.test", columns)
    data = preprocess(data, False)
    x_test = data.iloc[:,:-1]
    y_test = data.iloc[:,-1]
    #x_train,y_train,x_test,y_test = preprocess()
    scores = []
    accuarcy = []
    scoring = make_scorer(accuracy_score)
    rf = RandomForestClassifier(max_features = "log2",n_estimators = 700,max_depth=27,oob_score = True,warm_start = True)
    rf.fit(x_train,y_train)
    print(cross_val_score(rf,x_train,y_train,scoring=scoring).mean())
    predict = rf.predict(x_test)
    print(accuracy_score(predict,y_test))



    """
    x_train,y_train = get_data()
    print(x_train.shape)
    x_train = np.delete(x_train, 0, axis=1)
    print(x_train.shape)
    rf = RandomForestClassifier(n_estimators = 500,max_features = "auto",oob_score = True,warm_start = True)
    accuracy = cross_val_score(rf, x_train, y_train).mean()
    print(accuracy)
    """