import numpy as np
import pandas as pd

train_path = "./data/census-income.data.csv"
test_path = "./data/census-income.test"

def load_file(file_path):
    return pd.DataFrame(pd.read_csv(file_path,names = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation"
                                     ,"relationshop","race","sex","capital-gain","capital-loss","hours","country","label"]))


def consistent_label(pd_dataframe):
    pd_dataframe["label"] = pd_dataframe["label"].replace([" >50K.", " <=50K."], [" >50K", " <=50K"])
    return pd_dataframe

def dumpies_variables(pd_dataframe,testortrain):
    if testortrain == "train":
        return pd.get_dummies(pd_dataframe,columns = ["workclass","education","marital-status","occupation","relationshop",
                                                      "race","sex","country"]).drop(["country_ Holand-Netherlands"],axis = 1)
    elif testortrain == "test":
        return pd.get_dummies(pd_dataframe,
                              columns=["workclass", "education", "marital-status", "occupation", "relationshop",
                                       "race", "sex", "country"])

def split_data_label(pd_dataframe):
    return pd_dataframe.iloc[:,:-1],pd_dataframe.iloc[:,-1]

def preprocess():
    train_file = train_path
    test_file = test_path

    # load data
    train = load_file(train_file)
    test = load_file(test_file)

    # consist label
    test = consistent_label(test)

    # split
    train_data, train_label = split_data_label(train)
    test_data,test_label = split_data_label(test)

    #dumpies_variables
    train_data = dumpies_variables(train_data,"train")
    test_data = dumpies_variables(test_data,"test")
    return train_data,train_label,test_data,test_label
