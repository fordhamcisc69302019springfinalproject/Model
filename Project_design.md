##Problems

###data preprocess

1. "country_ Holand-Netherlands" not in test_data, therefore, normialization may cause dimension in train data and test data not equal. 
2. The labels in train data and test data are not the same

###model

#####single models

1.Because there are more than 100 features after simple preprocess, the SVM model fit extremely slow. Therefore, we need feature selection after normilization.
 
##Design

###data preprocess

1.Do normializtion:  
Z_socre:Age, flnwight,education,hours   
Max_min: gain,loss  
二进制特征处理：discrete feature

2.feature selection

question:
Whether do different feature selection for different method 

###Model

1.single model(decision_tree,svm,knn,RF)
2.ensemble method(voting,bagging,boosting)
3.validation,model selection,optimize parameter(weight in voting method)


