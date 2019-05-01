# Model
The model and algorithm implementation.

###Single Model Implement
All the accuarcy below is 10 folders validation using training data
####logistica regression
I have implement feature selection using RFECV which provides Feature ranking with recursive feature elimination and cross-validated selection of the best number of features
* L1 penalty  
Accuracy:  0.7908980582524272  
Feature index: [ 0  1  3  4  5  7  8  9 10 11 12]
* L2 penalty  
Accuracy:  0.7768001618122977  
Feature index: [ 0  1  4  5  7  8  9 10 12]
  
It often shows warning like that  
/Users/fuyuqi/Library/Python/3.7/lib/python/site-packages/sklearn/linear_model/sag.py:334: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  "the coef_ did not converge", ConvergenceWarning)  
  
 Without feature selection
 * L1 penalty  
 Accuracy:  0.7916059870550162
 * L2 penalty  
 Accuracy:  0.5003033980582524

####Random Forest
* When max depth is 8, we get maximum accuracy with 0.8597087378640776

####KNN
