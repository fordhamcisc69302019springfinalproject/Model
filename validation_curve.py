import matplotlib.pyplot as plt
from CISC6930_preprocessing import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from  sklearn.metrics import *
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
from sklearn import svm
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
rf = RandomForestClassifier()
KNN = neighbors.KNeighborsClassifier()
SVM = svm.SVC(kernel="sigmoid")

#get data
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
           'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
           'label']
data = load_data("data/census-income.data.csv", columns)
data = preprocess(data)
#data = get_dummy(data, False)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

scoring = make_scorer(accuracy_score)
l1 = LogisticRegression(penalty="l1", solver='liblinear', max_iter=100)
l2 = LogisticRegression(penalty="l2", solver='sag', max_iter=100)
param_range = np.array(list(range(1,10)))
train_scores, test_scores = validation_curve(
    SVM, X, y, param_name="C", param_range=param_range,
    cv=5, scoring=scoring, n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
print("sigmoid")
print("train_scores",train_scores_mean)
print("test socore",test_scores_mean)
test_scores_mean = list(test_scores_mean)
print(test_scores_mean.index(max(test_scores_mean)))

plt.title("Validation Curve")
plt.xlabel("max_depth")
plt.ylabel("C")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.savefig("./pictures/validation_curve/svm_sigmoid_c.png")
plt.show()