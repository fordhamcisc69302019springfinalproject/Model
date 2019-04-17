from CISC6930_preprocessing import get_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV


def logistic_regression(x_train,y_train,penalty,featureselection = False):
    if penalty == "l1":
        lr = LogisticRegression(penalty="l1", solver='liblinear', max_iter=100)
    else:
        lr = LogisticRegression(penalty="l2", solver='sag', max_iter=100)

    if featureselection:
        lr = RFECV(lr, step=1, cv=10)
    lr.fit(x_train,y_train)
    if featureselection:
        accuracy = lr.score(x_train, y_train)
        print("Feature index:",lr.get_support(True))
    else:
        accuracy = cross_val_score(lr, x_train, y_train).mean()

    print("Accuracy :", accuracy)



if __name__ == "__main__":
    x_train, y_train = get_data()
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
    logistic_regression(x_train,y_train,"l2",True)