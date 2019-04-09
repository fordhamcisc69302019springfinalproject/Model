import Simple_preprocess
train_data, train_label, test_data, test_label = Simple_preprocess.preprocess()

def predict(classifier,test = True):
    classifier.fit(train_data,train_label)
    if test:
        return classifier.predict(test_data)
    else:
        return classifier.predict(train_data)