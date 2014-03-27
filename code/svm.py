import pandas as pd
from sklearn.svm import LinearSVC


def svm(xTrain, yTrain, xTest):
    svmmodel = LinearSVC()
    svmmodel.fit(xTrain, yTrain)
    return svmmodel.predict(xTest)


def print_output(yTest, filename):
    output = {'Id': range(1, (len(yTest)+1)), 'Solution': yTest}
    withcounter = pd.DataFrame(output)
    withcounter.to_csv(filename, index=False)


xTrain = pd.read_csv('../data/train.csv', header=None)
yTrain = pd.read_csv('../data/trainLabels.csv', header=None)
yTrain = yTrain.values.flatten()

xTest = pd.read_csv('../data/test.csv', header=None)

yTest = svm(xTrain, yTrain, xTest)
print_output(yTest, '../data/submitsvm.csv')
