import getopt
import sys

import joblib
import numpy
import pandas
from IPython.display import display
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocessing import DataFrameImputer
from preprocessing import DataFrameOneHotEncoder
from preprocessing import binarize


class EmpathyClassification:

    def __init__(self, dataFile='data/responses.csv', outputFile='data/bestModel.pkl',
                 testSetOutputFile='data/testSet.csv', randomSeed=42, testSetSize=0.2):
        self.dataFile = dataFile
        self.outputFile = outputFile
        self.testSetOutputFile = testSetOutputFile
        self.seed = randomSeed
        self.testSize = testSetSize
        self.preprocessor = None

        # Define Classifiers
        self.classifiers = [
            ('Most-frequent Classifier (Baseline)', DummyClassifier(), [
                {
                    'strategy': ['most_frequent']
                }
            ]),
            ('KNN', KNeighborsClassifier(n_jobs=-1), [
                {
                    'n_neighbors': range(1, 21)
                }
            ]),
            ('Logistic Regression', LogisticRegression(max_iter=10000), [
                {
                    'C': numpy.logspace(1e-4, 10, num=50),
                    'solver': ['liblinear']
                },
                {
                    'C': numpy.logspace(1e-4, 10, num=50),
                    'solver': ['lbfgs'],
                    'n_jobs': [-1]
                }
            ]),
            ('Gaussian Naive Bayes', GaussianNB(), [{}]),
            ('Perceptron', Perceptron(random_state=self.seed, tol=1e-3), [
                {
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'max_iter': [30, 35, 40, 45, 50, 75, 100, 250, 500, 1000, 1500],
                    'eta0': [0.1 * i for i in range(1, 11)]
                }
            ]),
            ('Decision Tree', DecisionTreeClassifier(random_state=self.seed), [
                {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [10, 15, 20, 30, 50, 100, 150, 200, None]
                }
            ]),
            ('Random Forest', RandomForestClassifier(random_state=self.seed, n_jobs=-1), [
                {'n_estimators': range(50, 501, 10), 'criterion': ['gini', 'entropy']}
            ]),
            ('SVM', SVC(random_state=self.seed), [
                {
                    'kernel': ['linear', 'rbf'],
                    'C': [0.01 * i for i in range(1, 101, 5)],
                    'gamma': ['auto', 'scale']
                },
                {
                    'kernel': ['poly'],
                    'C': [0.01 * i for i in range(1, 101, 5)],
                    'gamma': ['auto', 'scale'],
                    'degree': range(2, 6)
                }
            ])
        ]

        self.gridSearches = []
        self.results = None
        self.bestModels = {}
        self.bestOverallModel = None

    def loadData(self, filename=None):
        csv = pandas.read_csv(self.dataFile if filename is None else filename)

        # Drop rows where dependent variable is missing
        csv.dropna(subset=['Empathy'], inplace=True)

        # Separate dependent and independent variables
        Yall = csv['Empathy']
        Xall = csv.drop(labels=['Empathy'], axis=1)

        # Binarize dependent variable s.t. y=[1,2,3] => 0, and y=[4,5] => 1
        Yall = binarize(Yall, threshold=3)

        return Xall, Yall

    def splitTrainAndTestSet(self, Xall, Yall):
        return train_test_split(Xall, Yall, test_size=0.2, random_state=self.seed)

    def doPreprocessing(self, Xtrain, Ytrain):
        # Impute missing values with mode and one-hot encode categorical variables

        self.preprocessor = Pipeline([
            ('imputer', DataFrameImputer()),
            ('onehot', DataFrameOneHotEncoder()),
            ('scaling', MinMaxScaler()),
            ('feature_selection', SelectFromModel(
                estimator=RandomForestClassifier(
                    random_state=self.seed,
                    criterion='entropy',
                    n_estimators=70, n_jobs=-1
                )
            ))
        ])
        self.preprocessor.fit(Xtrain, Ytrain)

        return self.preprocessor

    def applyPreprocessing(self, X):
        return self.preprocessor.transform(X)

    def trainClassifiers(self, Xtrain, Ytrain):
        results = {"Classifier": [], "Best Parameters": [], "CV Accuracy": []}
        for name, classifier, params in self.classifiers:
            print("\n\nTraining {} ...".format(name))
            gridSearch = GridSearchCV(classifier, param_grid=params,
                                      cv=StratifiedKFold(n_splits=8, random_state=self.seed),
                                      scoring='accuracy', n_jobs=-1)
            gridSearch.fit(Xtrain, Ytrain)

            self.gridSearches.append(gridSearch)
            results["Classifier"].append(name)
            results["Best Parameters"].append(gridSearch.best_params_)
            results["CV Accuracy"].append(gridSearch.best_score_)
            self.bestModels[name] = gridSearch.best_estimator_

            print("CV Accuracy:", gridSearch.best_score_)
            print("Best parameters:", gridSearch.best_params_)

        self.results = pandas.DataFrame(results)

    def findBestPerformingModel(self):
        bestPerformingModelName = self.results.iloc[self.results["CV Accuracy"].idxmax()]['Classifier']
        print("Best performing model:", bestPerformingModelName)
        self.bestOverallModel = self.bestModels[bestPerformingModelName]
        return self.bestOverallModel

    def writeTestSet(self, Xtest, Ytest):
        df = Xtest.copy()
        df['Empathy'] = Ytest
        df.to_csv(self.testSetOutputFile, index=False)

    def saveModel(self):
        saveData = {
            'preprocessor': self.preprocessor,
            'bestOverallModel': self.bestOverallModel,
            'gridSearches': self.gridSearches
        }
        joblib.dump(saveData, self.outputFile)

        print("Best performing model saved at:", self.outputFile)

    def loadModel(self, modelFile=None):
        if modelFile is None:
            modelFile = self.outputFile

        data = joblib.load(modelFile)
        self.preprocessor = data["preprocessor"]
        self.bestOverallModel = data['bestOverallModel']
        self.gridSearches = data['gridSearches']

        print("Best performing model loaded from:", modelFile)


def main(mode='test', dataFile='testSet.csv', modelFile='bestModel.pkl'):
    if (mode == 'train'):

        # Create instance of EmpathyClassification
        classification = EmpathyClassification(dataFile=dataFile, outputFile=modelFile)

        # Load dataset
        Xall, Yall = classification.loadData()

        # Split dataset into test and train
        Xtrain, Xtest, Ytrain, Ytest = classification.splitTrainAndTestSet(Xall, Yall)
        classification.writeTestSet(Xtest, Ytest)

        # Fit preprocessor
        classification.doPreprocessing(Xtrain, Ytrain)

        # Apply results of preprocessing
        print("Number of features before preprocessing:", Xtrain.shape[1])
        Xtrain = classification.applyPreprocessing(Xtrain)
        Xtest = classification.applyPreprocessing(Xtest)
        print("Number of features after preprocessing:", Xtrain.shape[1])

        # Train classifiers
        classification.trainClassifiers(Xtrain, Ytrain)
        print("\n\nResults:\n")
        display(classification.results)

        # Dump trained model and preprocessing objects
        bestModel = classification.findBestPerformingModel()
        classification.saveModel()

        # Test baseline model on test set
        accuracy = classification.gridSearches[0].best_estimator_.score(Xtest, Ytest)
        print("\nAccuracy of most-frequent (baseline) classifier on test set:", accuracy)

        # Test best model on test set
        accuracy = bestModel.score(Xtest, Ytest)
        print("Accuracy of best performing classifier on test set:", accuracy)

    elif mode == 'test':
        # Load test set
        classification = EmpathyClassification(testSetOutputFile=dataFile, outputFile=modelFile)
        Xtest, Ytest = classification.loadData(dataFile)

        # Load trained model
        classification.loadModel()

        # Apply preprocess steps
        Xtest = classification.applyPreprocessing(Xtest)

        # Score baseline model
        accuracy = classification.gridSearches[0].best_estimator_.score(Xtest, Ytest)
        print("\nAccuracy of most-frequent (baseline) classifier on test set:", accuracy)

        # Score best model on test data
        accuracy = classification.bestOverallModel.score(Xtest, Ytest)
        print("Accuracy of best performing classifier on test set:", accuracy)

    else:
        print("Invalid mode:", mode)


def usage():
    print(
        "Usage:\n\tpy main.py --mode=<train|test> --dataset=<path to responses.csv | path to test set> --model=<path to load/write trained model> [-h --help]")


if __name__ == '__main__':
    argv = sys.argv[1:]

    # argv = "--mode=train --dataset=data/responses.csv --model=data/bestModel.pkl".split(' ')
    # argv = "--mode=test --dataset=data/testSet.csv --model=data/bestModel.pkl".split(' ')

    dataFile = 'data/responses.csv'
    modelFile = 'data/bestModel.pkl'
    mode = 'train'

    try:
        opts, args = getopt.getopt(argv, 'hm:d:l:', ['help', 'mode=', 'dataset=', 'model='])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)

    if len(opts) == 0:
        usage()
        sys.exit()

    for option, value in opts:
        if option in ('-h', '--help'):
            usage()
            sys.exit()
        elif option in ('-m', '--mode'):
            if value.lower() in ('train', 'test'):
                mode = value
            else:
                sys.exit("Unknown mode: mode must be either 'train' ot 'test'")
        elif option in ('-d', '--dataset'):
            dataFile = value
        elif option in ('-l', '--model'):
            modelFile = value
        else:
            sys.exit("Unknown option: {}".format(option))

    main(mode, dataFile, modelFile)
