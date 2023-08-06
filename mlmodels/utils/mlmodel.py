import numpy as np
from random import sample
from sklearn import svm
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

class mlModel():
    def __init__(self, M_edges, T, mlModelType, ieeecase, modelConfig):
        super(mlModel, self).__init__()
        self.T = T
        self.M_edges = M_edges
        self.mlClassifiers = []
        self.trivialLine = np.zeros([M_edges,])
        self.mlModelType = mlModelType
        self.ieeecase = ieeecase
        if(modelConfig == None):
            self.modelConfig = self.getModelConfig(self.mlModelType.lower())
        else:
            self.modelConfig = modelConfig
        
    def trainModel(self, trainInputs, trainLabels):
        # train distinct ML models for each node
        self.mlClassifiers = []
        for ii in range(self.M_edges):
            if(np.unique(trainLabels[:,ii]).shape[0] <= 1): 
                self.trivialLine[ii] = 1
                continue
            yData = trainLabels[:,ii]

            self.mlClassifiers.append(self.getClassifier(trainInputs, yData))
        
    def evaluateModel(self, testInputs):
        # test the classifiers over train data
        y_pred = np.zeros(testInputs.shape)
        jj = 0
        for ii in range(self.M_edges):
            if(self.trivialLine[ii] == 1):
                y_pred[:,ii] = self.T
            else:
                y_pred[:,ii] = self.mlClassifiers[jj].predict(testInputs)
                jj = jj + 1

        return y_pred
    
    def getFinalStates(self, edgeLabels):
        finalState = np.zeros(edgeLabels.shape)
        finalState[edgeLabels > self.T-1] = 1

        return finalState

    def getClassifier(self, trainInputs, yData):
        match self.mlModelType.lower():
            case 'svm':
                return svm.LinearSVC(C=self.modelConfig['C'], dual=False, max_iter=self.modelConfig['max_iter']).fit(trainInputs, yData)
            case 'mnb':
                return MultinomialNB(alpha = self.modelConfig['alpha']).fit(trainInputs, yData)
            case 'catnb':
                return CategoricalNB(alpha = self.modelConfig['alpha']).fit(trainInputs, yData)
            case 'comnb':
                return ComplementNB(alpha = self.modelConfig['alpha']).fit(trainInputs, yData)
            case 'gnb':
                return GaussianNB().fit(trainInputs, yData)
            case 'bnb':
                return BernoulliNB(alpha = self.modelConfig['alpha']).fit(trainInputs, yData)
            case 'linr':
                return LinearRegression().fit(trainInputs, yData)
            case 'rr':
                return Ridge(alpha = self.modelConfig['alpha']).fit(trainInputs, yData)
            case 'logr':
                return LogisticRegression(C=self.modelConfig['C'], dual=False, max_iter=self.modelConfig['max_iter']).fit(trainInputs, yData)
            case 'dt':
                return DecisionTreeClassifier(criterion='entropy').fit(trainInputs, yData)
            case 'knn':
                return KNeighborsClassifier(n_neighbors = int(self.T+1)).fit(trainInputs, yData)    

    def getModelConfig(self, modelType):
        modelConfig = {}
        if(self.ieeecase.lower() == 'ieee89'):
            match modelType.lower():
                case 'svm':
                    modelConfig['C'] = 19.918
                    modelConfig['max_iter'] = 5000
                case 'mnb':
                    modelConfig['alpha'] = 0.5712
                case 'catnb':
                    modelConfig['alpha'] = 0.2816
                case 'comnb':
                    modelConfig['alpha'] = 0.7609
                case 'bnb':
                    modelConfig['alpha'] = 0.5712 #0.3907
                case 'rr':
                    modelConfig['alpha'] = 1.269
                case 'logr':
                    modelConfig['C'] = 4.705
                    modelConfig['max_iter'] = 5000
        elif(self.ieeecase.lower() == 'ieee118'):
            match modelType.lower():
                case 'svm':
                    modelConfig['C'] = 19.668
                    modelConfig['max_iter'] = 5000
                case 'mnb':
                    modelConfig['alpha'] = 0.564
                case 'catnb':
                    modelConfig['alpha'] = 0.154
                case 'comnb':
                    modelConfig['alpha'] = 1.894
                case 'bnb':
                    modelConfig['alpha'] = 0.564 #0.3792
                case 'rr':
                    modelConfig['alpha'] = 1.04
                case 'logr':
                    modelConfig['C'] = 5.849
                    modelConfig['max_iter'] = 5000
        elif(self.ieeecase.lower() == 'ieee1354'):
            match modelType.lower():
                case 'svm':
                    modelConfig['C'] = 20
                    modelConfig['max_iter'] = 5000
                case 'mnb':
                    modelConfig['alpha'] = 0.5
                case 'catnb':
                    modelConfig['alpha'] = 0.15
                case 'comnb':
                    modelConfig['alpha'] = 1.0
                case 'bnb':
                    modelConfig['alpha'] = 0.5 #0.4
                case 'rr':
                    modelConfig['alpha'] = 1.0
                case 'logr':
                    modelConfig['C'] = 5.0
                    modelConfig['max_iter'] = 5000
        return modelConfig