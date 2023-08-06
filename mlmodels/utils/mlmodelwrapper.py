import numpy as np
from tqdm.notebook import trange
from importlib import reload
import pickle
import mlmodel
reload(mlmodel)

class mlmodelWrapper():
    def __init__(self, dataParamsML, mlModelType, ieeecase, modelConfig = None):
        super(mlmodelWrapper, self).__init__()
        self.mlModelDict = {}
        self.dataParamsML = dataParamsML
        self.mlModelType = mlModelType.lower()
        self.modelConfig = modelConfig
        self.ieeecase = ieeecase

    def getMlModel(self, scalingValue):
        return self.mlModelDict[scalingValue]

    def learnMlModel(self, scalingValue):
        trainInput, trainLabels = self.dataParamsML.getDataValues('Train', scalingValue)

        M_edges = trainInput.shape[1]
        T = np.max(trainLabels)

        mlModel = mlmodel.mlModel(M_edges, T, self.mlModelType, self.ieeecase, self.modelConfig)
        mlModel.trainModel(trainInput, trainLabels)

        self.mlModelDict[scalingValue] = mlModel
    
    def learnAndSaveAllMlModels(self, modelFileName = None):
        if(modelFileName == None): # run training but do not save, and do not showing progress
            for ii in range(self.dataParamsML.scalingList.shape[0]):
                scalingValue = self.dataParamsML.scalingList[ii]
                self.learnMlModel(scalingValue)
        else: # run training and save while showing progress
            for ii in trange(self.dataParamsML.scalingList.shape[0]):
                scalingValue = self.dataParamsML.scalingList[ii]
                self.learnMlModel(scalingValue)
            with open(modelFileName, 'wb') as f:
                pickle.dump(self.mlModelDict, f)  

    def loadAllModels(self, modelFileName):
        with open(modelFileName, 'rb') as f:
            self.mlModelDict = pickle.load(f)

