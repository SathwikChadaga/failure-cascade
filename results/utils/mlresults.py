import numpy as np
from importlib import reload
import mlmodelwrapper
reload(mlmodelwrapper)
import testmlmodels
reload(testmlmodels)
import dataparamsml
reload(dataparamsml)

class mlResults():
    def __init__(self, device, modelTypeList, ieeecase, scalingList, dataDirectoryBase, mlModelDirectory):
        super(mlResults, self).__init__()
        # self.device = device
        self.scalingList = scalingList
        self.modelTypeList = modelTypeList
        self.ieeecase = ieeecase

        dataParamsML = dataparamsml.dataParamsML(ieeecase, dataDirectoryBase, scalingList)
        self.testMlModels = {}
        for modelType in modelTypeList:
            mlModel = mlmodelwrapper.mlmodelWrapper(dataParamsML, modelType, ieeecase, None)
            mlModel.loadAllModels(mlModelDirectory  + '/models/' + modelType.lower() + '_' + ieeecase.lower() + '.pkl')
            self.testMlModels[modelType] = testmlmodels.testMlModels(mlModel, dataParamsML)

    def getGraphMetrics(self, modelType):
        imFailureSizeError = []
        imFinalStateError = []
        imFailureStepError = []
        for scalingValue in self.scalingList:
            # get predicted and true labels
            _, _, trueFinalStates, predFinalStates, trueFailureSteps, predFailureSteps = self.testMlModels[modelType].getFinalStates(scalingValue)

            # calculate failure sizes
            trueSize = np.sum(trueFinalStates == 0, axis=1)
            predSize = np.sum(predFinalStates == 0, axis=1)

            # calculate error metrics
            imFailureSizeError.append(100*np.mean(np.abs(trueSize - predSize)/np.abs(trueSize)))
            imFinalStateError.append(100*np.mean(predFinalStates != trueFinalStates))
            imFailureStepError.append(np.mean(np.abs(predFailureSteps - trueFailureSteps)))
        return self.scalingList, np.array(imFailureSizeError), np.array(imFinalStateError), np.array(imFailureStepError)
    
    def getBranchMetrics(self, modelType, scalingValue, numBins):
        # get predicted and true labels
        _, initialActiveEdges, trueFinalStates, predFinalStates, trueFailureSteps, predFailureSteps = self.testMlModels[modelType].getFinalStates(scalingValue)

        # get error in prediction of final state
        lffFinalState = np.sum((1 - trueFinalStates)*initialActiveEdges, axis=0)/np.sum(initialActiveEdges, axis=0)
        errorFinalStates = 100*np.sum((predFinalStates != trueFinalStates)*initialActiveEdges, axis=0)/np.sum(initialActiveEdges, axis=0)
        
        # get error in prediction of failure steps
        denom = np.sum((1-trueFinalStates)*initialActiveEdges*(predFinalStates == trueFinalStates), axis=0)
        denom[denom == 0] = 1
        errorFailureSteps = np.sum(np.abs(predFailureSteps - trueFailureSteps)*(predFinalStates == trueFinalStates)*(1-trueFinalStates)*initialActiveEdges,axis=0)/denom
        
        # plot results for only those lines that have failed and predicted correctly at least once
        validResultLines = (np.sum((1-trueFinalStates)*initialActiveEdges*(predFinalStates == trueFinalStates), axis=0) != 0)
        lffFailureStep = lffFinalState[validResultLines]
        errorFailureSteps = errorFailureSteps[validResultLines]

        # group into bins and plot averages
        lffFinalState, errorFinalStates = self.separateArrayIntoBins(lffFinalState, errorFinalStates, numBins)
        lffFailureStep, errorFailureSteps = self.separateArrayIntoBins(lffFailureStep, errorFailureSteps, numBins)
        return lffFailureStep, errorFailureSteps, lffFinalState, errorFinalStates  
      
    def separateArrayIntoBins(self, xVals, yVals, numBins):
        xValsRounded = np.round(numBins*xVals)/numBins
        xValsBins = np.unique(xValsRounded)

        yValsAvg = np.zeros(xValsBins.shape)
        for kk, xx in enumerate(xValsBins):
            if(np.sum(xValsRounded == xx) < 2): yValsAvg[kk] = np.nan
            else: yValsAvg[kk] = np.sum(yVals[xValsRounded == xx])/np.sum(xValsRounded == xx)
    #         yValsAvg[kk] = np.sum(yVals[xValsRounded == xx])/np.sum(xValsRounded == xx)

        return xValsBins, yValsAvg
    
    def getTimingMetrics(self, modelType, isTrain = 'Test'):
        return self.testMlModels[modelType].collectAllProfileTiming(isTrain)
