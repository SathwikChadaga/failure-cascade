import numpy as np
from importlib import reload
import dataparamsgnn
reload(dataparamsgnn)
import dataparamsim
reload(dataparamsim)
import influencecascadewrapper
reload(influencecascadewrapper)

class imResults():
    def __init__(self, device, ieeecase, scalingList, dataDirectoryBase, imDirectoryBase):
        super(imResults, self).__init__()
        self.device = device
        numOutputLabels = dataparamsgnn.dataParams(device, ieeecase, ['random'], 'Train', None, dataDirectoryBase, False).numOutputLabels
        self.dataParamsIM  = dataparamsim.dataParams(device, ieeecase,  dataDirectoryBase, scalingList)
        self.failureCascadeWrapper = influencecascadewrapper.failureCascadeWrapper(self.dataParamsIM, device, numOutputLabels, imDirectoryBase + '/data/')

    def getGraphMetrics(self):
        imFailureSizeError = []
        imFinalStateError = []
        imFailureStepError = []
        for scalingValue in self.dataParamsIM.scalingList:
            _, trueFinalStates, predFinalStates, predFailureSteps, trueFailureSteps = self.failureCascadeWrapper.getFinalStatesWithFailureSteps(scalingValue, 'Test')
            
            trueSize = np.sum(trueFinalStates == 0, axis=1)
            predSize = np.sum(predFinalStates == 0, axis=1)
            
            imFailureSizeError.append(100*np.mean(np.abs(trueSize - predSize)/np.abs(trueSize)))
            imFinalStateError.append(100*np.mean(predFinalStates != trueFinalStates))
            imFailureStepError.append(np.mean(np.abs(predFailureSteps - trueFailureSteps)))
            
        return self.dataParamsIM.scalingList, np.array(imFailureSizeError), np.array(imFinalStateError), np.array(imFailureStepError)
    
    def getBranchMetrics(self, scalingValue, numBins = 10):
        # get predicted and true labels
        initialActiveEdges, trueFinalStates, predFinalStates, predFailureSteps, trueFailureSteps = self.failureCascadeWrapper.getFinalStatesWithFailureSteps(scalingValue, 'Test')
        
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
        
    def getTimingMetrics(self, isTrain = 'Test'):
        return self.failureCascadeWrapper.collectAllProfileTiming(isTrain)
    
    def separateArrayIntoBins(self, xVals, yVals, numBins):
        xValsRounded = np.round(numBins*xVals)/numBins
        xValsBins = np.unique(xValsRounded)

        yValsAvg = np.zeros(xValsBins.shape)
        for kk, xx in enumerate(xValsBins):
            if(np.sum(xValsRounded == xx) < 2): yValsAvg[kk] = np.nan
            else: yValsAvg[kk] = np.sum(yVals[xValsRounded == xx])/np.sum(xValsRounded == xx)
    #         yValsAvg[kk] = np.sum(yVals[xValsRounded == xx])/np.sum(xValsRounded == xx)

        return xValsBins, yValsAvg
