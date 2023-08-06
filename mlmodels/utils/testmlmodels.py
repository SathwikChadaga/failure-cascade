import numpy as np
import datetime

class testMlModels():
    def __init__(self, allMlModels, dataParamsML):
        super(testMlModels, self).__init__()
        self.allMlModels = allMlModels
        self.dataParamsML = dataParamsML

    def getFinalStates(self, scalingValue):
        mlModel = self.allMlModels.getMlModel(scalingValue)

        _, trainLabels = self.dataParamsML.getDataValues('Train', scalingValue)
        T = self.dataParamsML.TValuesAll['train_' + str(scalingValue)] 

        branchFailureFreq = np.sum((trainLabels != T)*(trainLabels != 0), axis=0)/np.sum((trainLabels != 0), axis=0)

        testInputs, testLabels = self.dataParamsML.getDataValues('Test', scalingValue)
        testFinalStates = (testLabels == T)

        predLabels = mlModel.evaluateModel(testInputs)
        predFinalStates = (predLabels == T)

        initialActiveEdges = (testLabels != 0)
        print('Tested on scaling value = ' + str(scalingValue) + ', predicted edge labels size = ' + str(predLabels.shape) + '.')
        return branchFailureFreq, initialActiveEdges, testFinalStates, predFinalStates, testLabels, predLabels
    
    def getAllFinalStates(self, customScalingList = None):
        if(customScalingList == None):
            customScalingList = self.dataParamsML.scalingList
        allBranchFailFreqs = None
        allTrueFinalStates = None
        allPredFinalStates = None
        for scalingValue in customScalingList:
            branchFailureFreq, testInitialStates, testFinalStates, predFinalStates, _, _ = self.getFinalStates(scalingValue)
            branchFailureFreq = np.expand_dims(branchFailureFreq, axis=0)
            testInitialStates = np.expand_dims(testInitialStates, axis=0)
            testFinalStates = np.expand_dims(testFinalStates, axis=0)
            predFinalStates = np.expand_dims(predFinalStates, axis=0)
            if(np.any(allBranchFailFreqs == None)):
                allBranchFailFreqs = branchFailureFreq
                allTrueFinalStates = testFinalStates
                allPredFinalStates = predFinalStates
            else:
                allBranchFailFreqs = np.append(allBranchFailFreqs, branchFailureFreq, axis=0) 
                allTrueFinalStates = np.append(allTrueFinalStates, testFinalStates, axis=0) 
                allPredFinalStates = np.append(allPredFinalStates, predFinalStates, axis=0) 

        return allBranchFailFreqs, allTrueFinalStates, allPredFinalStates

    def simulateCascadeTiming(self, onTrain, scalingValue):
            
        mlModel = self.allMlModels.getMlModel(scalingValue)

        
        testInputs, _ = self.dataParamsML.getDataValues(onTrain, scalingValue)

        startTime = datetime.datetime.now()
        _ = mlModel.evaluateModel(testInputs)
        endTime = datetime.datetime.now()

        timeDiff = (endTime - startTime)
        timeTaken_seconds = (timeDiff).seconds + 1e-6*(timeDiff).microseconds
        samplesRun = testInputs.shape[0]

        print('Ran timing analysis on scaling value = ' + str(scalingValue) + ', input size = ' + str(testInputs.shape) + '.')
        return timeTaken_seconds, samplesRun
    
    
    def collectAllProfileTiming(self, onTrain):
        # loop over various scaling values
        totalTime_seconds = 0
        totalSamplesRun = 0
        for scalingValue in self.dataParamsML.scalingList:
            # simulate cascade for this scaling value and append the results
            timeTaken, samplesRun = self.simulateCascadeTiming(onTrain, scalingValue)
            
            totalTime_seconds = totalTime_seconds + timeTaken
            totalSamplesRun = totalSamplesRun + samplesRun

        return totalTime_seconds, totalSamplesRun


