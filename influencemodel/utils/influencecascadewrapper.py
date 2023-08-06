import numpy as np
import torch
import influencecascade
from importlib import reload
reload(influencecascade)
import datetime

class failureCascadeWrapper():
    def __init__(self, dataParams, device, numOutputLabels = 25, imDirectory = None):
        super(failureCascadeWrapper, self).__init__()
        # save parameter values
        failureCascade  = influencecascade.failureCascade(device, dataParams, imDirectory)
        self.failureCascade = failureCascade
        self.dataParams = dataParams
        self.numOutputLabels = numOutputLabels

    def simulateCascade(self, powerScalingIndex, onTrain):
        scalingValue = self.dataParams.scalingList[powerScalingIndex]

        # change model settings to test
        self.failureCascade.updateWeights(scalingValue)

        # read test data values and initial/final time steps
        _, edgeFeatures, activeEdges = self.dataParams.readDataValues(onTrain, scalingValue)

        # simulate cascade for each cascade data samples
        print('Scaling value = ' + str(scalingValue) + ', initial contingencies matrix shape = ' + str(activeEdges.shape) + '.')
        errorVectorStatus = np.zeros(activeEdges.shape[1])
        lineFailureFrequency = np.zeros(activeEdges.shape[1])
        count = np.zeros(activeEdges.shape[1])

        for ii in range(activeEdges.shape[0]):
            # get true active edges and power values at initial and final time steps
            initialActiveEdges  = activeEdges[ii:ii+1,:]
            finalLineStatusTrue = (edgeFeatures[ii,:] == - 1).long().cpu().numpy()       

            # get predicted active edges and power values final time steps
            finalLineStatusPred = self.failureCascade.runCascadeSequence(initialActiveEdges)
            finalLineStatusPred = finalLineStatusPred.long().cpu().numpy()

            # update error values
            initialActiveEdges = initialActiveEdges[0,:].cpu().numpy()
            errorVectorStatus = errorVectorStatus + (finalLineStatusPred != finalLineStatusTrue)*(initialActiveEdges)
            lineFailureFrequency = lineFailureFrequency + (1-finalLineStatusTrue)*initialActiveEdges
            count = count + initialActiveEdges

        errorVectorStatus = errorVectorStatus/count
        lineFailureFrequency = lineFailureFrequency/count

        return errorVectorStatus, lineFailureFrequency
    
    def collectAllProfileErrors(self, onTrain):
        # loop over various scaling values
        errorVectorList = []
        lineFailureFrequencyList = []
        for powerScalingIndex in range(self.dataParams.scalingList.shape[0]):

            # simulate cascade for this scaling value and append the results
            errorVector, lineFailureFrequency = self.simulateCascade(powerScalingIndex, onTrain)
            errorVectorList.append(errorVector)
            lineFailureFrequencyList.append(lineFailureFrequency)

        return errorVectorList, lineFailureFrequencyList
    
    
    def getFinalStatesWithFailureSteps(self, scalingValue, onTrain):
        # change model settings to test
        self.failureCascade.updateWeights(scalingValue)

        # read test data values and initial/final time steps
        _, edgeFeatures, activeEdges = self.dataParams.readDataValues(onTrain, scalingValue)

        # simulate cascade for each cascade data samples
        allTrueFinalStates = np.zeros(activeEdges.shape)
        allPredFinalStates = np.zeros(activeEdges.shape)
        allInitialActiveEdges = np.zeros(activeEdges.shape)
        allFailureSteps = np.zeros(activeEdges.shape)

        for ii in range(activeEdges.shape[0]):
            # get true active edges and power values at initial and final time steps
            initialActiveEdges  = activeEdges[ii:ii+1,:]
            finalLineStatusTrue = (edgeFeatures[ii,:] == - 1).long().cpu().numpy()       

            # get predicted active edges and power values final time steps
            finalLineStatusPred, failureSteps = self.failureCascade.runCascadeSequenceWithFailureSteps(initialActiveEdges)
            finalLineStatusPred = finalLineStatusPred.long().cpu().numpy()
            failureSteps = failureSteps.long().cpu().numpy()

            # update error values
            initialActiveEdges = initialActiveEdges[0,:].cpu().numpy()
            
            allTrueFinalStates[ii,:] = finalLineStatusTrue
            allPredFinalStates[ii,:] = finalLineStatusPred
            allInitialActiveEdges[ii,:] = initialActiveEdges
            allFailureSteps[ii,:] = failureSteps
        
        trueFailureSteps = edgeFeatures.float().cpu().numpy()
        maxLength = self.numOutputLabels-1
        trueFailureSteps[trueFailureSteps == -1] = maxLength
        allFailureSteps[allFailureSteps == -1] = maxLength

        print('Tested on scaling value = ' + str(scalingValue) + ', predicted edge labels size = ' + str(allFailureSteps.shape) + '.')

        return allInitialActiveEdges, allTrueFinalStates, allPredFinalStates, allFailureSteps, trueFailureSteps
    
    
    def simulateCascadeTiming(self, onTrain, scalingValue):
        # change model settings to test
        self.failureCascade.updateWeights(scalingValue)

        # read test data values and initial/final time steps
        _, edgeFeatures, activeEdges = self.dataParams.readDataValues(onTrain, scalingValue)

        # simulate cascade for each cascade data samples
        timeTaken_seconds = 0
        samplesRun = 0

        for ii in range(activeEdges.shape[0]):
            # get true active edges and power values at initial and final time steps
            initialActiveEdges  = activeEdges[ii:ii+1,:]

            # get predicted active edges and power values final time steps
            startTime = datetime.datetime.now()
            _ = self.failureCascade.runCascadeSequence(initialActiveEdges)
            endTime = datetime.datetime.now()
            timeDiff = (endTime - startTime)
            currTimeTaken_seconds = (timeDiff).seconds + 1e-6*(timeDiff).microseconds
            timeTaken_seconds = timeTaken_seconds + currTimeTaken_seconds
            samplesRun = samplesRun + 1
        print('Ran timing analysis on scaling value = ' + str(scalingValue) + ', input size = ' + str(activeEdges.shape) + '.')
        return timeTaken_seconds, samplesRun
    
    def collectAllProfileTiming(self, onTrain):
        # loop over various scaling values
        totalTime_seconds = 0
        totalSamplesRun = 0
        for scalingValue in self.dataParams.scalingList:
            # simulate cascade for this scaling value and append the results
            timeTaken, samplesRun = self.simulateCascadeTiming(onTrain, scalingValue)
            
            totalTime_seconds = totalTime_seconds + timeTaken
            totalSamplesRun = totalSamplesRun + samplesRun

        return totalTime_seconds, totalSamplesRun