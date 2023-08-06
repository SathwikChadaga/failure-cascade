# Run cascade sequence predictions using the GNN model. 
# The class has two main functions
# 1. collectAllProfileErrors - this collects the error metrics for all loading profiles (scaling values)
# 2. collectAllProfileTiming - this collects runtime metrics for all loading profiles (scaling values)
# The other functions are helpers for the above two main functions.

import numpy as np
import torch
import datetime

class cascadeAnalysis():
    def __init__(self, device, pfModel, dataParams):
        super(cascadeAnalysis, self).__init__()
        # save data class and the  models class
        self.dataParams = dataParams
        self.pfModel = pfModel
        self.device = device
  
    def getPredictedValuesInBatches(self, nodeFeatures, activeEdges):
        # testing on all samples can lead to memory overflow, so run test in batches
        predEdgeFeatures = torch.zeros([activeEdges.shape[0], activeEdges.shape[1], self.dataParams.numOutputLabels])
        ii = 0; testBatchSize = 64
        while(ii < activeEdges.shape[0]):
            predEdgeFeaturesCurrent, _ = self.pfModel.validatenn(nodeFeatures[ii:ii+testBatchSize,:,:].to(self.device), activeEdges[ii:ii+testBatchSize,:].to(self.device))
            predEdgeFeatures[ii:ii+testBatchSize,:,:] = predEdgeFeaturesCurrent.cpu()
            ii = ii + testBatchSize
        return predEdgeFeatures  
    
    def simulateCascade(self, onTrain, scalingValue):
        # get input and true output labels
        nodeFeatures, edgeFeatures, activeEdges = self.dataParams.getSpecificBatch(onTrain, scalingValue)

        # get true final states
        finalStateTrue = (edgeFeatures == (self.dataParams.numOutputLabels - 1)).long().cpu().numpy()

        # get predicted final states
        # run test in batches in case data size is too big for the gpu
        predEdgeFeatures = self.getPredictedValuesInBatches(nodeFeatures, activeEdges)
        predEdgeFeatureLabels = torch.argmax(predEdgeFeatures, dim=-1)
        finalStatePred = (predEdgeFeatureLabels == (self.dataParams.numOutputLabels - 1)).long().cpu().numpy()
        print('Tested on scaling value = ' + str(scalingValue) + ', predicted edge labels size = ' + str(predEdgeFeatures.shape) + '.')

        # calculate error and line frequency vectors
        activeEdges_np = activeEdges.long().cpu().numpy()
        errorVector = np.sum((finalStatePred != finalStateTrue)*activeEdges_np, axis=0)/np.sum(activeEdges_np, axis=0)
        lineFailureFrequency = np.sum((1-finalStateTrue)*activeEdges_np, axis=0)/np.sum(activeEdges_np, axis=0)

        return errorVector, lineFailureFrequency, predEdgeFeatures, predEdgeFeatureLabels, nodeFeatures, edgeFeatures, activeEdges

    def collectAllProfileErrors(self, onTrain):
        # loop over various scaling values
        errorVectorList = []
        lineFailureFrequencyList = []
        for scalingValue in self.dataParams.scalingList:
            # simulate cascade for this scaling value and append the results
            errorVector, lineFailureFrequency,_,_,_,_,_ = self.simulateCascade(onTrain, scalingValue)
            
            errorVectorList.append(errorVector)
            lineFailureFrequencyList.append(lineFailureFrequency)

        return errorVectorList, lineFailureFrequencyList
    
    def simulateCascadeTiming(self, onTrain, scalingValue):
        # get input and true output labels
        nodeFeatures, edgeFeatures, activeEdges = self.dataParams.getSpecificBatch(onTrain, scalingValue)

        # set timers to zero
        timeTaken_seconds = 0
        samplesRun = 0
        ii = 0
        testBatchSize = activeEdges.shape[0]
        while(ii < activeEdges.shape[0]):
            # time the runs for each batch
            startTime = datetime.datetime.now()
            _, _ = self.pfModel.validatenn(nodeFeatures[ii:ii+testBatchSize,:,:].to(self.device), activeEdges[ii:ii+testBatchSize,:].to(self.device))
            endTime = datetime.datetime.now()
            timeDiff = (endTime - startTime)

            # update counters
            timeTaken_seconds = timeTaken_seconds + (timeDiff).seconds + 1e-6*(timeDiff).microseconds
            samplesRun = samplesRun + nodeFeatures[ii:ii+testBatchSize,:,:].shape[0]
            ii = ii + testBatchSize
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