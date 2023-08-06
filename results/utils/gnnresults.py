import numpy as np
import torch
import matplotlib.pyplot as plt
from importlib import reload
import dataparamsgnn
reload(dataparamsgnn)
import nnmodel
reload(nnmodel)
import cascadewrapper
reload(cascadewrapper)

class gnnResults():
    def __init__(self, device, ieecase, scalingList, dataDirectoryBase):
        super(gnnResults, self).__init__()
        self.device = device
        self.dataParamsRandom = dataparamsgnn.dataParams(device, ieecase, ['random'], 'Train', None, dataDirectoryBase, False)
        self.dataParams = dataparamsgnn.dataParams(device, ieecase, scalingList, 'Train', self.dataParamsRandom.numOutputLabels, dataDirectoryBase, False)
        self.pfModel = self.getGNNModel(device, self.dataParamsRandom)

    def getGraphMetricsPerSample(self, scalingValue):
        if(scalingValue == 'random'): dataParamsCurr = self.dataParamsRandom
        else: dataParamsCurr = self.dataParams
        device = self.device

        # simulate cascades
        cascadeAnalysis = cascadewrapper.cascadeAnalysis(device, self.pfModel, dataParamsCurr)
        _, _, _, predEdgeFeatureLabels, nodeFeatures, edgeFeatures, _ = cascadeAnalysis.simulateCascade('Test', scalingValue)
        
        # get final states
        trueFinalStates = (edgeFeatures == (dataParamsCurr.numOutputLabels-1)).to(device)
        predFinalStates = (predEdgeFeatureLabels == (dataParamsCurr.numOutputLabels-1)).to(device)
        
        # get failure sizes
        trueFailureSize = torch.sum(trueFinalStates == False, dim=1).to(device)
        predFailureStates = torch.sum(predFinalStates == False, dim=1).to(device)

        # calculate graph level metrics
        failureSizeError = 100*torch.abs(trueFailureSize - predFailureStates)/torch.abs(trueFailureSize)
        finalStateError = 100*torch.mean((trueFinalStates != predFinalStates).float(), dim=1)
        failureStepError = torch.mean(torch.abs(edgeFeatures.to(device) - predEdgeFeatureLabels.to(device)).float(), dim=1)
        
        # convert results to numpy and bring to cpu
        failureSizeError_np = failureSizeError.cpu().numpy()
        finalStateError_np = finalStateError.cpu().numpy()
        failureStepError_np = failureStepError.cpu().numpy()
        
        # if scaling is 'random', then return load scaling values of each sample
        if(scalingValue == 'random'):
            possibleScalingValues = torch.sort(torch.rand(nodeFeatures.shape[0])*(2.05-0.95) + 0.95).values
            powerInjections = torch.mean(torch.abs(nodeFeatures[:,:,0] - nodeFeatures[:,:,1]), dim=1)
            scalingValues = torch.zeros(powerInjections.shape)
            scalingValues[torch.argsort(powerInjections)] = possibleScalingValues
            scalingValues_np = scalingValues.cpu().numpy()

            return failureSizeError_np, finalStateError_np, failureStepError_np, scalingValues_np
        # else return only the resultant vectors
        else: 
            return failureSizeError_np, finalStateError_np, failureStepError_np

    def getGraphMetricsRandomScaling(self, numBins = 21):
        failureSizeError_curr, finalStateError_curr, failureStepError_curr, scalingValues_curr = self.getGraphMetricsPerSample('random')
        scalingValues_bins, failureSizeError = self.groupScalingIntoBins(scalingValues_curr, failureSizeError_curr, numBins)
        _, finalStateError = self.groupScalingIntoBins(scalingValues_curr, finalStateError_curr, numBins)
        _, failureStepError = self.groupScalingIntoBins(scalingValues_curr, failureStepError_curr, numBins)
        return scalingValues_bins, failureSizeError, finalStateError, failureStepError
        
    def getGraphMetricsKnownScaling(self, scalingList):
        for index, scalingValue in enumerate(scalingList):
            failureSizeError_curr, finalStateError_curr, failureStepError_curr = self.getGraphMetricsPerSample(scalingValue)
            if(index == 0):
                failureSizeError = np.zeros([scalingList.shape[0], failureSizeError_curr.shape[0]])
                finalStateError  = np.zeros([scalingList.shape[0], finalStateError_curr.shape[0]])
                failureStepError = np.zeros([scalingList.shape[0], failureStepError_curr.shape[0]])

            failureSizeError[index, :] = failureSizeError_curr
            finalStateError[index, :]  = finalStateError_curr
            failureStepError[index, :] = failureStepError_curr      
        return scalingList, failureSizeError, finalStateError, failureStepError
    
    def getGraphMetrics(self, numBins = 21, scalingList = None):
        if(scalingList == None): scalingList = self.dataParams.scalingList
        scalingValsRand, failureSizeErrorRand, finalStateErrorRand, failureStepErrorRand = self.getGraphMetricsRandomScaling(numBins)
        scalingValsKnow, failureSizeErrorKnow, finalStateErrorKnow, failureStepErrorKnow = self.getGraphMetricsKnownScaling(scalingList)

        failureSizeError = np.mean(failureSizeErrorRand, axis=1)
        finalStateError = np.mean(finalStateErrorRand, axis=1)
        failureStepError = np.mean(failureStepErrorRand, axis=1)
        for indexKnow, scalingValue in enumerate(scalingValsKnow):
            indexRand = np.argmin(np.abs(scalingValsRand - scalingValue))
            failureSizeError[indexRand] = np.mean(np.append(failureSizeErrorRand[indexRand,:], failureSizeErrorKnow[indexKnow,:]))
            finalStateError[indexRand]  = np.mean(np.append(finalStateErrorRand[indexRand,:],  finalStateErrorKnow[indexKnow,:]))
            failureStepError[indexRand] = np.mean(np.append(failureStepErrorRand[indexRand,:], failureStepErrorKnow[indexKnow,:]))
            
        return scalingValsRand, failureSizeError, finalStateError, failureStepError
    
    def getBranchMetrics(self, scalingValue, numBins = 10):
        if(scalingValue == 'random'): dataParamsCurr = self.dataParamsRandom
        else: dataParamsCurr = self.dataParams

        # get predicted and true labels
        cascadeAnalysis = cascadewrapper.cascadeAnalysis(self.device, self.pfModel, dataParamsCurr)
        _, _, _, predEdgeFeatureLabels, _, edgeFeatures, activeEdges = cascadeAnalysis.simulateCascade('Test', scalingValue)

        # get initial contingencies
        initialActiveEdges = activeEdges.long().cpu().numpy()

        # get final states
        predFinalStates = (predEdgeFeatureLabels == dataParamsCurr.numOutputLabels-1).cpu().numpy()
        trueFinalStates = (edgeFeatures == dataParamsCurr.numOutputLabels-1).cpu().numpy()

        # get failure steps
        predFailureSteps = predEdgeFeatureLabels.long().cpu().numpy()
        trueFailureSteps = edgeFeatures.long().cpu().numpy()
        
        # calculate branch-level final state error metric
        lffFinalState = np.sum((1 - trueFinalStates)*initialActiveEdges, axis=0)/np.sum(initialActiveEdges, axis=0)
        errorFinalStates = 100*np.sum((predFinalStates != trueFinalStates)*initialActiveEdges, axis=0)/np.sum(initialActiveEdges, axis=0)
        
        # calculate branch-level failure step error metric
        denom = np.sum((1-trueFinalStates)*initialActiveEdges*(predFinalStates == trueFinalStates), axis=0)
        denom[denom == 0] = 1
        errorFailureSteps = np.sum(np.abs(predFailureSteps - trueFailureSteps)\
            *(predFinalStates == trueFinalStates)*(1-trueFinalStates)*initialActiveEdges,axis=0)/denom
        
        # plot results for only those lines that have failed and predicted correctly at least once
        validResultLines = (np.sum((1-trueFinalStates)*initialActiveEdges*(predFinalStates == trueFinalStates), axis=0) != 0)
        lffFailureStep = lffFinalState[validResultLines]
        errorFailureSteps = errorFailureSteps[validResultLines]

        # group branches into bins
        lffFinalState, errorFinalStates = self.groupBranchesIntoBins(lffFinalState, errorFinalStates, numBins)
        lffFailureStep, errorFailureSteps = self.groupBranchesIntoBins(lffFailureStep, errorFailureSteps, numBins)
        return lffFailureStep, errorFailureSteps, lffFinalState, errorFinalStates
    
    def getTimingMetrics(self, isTrain = 'Test'):
        return cascadewrapper.cascadeAnalysis(self.device, self.pfModel, self.dataParams).collectAllProfileTiming(isTrain)

    def groupBranchesIntoBins(self, xVals, yVals, numBins):
        xValsRounded = np.round(numBins*xVals)/numBins
        xValsBins = np.unique(xValsRounded)

        yValsAvg = np.zeros(xValsBins.shape)
        for kk, xx in enumerate(xValsBins):
            if(np.sum(xValsRounded == xx) < 2): yValsAvg[kk] = np.nan
            else: yValsAvg[kk] = np.sum(yVals[xValsRounded == xx])/np.sum(xValsRounded == xx)
        return xValsBins, yValsAvg

    def groupScalingIntoBins(self, xVals, yVals, numBins):
        sortedIndices = np.argsort(xVals)
        N = int(yVals.shape[0]/numBins)
        averagePlotX = []
        averagePlotY = np.zeros([numBins, N])
        for ii in range(numBins):
            myIndices = sortedIndices[ii*N: (ii+1)*N]
            averagePlotX.append(np.mean(xVals[myIndices]))
            averagePlotY[ii,:] = yVals[myIndices]
        return np.array(averagePlotX), averagePlotY
        
    def getGNNModel(self, device, dataParamsRandom):
        nodeScale = torch.std(torch.abs(dataParamsRandom.nodeDataTrain[:,:,0] - dataParamsRandom.nodeDataTrain[:,:,1]))
        numNodeFeatures = 1
        hiddenLayerDepth = 2
        if(dataParamsRandom.ieeecase == 'IEEE118'):
            numHiddenFeatures = 150
            numAvgLayers = 15
            attentionLayerDepth = 30
            attentionLayerWidth = 2048
        else:
            numHiddenFeatures = 200
            numAvgLayers = 20
            attentionLayerDepth = 10
            attentionLayerWidth = 2000

        pfModel = nnmodel.GCN(numNodeFeatures, numHiddenFeatures, numAvgLayers, \
                            attentionLayerDepth, attentionLayerWidth, hiddenLayerDepth, dataParamsRandom.numOutputLabels, \
                            dataParamsRandom.node2edgeAdjMatrix, dataParamsRandom.edge2edgeAdjMatrix, nodeScale, 0, False, device).to(device)

        saveWeightsLocation_checkpoint = dataParamsRandom.dataDirectoryHome + '/weights'
        saveWeightsLocation_checkpoint = saveWeightsLocation_checkpoint + '/checkpoint_hiddim' + str(numHiddenFeatures) + '_'
        saveWeightsLocation_checkpoint = saveWeightsLocation_checkpoint + '2023-06-30-2200'
        pfModel.load_state_dict(torch.load(saveWeightsLocation_checkpoint))
        # print('Loaded weights from ' + saveWeightsLocation_checkpoint)

        return pfModel
