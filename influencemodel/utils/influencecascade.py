import torch
import torch.nn as nn
import numpy as np

class failureCascade():
    def __init__(self, device, dataParams, imDirectory = None):
        super(failureCascade, self).__init__()
        self.M = None
        self.b_est = None
        self.epsilonValues = None
        self.initialStates = None
        self.device = device
        self.ieeecase = dataParams.ieeecase
        if(imDirectory == None): imDirectory = './data/'
        self.dataDirectoryHome = imDirectory

    def readWeightsFromFile(self, powerScalingValue):
        dataDirectory = self.dataDirectoryHome + self.ieeecase
        powerScalingValue = '{:.2f}'.format(powerScalingValue)
        dataDirectory = dataDirectory + '/load' + powerScalingValue + 'gen' + powerScalingValue

        A11 = torch.Tensor(np.load(dataDirectory + '/A11.npy')).to(self.device)
        A01 = torch.Tensor(np.load(dataDirectory + '/A01.npy')).to(self.device)
        D = torch.Tensor(np.load(dataDirectory + '/D.npy')).to(self.device)
        epsilonValues = torch.Tensor(np.load(dataDirectory + '/epsilonVals.npy')).to(self.device)
        initialStates = torch.Tensor(np.load(dataDirectory + '/initialStates.npy')).to(self.device)

        return A11, A01, D, epsilonValues, initialStates
    
    def updateWeights(self, powerScalingValue):
        A11, A01, D, epsilonValues, initialStates = self.readWeightsFromFile(powerScalingValue)

        M_lines = D.shape[0]
        self.M = D*((A11-A01).T)
        self.b_est = torch.zeros(M_lines,).to(self.device)
        for ii in range(M_lines):
            self.b_est[ii] = D[ii,:]@A01[:,ii]

        self.epsilonValues = epsilonValues
        self.initialStates = initialStates

    def oneStepPowerCascade(self, activeEdgesHard, activeEdgesSoft, epsilonThreshold):
        newActiveEdgesSoft = self.M@activeEdgesSoft + self.b_est

        newActiveEdgesHard = torch.zeros(newActiveEdgesSoft.shape).to(self.device)
        newActiveEdgesHard[newActiveEdgesSoft > epsilonThreshold] = 1
        newActiveEdgesHard[newActiveEdgesSoft < epsilonThreshold] = 0
        newActiveEdgesHard[activeEdgesHard == 0] = 0

        wasOverLoaded = (torch.sum(activeEdgesHard != newActiveEdgesHard) != 0)
    
        return newActiveEdgesHard, newActiveEdgesSoft, wasOverLoaded

    def getEpsilonThreshold(self, intialActiveEdges):
        tempVec = intialActiveEdges@self.initialStates
        bestEpsilonInd = torch.argwhere(tempVec >= torch.max(tempVec))
        epsilonThreshold = torch.median(self.epsilonValues[:,bestEpsilonInd.flatten()], axis=1).values
        return epsilonThreshold
        
    def runCascadeSequence(self, initialActiveEdges):
        initialActiveEdges = initialActiveEdges[0,:]
        epsilonThreshold = self.getEpsilonThreshold(initialActiveEdges)

        currentActiveEdgesSoft = initialActiveEdges
        currentActiveEdgesHard = initialActiveEdges
        isOverLoaded = True
        while(isOverLoaded):
            newActiveEdgesHard, newActiveEdgesSoft, isOverLoaded = self.oneStepPowerCascade(currentActiveEdgesHard, currentActiveEdgesSoft, epsilonThreshold)

            currentActiveEdgesSoft = newActiveEdgesSoft
            currentActiveEdgesHard = newActiveEdgesHard
        
        return currentActiveEdgesHard
    
    def runCascadeSequenceWithFailureSteps(self, initialActiveEdges):
        initialActiveEdges = initialActiveEdges[0,:]
        epsilonThreshold = self.getEpsilonThreshold(initialActiveEdges)

        currentActiveEdgesSoft = initialActiveEdges
        currentActiveEdgesHard = initialActiveEdges
        failureSteps = initialActiveEdges
        isOverLoaded = True
        while(isOverLoaded):
            newActiveEdgesHard, newActiveEdgesSoft, isOverLoaded = self.oneStepPowerCascade(currentActiveEdgesHard, currentActiveEdgesSoft, epsilonThreshold)

            currentActiveEdgesSoft = newActiveEdgesSoft
            currentActiveEdgesHard = newActiveEdgesHard
            failureSteps = failureSteps + currentActiveEdgesHard
        
        failureSteps[currentActiveEdgesHard == 1] = -1
        
        return currentActiveEdgesHard, failureSteps