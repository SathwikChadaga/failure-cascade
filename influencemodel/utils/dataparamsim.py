import scipy.io
import numpy as np
import torch
from random import sample

class dataParams():
    def __init__(self, device, ieeecase, dataDirectoryHome, scalingList):
        super(dataParams, self).__init__()

        # save parameter values
        self.device   = device
        self.ieeecase = ieeecase
        self.dataDirectoryHome = dataDirectoryHome
        self.scalingList = scalingList
        
    def makeNodeEdgeFeaturePair(self, batchIndices, nodeValues, edgeValues):
        # get power generation at each node 
        nodeFeatures = nodeValues[batchIndices, :, :]
        
        # data on which edges are active
        activeEdges = edgeValues[batchIndices, :, 0]

        # failure times of each edge
        edgeFeatures = edgeValues[batchIndices, :, 1].long()
        
        return nodeFeatures, edgeFeatures, activeEdges

    def getFileLocation(self, scalingValue):
        scalingValue = '{:.2f}'.format(scalingValue)
        return self.dataDirectoryHome + '/data/' + self.ieeecase + '/load' + scalingValue + 'gen' + scalingValue
    
    def readDataValues(self, isTrain, scalingValue):
        dataDirectory = self.getFileLocation(scalingValue)
        
        if(isTrain.lower() == 'train'):
            edgeData = torch.Tensor(np.load(dataDirectory + '/edgeDataTrain.npy'))
            nodeData = torch.Tensor(np.load(dataDirectory + '/nodeDataTrain.npy'))
        else:
            edgeData = torch.Tensor(np.load(dataDirectory + '/edgeDataTest.npy'))
            nodeData = torch.Tensor(np.load(dataDirectory + '/nodeDataTest.npy'))

        batchIndices = np.arange(nodeData.shape[0])
        nodeFeatures, edgeFeatures, activeEdges = self.makeNodeEdgeFeaturePair(batchIndices, nodeData.to(self.device), edgeData.to(self.device))

        return nodeFeatures, edgeFeatures, activeEdges