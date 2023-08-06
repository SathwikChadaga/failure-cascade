import scipy.io
import numpy as np
import torch
from random import sample

class dataParamsML():
    def __init__(self, ieeecase, dataDirectoryHome, scalingList):
        super(dataParamsML, self).__init__()

        # save parameter values
        self.ieeecase = ieeecase
        self.dataDirectoryHome = dataDirectoryHome
        self.scalingList = scalingList

        self.inputsAll = {}
        self.labelsAll = {}
        self.TValuesAll = {}

        self.updateDataMatrices('Train')
        self.updateDataMatrices('Test')

    def getDataValues(self, isTrain, scalingValue):
        inputState = self.inputsAll[isTrain.lower() + '_' + str(scalingValue)]
        edgeLabels = self.labelsAll[isTrain.lower() + '_' + str(scalingValue)]
        return inputState, edgeLabels

    def updateDataMatrices(self, isTrain):
        for scalingValue in self.scalingList:
            inputState, edgeLabels = self.readDataValues(isTrain, scalingValue)

            if(isTrain.lower() == 'train'):
                T = np.max(edgeLabels)+1
                self.TValuesAll[isTrain.lower() + '_' + str(scalingValue)] = T
            else:
                T = self.TValuesAll['train_' + str(scalingValue)]

            edgeLabels[edgeLabels == -1] = T
            self.inputsAll[isTrain.lower() + '_' + str(scalingValue)] = inputState
            self.labelsAll[isTrain.lower() + '_' + str(scalingValue)] = edgeLabels
        
    def makeIOPair(self, batchIndices, edgeValues):
        # failure times of each edge
        edgeLabels = edgeValues[batchIndices, :, 1]
        inputState = (edgeLabels == 0)
        
        return inputState, edgeLabels

    def getFileLocation(self, scalingValue):
        scalingValue = '{:.2f}'.format(scalingValue)
        return self.dataDirectoryHome + '/data/' + self.ieeecase + '/load' + scalingValue + 'gen' + scalingValue
    
    def readDataValues(self, isTrain, scalingValue):
        dataDirectory = self.getFileLocation(scalingValue)
        
        if(isTrain.lower() == 'train'):
            edgeData = (np.load(dataDirectory + '/edgeDataTrain.npy'))
        else:
            edgeData = (np.load(dataDirectory + '/edgeDataTest.npy'))

        batchIndices = np.arange(edgeData.shape[0])
        inputState, edgeLabels = self.makeIOPair(batchIndices, edgeData)

        return inputState, edgeLabels