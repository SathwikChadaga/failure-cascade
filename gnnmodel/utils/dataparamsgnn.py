# Load cascade sequence data and some other relevant data from file.
# - The data can either be used for training or testing. 
# - When defining the class, you need to specify the scaling list
# - This list will be helpful when we loop through different load profiles when genearting results.
# The class has various functions, see comments above functions for more details.

import numpy as np
import torch
from random import sample

class dataParams():
    def __init__(self, device, ieeecase, scalingList, forTrain, customNumOutputLabels = None, dataDirectoryHome = '.', verbose = True):
        super(dataParams, self).__init__()
        # save parameter values
        self.device            = device
        self.ieeecase          = ieeecase
        self.scalingList       = scalingList
        self.dataDirectoryHome = dataDirectoryHome
        self.verbose           = verbose

        # set number of nodes and edges
        self.N_nodes = {'IEEE89':  89, 'IEEE118': 118, 'IEEE1354': 1354}[ieeecase]
        self.M_lines = {'IEEE89': 206, 'IEEE118': 179, 'IEEE1354': 1710}[ieeecase]

        # read and collect training data for all given scaling values
        nodeDataTrain, edgeDataTrain, trainableSamples, numOutputLabels = self.readAllTrainData(scalingList)
        
        # save the train data only if this dataset is needed for training, else just use it to get number of output labels
        if(forTrain == 'Train'):
            self.nodeDataTrain    = nodeDataTrain
            self.edgeDataTrain    = edgeDataTrain
            self.trainableSamples = trainableSamples
            if(self.verbose):
                print('Node features training dataset size = ' + str(nodeDataTrain.shape) + '.')
                print('Edge features training dataset size = ' + str(edgeDataTrain.shape) + '.')
                print('Trainable data sample indices = ' + str(trainableSamples) + '.')
        if(self.verbose): print('Unique output labels in this dataset  = ' + str(torch.unique(edgeDataTrain[:,:,1])) + '.')
        
        # set number of output labels. If custom value is given, then assert if it is higher than the one on file.
        if(customNumOutputLabels == None):
            self.numOutputLabels = numOutputLabels
        else:
            if(customNumOutputLabels < numOutputLabels): print('Warning: given number-of-labels less than read number-of-labels.')
            self.numOutputLabels = customNumOutputLabels

        # read base graph structure
        self.linkSet  = torch.Tensor(np.load(self.getFileLocation(scalingList[0]) + '/linkSet.npy')).to(device).long()
        
        # create node-to-edge adjacency matrix
        self.node2edgeAdjMatrix = torch.zeros([self.linkSet.shape[0], self.N_nodes]).to(device)
        for ii in range(self.node2edgeAdjMatrix.shape[0]):
            self.node2edgeAdjMatrix[ii, self.linkSet[ii,0]] = 1
            self.node2edgeAdjMatrix[ii, self.linkSet[ii,1]] = -1

        # create edge-to-edge adjacency matrix
        self.edge2edgeAdjMatrix = (2*torch.eye(self.M_lines, self.M_lines).to(device) - self.node2edgeAdjMatrix@self.node2edgeAdjMatrix.T)

    # read and collect training data for all given scaling values
    def readAllTrainData(self, scalingList):
        # initialize a variable to collect the values
        nodeDataTrain = torch.zeros([0, self.N_nodes, 2])
        edgeDataTrain = torch.zeros([0, self.M_lines, 2])
        
        # iterate through all given scaling values
        for scalingValue in scalingList:
            # get file location for the given scaling value
            dataDirectory = self.getFileLocation(scalingValue)

            # read data from the file location. Since we are reading ALL training data, keep that in the cpu to save gpu memory.
            nodeData, edgeData = self.readFromFile(torch.device('cpu'), dataDirectory, 'Train')

            # collect the read data values
            nodeDataTrain = torch.concat((nodeDataTrain, nodeData), dim=0)
            edgeDataTrain = torch.concat((edgeDataTrain, edgeData), dim=0)

            # print progress
            if(self.verbose):
                print('Train data read from file ' + dataDirectory + '.')
                print('Unique output labels on file = ' + str(torch.unique(edgeData[:,:,1])) + '.')
        if(self.verbose): print(' ')

        # number of unique labels in the data (the +2 is to account for 0's and -1's)
        numOutputLabels = torch.max(edgeDataTrain[:,:,1]).long().cpu().numpy() + 2

        # when a branch never fails, the corresponding label is saved as -1. Change these -1's to max values.
        edgeDataTrain = self.resetNonFailedEdgeLabels(edgeDataTrain, numOutputLabels)

        # by default use all samples to train (sometimes customizing this set of trainable samples might be helpful for debugging)
        trainableSamples = np.arange(0, edgeDataTrain.shape[0])

        return nodeDataTrain, edgeDataTrain, trainableSamples, numOutputLabels 
    
    # get file location of the data for given scaling value
    def getFileLocation(self, scalingValue):
        if(scalingValue != 'random' and scalingValue != 'random-nonuniform' and scalingValue != 'non-uniform'): 
            scalingValue = '{:.2f}'.format(scalingValue)
        return self.dataDirectoryHome + '/data/' + self.ieeecase + '/load' + scalingValue + 'gen' + scalingValue
    
    # read data from file
    def readFromFile(self, device, dataDirectory, isTrain):
        if(isTrain.lower() == 'train'):
            edgeData = torch.Tensor(np.load(dataDirectory + '/edgeDataTrain.npy')).to(device)
            nodeData = torch.Tensor(np.load(dataDirectory + '/nodeDataTrain.npy')).to(device)
        else:
            edgeData = torch.Tensor(np.load(dataDirectory + '/edgeDataTest.npy')).to(device)
            nodeData = torch.Tensor(np.load(dataDirectory + '/nodeDataTest.npy')).to(device)            
        return nodeData, edgeData
    
    # when a branch never fails, the corresponding label is saved as -1. Change these -1's to max values.
    def resetNonFailedEdgeLabels(self, edgeData, numOutputLabels):
        tempEdgeData = edgeData[:,:,1]
        tempEdgeData[tempEdgeData == -1] = numOutputLabels - 1
        edgeData[:,:,1] = tempEdgeData
        return edgeData
    
    # return a train batch of given batch size
    def getTrainBatch(self, batchSize = 64):
        # if batch size is larger than the train set, then reduce it
        if(batchSize > self.trainableSamples.shape[0]): batchSize = self.trainableSamples.shape[0]

        # get random batch indices
        batchIndices = np.random.choice(self.trainableSamples, batchSize, replace=False)

        # create input output pairs and return them
        return self.makeNodeEdgeFeaturePair(batchIndices, self.nodeDataTrain, self.edgeDataTrain)

    # return a specfic batch of given scaling value, given batch size, from test or train
    def getSpecificBatch(self, isTrain, scalingValue, batchSize = 'all'):
        # get file location
        dataDirectory = self.getFileLocation(scalingValue)

        # read data samples from the file location
        nodeData, edgeData = self.readFromFile(self.device, dataDirectory, isTrain)

        # change -1's to max label
        edgeData = self.resetNonFailedEdgeLabels(edgeData, self.numOutputLabels)

        # get random batch indices if batch size is specified, else return all samples
        if(batchSize == 'all'): batchIndices = np.arange(nodeData.shape[0])
        else: batchIndices = np.random.choice(np.arange(nodeData.shape[0]), batchSize, replace=False)

        # create input output pairs and return them
        return self.makeNodeEdgeFeaturePair(batchIndices, nodeData, edgeData)
    
    # create input output pairs suitable for the model
    def makeNodeEdgeFeaturePair(self, batchIndices, nodeValues, edgeValues):
        # the input node features
        nodeFeatures = nodeValues[batchIndices, :, :]

        # the input initial contingencies
        activeEdges = edgeValues[batchIndices, :, 0]

        # the output edge labels
        edgeFeatures = edgeValues[batchIndices, :, 1].long()

        return nodeFeatures, edgeFeatures, activeEdges
