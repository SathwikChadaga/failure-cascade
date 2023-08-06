# Class to train the model

import numpy as np
import torch

class nnTrainer():
    def __init__(self, device, criterion, scheduler, isBFGS = False):
        super(nnTrainer, self).__init__()

        self.device         = device
        self.criterion      = criterion
        self.scheduler      = scheduler
        self.isBFGS         = isBFGS

    def customLossFunction(self, predEdgeFeatures, edgeFeatures, activeEdges):
        predEdgeFeatures = torch.transpose(predEdgeFeatures, 1,2)
        allLoss = self.criterion(predEdgeFeatures, edgeFeatures)
        # flowRegularizationLoss = 0.1*self.criterion(self.edge2NodeAdjMatrix@allFlows, nodeFeatures.flatten())
        return torch.mean(allLoss[activeEdges == 1])
    
    def batchStep(self, nodeFeatures, edgeFeatures, activeEdges, pfModel, optimizer):
        nodeFeatures = nodeFeatures.to(self.device)
        edgeFeatures = edgeFeatures.to(self.device)
        activeEdges = activeEdges.to(self.device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward 
        predEdgeFeatures = pfModel(nodeFeatures, activeEdges)

        # backward
        loss = self.customLossFunction(predEdgeFeatures, edgeFeatures, activeEdges)
        loss.backward()

        # optimize
        optimizer.step()
        
        # step size update
        self.scheduler.step()
        
        return loss.detach().item()



