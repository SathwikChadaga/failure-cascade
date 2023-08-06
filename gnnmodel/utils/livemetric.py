# Class used to send live metrics to wandb so that we can keep track of the training progress

import matplotlib.pyplot as plt
import sys
import numpy as np
import torch
import wandb

class liveMetricCalculator():
  def __init__(self, device, dataParamsRandom, dataParams):
    super(liveMetricCalculator, self).__init__()

    self.dataParams = dataParams
    self.dataParamsRandom = dataParamsRandom
    self.device = device
    self.lossAccum = []

  # get accuracy on train data i.e. randomly scaled cascade samples
  def getTrainAccuracy(self, pfModel, batchSize):
    nodeFeatures, edgeFeatures, activeEdges = self.dataParamsRandom.getTrainBatch(batchSize)
    predEdgeFeatures,_ = pfModel.validatenn(nodeFeatures.to(self.device), activeEdges.to(self.device))
    predEdgeFeatureLabels = torch.argmax(predEdgeFeatures, dim=-1)
    return 100*torch.min(torch.mean((predEdgeFeatureLabels == edgeFeatures.to(self.device)).float(), dim=0)) 

  # get accuracy for test data i.e. cascade data scaled by known scaling values
  def getTestAccuracy(self, pfModel, scalingValue, batchSize):
    nodeFeatures, edgeFeatures, activeEdges = self.dataParams.getSpecificBatch('Train', scalingValue, batchSize)
    predEdgeFeatures,_ = pfModel.validatenn(nodeFeatures.to(self.device), activeEdges.to(self.device))
    predEdgeFeatureLabels = torch.argmax(predEdgeFeatures, dim=-1)
    return 100*torch.min(torch.mean((predEdgeFeatureLabels == edgeFeatures.to(self.device)).float(), dim=0))

  # logging routine to run every batch
  def batchRoutine(self, currLoss, epoch, pfModel, scalingList, batchSize):
    self.lossAccum.append(currLoss)
    sys.stdout.write('\rMemory usage = ' + str(torch.cuda.memory_allocated(0)/1e9) + '/' + str(torch.cuda.memory_reserved(0)/1e9) + '. ')
    sys.stdout.flush()
    if(epoch%50 == 0 or np.isnan(currLoss) or currLoss > 1e4): 
      batchAccuracy = self.getTrainAccuracy(pfModel, batchSize)
      wandb.log({"iteration": epoch})
      wandb.log({"min-accuracy": batchAccuracy})
      wandb.log({"mean-loss": np.mean(self.lossAccum)})
      self.lossAccum = [] 
      for scalingVal in scalingList:
        testAccuracy = self.getTestAccuracy(pfModel, scalingVal, batchSize)
        wandb.log({"min-accuracy-{:0.2f}".format(scalingVal): testAccuracy})

  # loggin routine to run past training completion
  def postRunRoutine(self, pfModel, batchSize):
    batchAccuracy = self.getTrainAccuracy(pfModel, batchSize)
    finalMesage = 'Final running average accuracy is {:0.4f}%.'.format(batchAccuracy)
    wandb.alert(title = 'Accuracy = {:0.2f}% -- Finished Execution'.format(batchAccuracy), text = finalMesage)
    print(finalMesage)


  