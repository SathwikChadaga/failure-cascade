# The GNN model class
# This class has two main functions - forward and validatenn
# 1. forward - This function runs a forward pass through the model
# 2. validatenn - This funtion runs a forward pass through the model without saving gradients and also returns the prediction. T
#                 This is similar to the evaluate function.
# The opther funcitons are helpers to the above functions.

import torch
import torch.nn as nn

class GCN(torch.nn.Module):
  def __init__(self, numNodeFeatures, numHiddenFeatures, numAvgLayers, attentionLayerDepth, attentionLayerWidth, hiddenLayerDepth, numOutputLabels, \
                node2edgeAdjMatrix, edge2edgeAdjMatrix, nodeScale, nodeBias, detachAttention, device):
    super().__init__()
    
    # save parameters
    self.device = device
    self.node2edgeAdjMatrix = (1/2)*node2edgeAdjMatrix.unsqueeze(dim=0)
    self.edge2edgeAdjMatrix = edge2edgeAdjMatrix
    self.nodeScale = nodeScale
    self.nodeBias = nodeBias
    M_lines = node2edgeAdjMatrix.shape[0]
    self.numOutputLabels = numOutputLabels
    self.defaultOneHotVector = torch.zeros([self.numOutputLabels,]).to(self.device)
    self.defaultOneHotVector[0] = 1
    self.detachAttention = detachAttention

    # define the attention stage of the model
    attentionLayer_outdim_node2edge = torch.sum(node2edgeAdjMatrix != 0)
    attentionLayer_outdim_edge2edge = torch.sum(edge2edgeAdjMatrix != 0)
    self.attentionLayer1 = self.defineAttentionLayers(M_lines, attentionLayer_outdim_edge2edge, attentionLayerDepth, attentionLayerWidth)
    self.attentionLayer2 = self.defineAttentionLayers(M_lines, attentionLayer_outdim_node2edge, attentionLayerDepth, attentionLayerWidth)

    # define the initial stage of the model
    self.initialLayer = nn.Sequential(
      nn.Linear(numNodeFeatures, numHiddenFeatures),
      nn.LeakyReLU(),
      nn.Linear(numHiddenFeatures, numHiddenFeatures),
      nn.LeakyReLU(),
      nn.Linear(numHiddenFeatures, numHiddenFeatures),
    )
    
    # define the averaging stage of the model
    self.numHidden = numAvgLayers
    self.hiddenLayersEdge2Edge = nn.ModuleList([self.defineHiddenLayer(numHiddenFeatures, hiddenLayerDepth) for ii in range(self.numHidden)])
    self.hiddenLayersNode2Edge = nn.ModuleList([self.defineHiddenLayer(numHiddenFeatures, hiddenLayerDepth) for ii in range(self.numHidden)])
   
   # define the final stage of the model
    self.finalLayer = nn.Sequential(
      nn.Linear(numHiddenFeatures, numHiddenFeatures),
      nn.LeakyReLU(),
      nn.Linear(numHiddenFeatures, int(numHiddenFeatures/2)),
      nn.LeakyReLU(),
      nn.Linear(int(numHiddenFeatures/2), self.numOutputLabels),
    )

  def forward(self, nodeFeatures, activeEdges):    
    # get attention coefficients - edge-to-edge
    attentionValues = self.attentionLayer1(activeEdges)
    attention1 = self.edge2edgeAdjMatrix.unsqueeze(dim=0).repeat((activeEdges.shape[0],1,1))
    attention1[attention1 != 0] = attentionValues.flatten()
    
    # get attention coefficients - node-to-edge
    attentionValues = self.attentionLayer2(activeEdges)
    attention2 = self.node2edgeAdjMatrix.repeat((activeEdges.shape[0],1,1))
    attention2[attention2 != 0] = attentionValues.flatten()

    # normalization
    nodeFeatures = (nodeFeatures[:,:,0]-nodeFeatures[:,:,1]).unsqueeze(dim=-1)
    nodeFeatures = (nodeFeatures-self.nodeBias)/self.nodeScale

    # transform node features to a hidden dimension
    nodeFeatures = self.initialLayer(nodeFeatures)   

    # get edge features from neighboring node features
    edgeFeatures = self.node2edgeAdjMatrix@nodeFeatures

    # get edge adjacency
    edge2edgeAdjMatrixBatch = activeEdges.unsqueeze(dim=-1)*self.edge2edgeAdjMatrix*activeEdges.unsqueeze(dim=1)
    edgeDegreeScaling = (1/torch.sqrt(1 + torch.sum(torch.abs(edge2edgeAdjMatrixBatch),dim=-1))).unsqueeze(dim=-1)

    # averaging layer: message passing between neighboring nodes in hidden dimension
    for ii in range(self.numHidden):
      edgeFeatures = self.passThroughHidden(edgeFeatures, nodeFeatures, self.hiddenLayersEdge2Edge[ii], self.hiddenLayersNode2Edge[ii], attention1, attention2, edgeDegreeScaling, edge2edgeAdjMatrixBatch)
      if(self.detachAttention and ii == 20):
        attention1 = attention1.detach()
        attention2 = attention2.detach()

    # transform edge power values back to power dimension
    edgeFeatures = self.finalLayer(edgeFeatures)
    edgeFeatures[activeEdges == 0] = self.defaultOneHotVector

    return edgeFeatures
  
  # forward pass of the model to be used during test phase. Similar to evaluate.
  def validatenn(self, nodeFeatures, activeEdges): 
    with torch.no_grad():
      predvals_soft = self(nodeFeatures, activeEdges).detach()
    return predvals_soft, predvals_soft

  # forward pass through the averaging layer 
  def passThroughHidden(self, edgeFeatures, nodeFeatures, hiddenLayer1, hiddenLayer2, attention1, attention2, edgeDegreeScaling, edge2edgeAdjMatrixBatch):
    edgeFeatures = (edgeDegreeScaling*edgeFeatures*edgeDegreeScaling
                    + hiddenLayer1(edgeDegreeScaling*(edge2edgeAdjMatrixBatch*attention1)@(edgeDegreeScaling*edgeFeatures))
                    + hiddenLayer2((self.node2edgeAdjMatrix*attention2)@nodeFeatures))
    return edgeFeatures
    
  # define the steps of the averaging stage
  def defineHiddenLayer(self, numHiddenFeatures, num_layers):
    modules = []
    for ii in range(num_layers):
      modules.append(nn.Linear(numHiddenFeatures, numHiddenFeatures))
      modules.append(nn.LeakyReLU())
    modules.append(nn.Linear(numHiddenFeatures, numHiddenFeatures))
    return nn.Sequential(*modules)

  # define the attention stage of the model
  def defineAttentionLayers(self, in_dim, out_dim, num_layers, width):
    modules = []
    modules.append(nn.Linear(in_dim, width))
    modules.append(nn.LeakyReLU())
    for ii in range(num_layers):
      modules.append(nn.Linear(width, width))
      modules.append(nn.LeakyReLU())
    modules.append(nn.Linear(width, out_dim))
    return nn.Sequential(*modules)