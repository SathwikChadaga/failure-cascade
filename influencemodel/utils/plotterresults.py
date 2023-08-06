import numpy as np
import torch
import matplotlib.pyplot as plt

class resultPlotter():
    def __init__(self, scalingList):
        super(resultPlotter, self).__init__()
        # save parameter values
        self.scalingList = scalingList
    
    def plotFinalStateError(self, errorVectorList, ny, nx, predictMode, modelName, lineFailureFrequencyList = None):

        fig = plt.figure(figsize=(4*nx,4*ny))
        gs = fig.add_gridspec(ny,nx, wspace=0)
        axs = gs.subplots(sharey=True, sharex=True)  
        fig.suptitle('Error in prediction of line\'s final ' + predictMode.lower() + ' with ' + modelName)
          
        a = np.arange(0,100)/100
        for ii in range(ny):  
            for jj in range(nx):
                if(predictMode.lower() == 'status'):
                    if(ii*nx+jj>=len(lineFailureFrequencyList)): continue
                    nodeFailFreqRounded = np.round(10*lineFailureFrequencyList[ii*nx+jj])/10
                    nodeFailFreqBins = np.unique(nodeFailFreqRounded)

                    finalPredErrRate = errorVectorList[ii*nx+jj]
                    finalPredErrRateAvg = np.zeros(nodeFailFreqBins.shape)
                    for kk, failFreq in enumerate(nodeFailFreqBins):
                        finalPredErrRateAvg[kk] = np.sum(finalPredErrRate[nodeFailFreqRounded == failFreq])/np.sum(nodeFailFreqRounded == failFreq)

                    axs[ii, jj].plot(lineFailureFrequencyList[ii*nx+jj], 100*errorVectorList[ii*nx+jj], 'o')
                    axs[ii, jj].plot(nodeFailFreqBins, 100*finalPredErrRateAvg, '-.o')

                    axs[ii, jj].plot(a, 200*a*(1-a), '-')
                    axs[ii, jj].plot(a, 100*(0.5-np.abs(0.5-a)), '-')
                    axs[ii, jj].set_xlabel('Failure frequency')
                    axs[ii, jj].set_ylabel('Error (%)')
                else:
                    axs[ii, jj].plot(self.closenessToCapacity, errorVectorList[ii*nx+jj], 'o')
                    axs[ii, jj].set_xlabel('Closeness to capacity')
                    axs[ii, jj].set_ylabel('Absolute error')

                axs[ii, jj].set_title('Scaling = ' + str(self.scalingList[ii*nx+jj]))
                axs[ii, jj].label_outer()
                axs[ii, jj].grid()

        fig.tight_layout()