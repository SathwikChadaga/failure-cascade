import numpy as np
import matplotlib.pyplot as plt

def plotFinalStateError(errorVectorList, ny, nx, titleText, lineFailureFrequencyList, scalingList):

    fig = plt.figure(figsize=(2*nx,2*ny))
    gs = fig.add_gridspec(ny,nx, wspace=0)
    axs = gs.subplots(sharey=True, sharex=True)  
    fig.suptitle(titleText)

    a = np.arange(0,100)/100
    for ii in range(ny):  
        for jj in range(nx):
            if(ii*nx+jj>=lineFailureFrequencyList.shape[0]): continue
            nodeFailFreqRounded = np.round(10*lineFailureFrequencyList[ii*nx+jj, :])/10
            nodeFailFreqBins = np.unique(nodeFailFreqRounded)

            finalPredErrRate = errorVectorList[ii*nx+jj, :]
            finalPredErrRateAvg = np.zeros(nodeFailFreqBins.shape)
            for kk, failFreq in enumerate(nodeFailFreqBins):
                finalPredErrRateAvg[kk] = np.sum(finalPredErrRate[nodeFailFreqRounded == failFreq])/np.sum(nodeFailFreqRounded == failFreq)

            axs[ii, jj].plot(lineFailureFrequencyList[ii*nx+jj, :], 100*errorVectorList[ii*nx+jj, :], '.')
            axs[ii, jj].plot(nodeFailFreqBins, 100*finalPredErrRateAvg, '-.o')

            axs[ii, jj].plot(a, 200*a*(1-a), '-')
            axs[ii, jj].plot(a, 100*(0.5-np.abs(0.5-a)), '-')
            axs[ii, jj].set_xlabel('Failure frequency')
            axs[ii, jj].set_ylabel('Error (%)')
        

            axs[ii, jj].set_title('Scaling = ' + '{:.2f}'.format(scalingList[ii*nx+jj]))
            axs[ii, jj].label_outer()
            axs[ii, jj].grid()

    fig.tight_layout()