import matplotlib.pyplot as plt
import tikzplotlib

def plotAndSaveResult(xVariable, yMetric, resultMetrics, ieeecaseList, modelTypeList, legendLocation, saveLocation, connectDots = '', scalingValue = 0):
    subPlotDims = 101+10*len(ieeecaseList)
    for index, ieeecase in enumerate(ieeecaseList):
        if(index == 0): ax = plt.subplot(subPlotDims + index)
        else: ax = plt.subplot(subPlotDims + index, sharex=ax)

        for modelType in modelTypeList:
            addPlotToAxis(ax, xVariable, yMetric, resultMetrics, ieeecase, modelType, connectDots, scalingValue)

        xLabel, yLabel = getLabels(xVariable, yMetric)
        if(index == 0): ax.set_ylabel(yLabel)
        ax.set_xlabel(xLabel)
        ax.legend(loc = legendLocation)

    tikzplotlib.save(saveLocation)
    plt.show()

def addPlotToAxis(ax, xVariable, yMetric, resultMetrics, ieeecase, modelType, connectDots, scalingValue):
    xVal = resultMetrics[modelType][ieeecase][xVariable]
    yVal = resultMetrics[modelType][ieeecase][yMetric]
    plotStyle, plotLabel, plotColor, markerSize, lineWidth = getPlotStyles(xVariable, modelType, connectDots, scalingValue)
    ax.plot(xVal, yVal, plotStyle, label = plotLabel, color = plotColor, ms = markerSize, lw = lineWidth)
    if(any(yVal > 25)): ax.set_ylim([0,25])

def getPlotStyles(xVariable, modelType, connectDots, scalingValue):
    modelType = modelType.lower()
    
    # get plot label to go on the legend
    if(scalingValue > 0): scalingText = ' (%.2f)'%scalingValue
    else: scalingText = ''
    if(modelType == 'gnn'): plotLabel = {'scaling': 'GNN (bin average)', 'failfreqs-final-state': 'Generalized GNN', 'failfreqs-failure-step': 'Generalized GNN'}[xVariable]
    elif(modelType == 'im'): plotLabel = {'scaling': 'Influence models', 'failfreqs-final-state': 'Influence model' + scalingText, 'failfreqs-failure-step': 'Influence model' + scalingText}[xVariable]
    elif(modelType == 'bnb'): plotLabel = {'scaling': 'Naive Bayes models', 'failfreqs-final-state': 'Naive Bayes model' + scalingText, 'failfreqs-failure-step': 'Naive Bayes model' + scalingText}[xVariable]
    elif(modelType == 'svm'): plotLabel = {'scaling': 'SVM models', 'failfreqs-final-state': 'SVM model' + scalingText, 'failfreqs-failure-step': 'SVM model' + scalingText}[xVariable]
    elif(modelType == 'logr'): plotLabel = {'scaling': 'Regression models', 'failfreqs-final-state': 'Regression model' + scalingText, 'failfreqs-failure-step': 'Regression model' + scalingText}[xVariable]

    # get the style, color, and size of the markers and lines
    if(connectDots == '' and modelType == 'gnn'): connectDots = '--'
    plotStyle = connectDots + {'gnn': 's', 'im': 'o', 'bnb': 'p', 'svm': '*', 'logr': 'v'}[modelType]
    plotColor = {'gnn': '#0000a7', 'im': '#c1272d', 'bnb': 'm', 'svm': 'c', 'logr': 'g'}[modelType]
    markerSize = {'gnn': 4.5, 'im': 4.5, 'bnb': None, 'svm': None, 'logr': 4.5}[modelType]
    lineWidth = {'gnn': None, 'im': None, 'bnb': None, 'svm': None, 'logr': None}[modelType]

    return plotStyle, plotLabel, plotColor, markerSize, lineWidth

def getLabels(xVariable, yMetric):
    xLabel = {'scaling': 'Load scaling', 'failfreqs-final-state': 'Branch failure frequency', 'failfreqs-failure-step': 'Branch failure frequency'}[xVariable]
    yLabel = {'failure-size': 'Failure size error (%)', 'final-state': 'Final state error (%)', 'failure-step': 'Failure time step error (generations)'}[yMetric]
    return xLabel, yLabel