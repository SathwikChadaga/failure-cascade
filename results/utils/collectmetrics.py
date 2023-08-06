from importlib import reload
import gnnresults
reload(gnnresults)
import imresults
reload(imresults)
import mlresults
reload(mlresults)

class metricCollector():
    def __init__(self, device, scalingList, dataDirectoryBase, imDirectoryBase, mlModelDirectory):
        super(metricCollector, self).__init__()
        self.device = device
        self.scalingList = scalingList
        self.dataDirectoryBase = dataDirectoryBase
        self.imDirectoryBase = imDirectoryBase
        self.mlModelDirectory = mlModelDirectory
        self.graphMetrics = {}
        self.branchMetrics = {}

    def getGraphMetricsFromModel(self, modelType, ieeecase):
        match modelType.lower():
            case 'gnn': 
                return gnnresults.gnnResults(self.device, ieeecase, self.scalingList, self.dataDirectoryBase).getGraphMetrics()
            case 'im': 
                return imresults.imResults(self.device, ieeecase, self.scalingList, self.dataDirectoryBase, self.imDirectoryBase).getGraphMetrics()
            case _: 
                return mlresults.mlResults(self.device, [modelType], ieeecase, self.scalingList, self.dataDirectoryBase, self.mlModelDirectory).getGraphMetrics(modelType)
            
    def collectGraphMetric(self, modelType, ieeecase):
        print('Collecting results for model = ' + modelType + ', case = ' + ieeecase + '.')
        if((ieeecase == 'IEEE1354') and(modelType == 'GNN' or modelType == 'IM')): 
            print('Skipped results for model = ' + modelType + ', case = ' + ieeecase + '.')
            print(' ')
            return
        scalingVals, failureSizes, finalStates, failureSteps = self.getGraphMetricsFromModel(modelType, ieeecase)

        if((modelType in self.graphMetrics) == False): self.graphMetrics[modelType] = {}
        self.graphMetrics[modelType][ieeecase] = {}
        self.graphMetrics[modelType][ieeecase]['scaling'] = scalingVals
        self.graphMetrics[modelType][ieeecase]['failure-size'] = failureSizes
        self.graphMetrics[modelType][ieeecase]['final-state'] = finalStates
        self.graphMetrics[modelType][ieeecase]['failure-step'] = failureSteps

        print('Collected results for model = ' + modelType + ', case = ' + ieeecase + '.')
        print(' ')

    def getBranchMetricsFromModel(self, modelType, ieeecase, scalingValue, numBins):
        match modelType.lower():
            case 'gnn': 
                if(scalingValue == 'random'): myScalingList = ['random']
                else: myScalingList = self.scalingList
                return gnnresults.gnnResults(self.device, ieeecase, myScalingList, self.dataDirectoryBase).getBranchMetrics(scalingValue, numBins)
            case 'im': 
                return imresults.imResults(self.device, ieeecase, self.scalingList, self.dataDirectoryBase, self.imDirectoryBase).getBranchMetrics(scalingValue, numBins)
            case _: 
                return mlresults.mlResults(self.device, [modelType], ieeecase, self.scalingList, self.dataDirectoryBase, self.mlModelDirectory).getBranchMetrics(modelType, scalingValue, numBins)

    def collectBranchMetric(self, scalingListToTest, modelType, ieeecase, numBins = 10):
        print('Collecting results for model = ' + modelType + ', case = ' + ieeecase + '.')
        if((ieeecase == 'IEEE1354') and (modelType == 'GNN' or modelType == 'IM')): 
            print('Skipped results for model = ' + modelType + ', case = ' + ieeecase + '.')
            print(' ')
            return
        for scalingValue in scalingListToTest:
            if(scalingValue == 'random'): 
                if(modelType != 'GNN'):
                    print('Skipped test for scaling value = ' + str(scalingValue)  + '.')
                    continue
            else: scalingValue = float(scalingValue)
            failfreqsFailureSteps, failureSteps, failfreqsFinalStates, finalStates = self.getBranchMetricsFromModel(modelType, ieeecase, scalingValue, numBins)

            if((scalingValue in self.branchMetrics) == False): self.branchMetrics[scalingValue] = {}
            if((modelType in self.branchMetrics[scalingValue]) == False): self.branchMetrics[scalingValue][modelType] = {}
            self.branchMetrics[scalingValue][modelType][ieeecase] = {}
            self.branchMetrics[scalingValue][modelType][ieeecase]['failfreqs-failure-step'] = failfreqsFailureSteps
            self.branchMetrics[scalingValue][modelType][ieeecase]['failfreqs-final-state'] = failfreqsFinalStates
            self.branchMetrics[scalingValue][modelType][ieeecase]['final-state'] = finalStates
            self.branchMetrics[scalingValue][modelType][ieeecase]['failure-step'] = failureSteps

        print('Collected results for model = ' + modelType + ', case = ' + ieeecase + '.')
        print(' ')
