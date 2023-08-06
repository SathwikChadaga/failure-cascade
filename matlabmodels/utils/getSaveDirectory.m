function [dataDirectory, linksetDataDirectory] = getSaveDirectory(params, loadScale)

% create save directories
linksetDataDirectory = [params.dataPath '/' params.flowType];
linksetDataDirectory = [linksetDataDirectory '/' params.caseName];
dataDirectory = [linksetDataDirectory '/load' num2str(loadScale,'%.2f') 'Gen' num2str(loadScale,'%.2f')];
dataDirectory = [dataDirectory '/samples' num2str(params.numSamples)];
if ~exist(dataDirectory, 'dir')
    mkdir(dataDirectory)
end
