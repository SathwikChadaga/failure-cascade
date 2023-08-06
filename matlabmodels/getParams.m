function params = getParams(isDataGen)
% define simulation parameters here
dataPath           = './data';
utilsPath          = './utils';
matpowerFolder     = './utils\matpower7.1\matpower7.1';
flowType           = 'dc';
caseName           = 'IEEE118';
numSamples         = 10000;
numInitialFailures = 2;
% loadScaleList      = ["random"];
loadScaleList      = [1.00:0.1:2.00]; 
enableSaveData     = 1;

% pack all parameters into a struct
addpath(utilsPath);
params = defineParamStruct(isDataGen, dataPath, ...
    flowType, caseName, numSamples, numInitialFailures, ...
    loadScaleList, enableSaveData, matpowerFolder);
end