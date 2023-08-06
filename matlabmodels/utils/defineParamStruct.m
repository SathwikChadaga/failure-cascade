function params = defineParamStruct(isDataGen, dataPath, ...
    flowType, caseName, numSamples, numInitialFailures, ...
    loadScaleList, enableSaveData, matpowerFolder)

% pack all simulation parameters to params struct
params.dataPath = dataPath;
params.flowType = flowType;
params.caseName = caseName;
params.numSamples = numSamples;
params.numInitialFailures = numInitialFailures;
params.loadScaleList = loadScaleList;
params.enableSaveData = enableSaveData;
if(isDataGen)

    matpowerStartup(matpowerFolder);  
    if(strcmp(caseName, 'IEEE30'))
        params.caseMatpowerStuct = case30;
        params.caseMatpowerStuct.missingCapacityValues = 9900;
    elseif(strcmp(caseName, 'IEEE39'))
        params.caseMatpowerStuct = case39;
        params.caseMatpowerStuct.missingCapacityValues = 9900;
    elseif(strcmp(caseName, 'IEEE57'))
        params.caseMatpowerStuct = case57;
        params.caseMatpowerStuct.missingCapacityValues = 9900;
    elseif(strcmp(caseName, 'IEEE89'))
        params.caseMatpowerStuct = case89pegase;
        params.caseMatpowerStuct.missingCapacityValues = 9900;
        params.caseMatpowerStuct.scaleLimitLow = 0.95;
        params.caseMatpowerStuct.scaleLimitHigh = 2.05;
    elseif(strcmp(caseName, 'IEEE118'))
        params.caseMatpowerStuct = case118;
        params.caseMatpowerStuct.missingCapacityValues = 475;
        params.caseMatpowerStuct.scaleLimitLow = 0.95;
        params.caseMatpowerStuct.scaleLimitHigh = 2.05;
    elseif(strcmp(caseName, 'IEEE141'))
        params.caseMatpowerStuct = case141;
        params.caseMatpowerStuct.missingCapacityValues = 9900;
    elseif(strcmp(caseName, 'IEEE1354'))
        params.caseMatpowerStuct = case1354pegase;
        params.caseMatpowerStuct.missingCapacityValues = 9900;
        params.caseMatpowerStuct.scaleLimitLow = 0.95;
        params.caseMatpowerStuct.scaleLimitHigh = 2.05;
    end
end

% scaleLimitLow = 1.32; %params.caseMatpowerStuct.scaleLimitLow;
% scaleLimitHigh = params.caseMatpowerStuct.scaleLimitHigh;
% params.caseMatpowerStuct.knownScaling = scaleLimitLow +(scaleLimitHigh - scaleLimitLow)*(rand(size(params.caseMatpowerStuct.bus(:,3))));
end