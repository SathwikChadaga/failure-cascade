clear

% read simulation parameters
params = getParams(true);

for loadScale = params.loadScaleList    
    startTime =  datestr(now);
    % get save location for this scaling value
    [dataDirectory, ~] = getSaveDirectory(params, loadScale);

    % read failure data
    cascadeLinks = readDataAsCascadeMatrix(dataDirectory, params.numSamples);
    
%     disp('Load scale and mean length')
%     disp(loadScale)
%     disp(mean(max(cascadeLinks)))
%     continue

    % estimate A11 and A01
    [A11, A01] = estimateA(cascadeLinks);

    % estimate D
    D = estimateD(A11, A01, cascadeLinks);

    % estimate threshold values
    [epsilonVals, initialStates, finalStates] = estimateThreshold(cascadeLinks, A01, A11, D);

    % save estimated weights
    if(params.enableSaveData)
        save([dataDirectory '/influenceModelParams.mat'], 'A11', 'A01', 'D', 'epsilonVals', 'initialStates', 'finalStates')
    end
    endTime =  datestr(now);
end

disp(startTime)
disp(endTime)