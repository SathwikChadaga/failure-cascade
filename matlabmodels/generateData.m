clear

% read simulation parameters
params         = getParams(true);
flowType       = params.flowType;
mpc            = params.caseMatpowerStuct;
numSamples     = params.numSamples;
enableSaveData = params.enableSaveData;
loadScaleList  = params.loadScaleList;
 
% prepare the MPC struct by doing some modifications to it to fit our usage
[mpc, capacityVector, adjacencyMatrix] = prepareGridStruct(mpc, flowType);

% get random initial failure contingencies
initialFailureIndicesAll = getInitialFailures(numSamples, mpc.numInitialLines, params.numInitialFailures);

totalTime_seconds = 0;
totalRuns = 0;
for loadScale = loadScaleList    
    disp(['Generating data for scaling: ' num2str(loadScale)])
    % get save location for this scaling value
    [dataDirectory, linksetDataDirectory] = getSaveDirectory(params, loadScale);

    % save original edge connection information and capacity values
    saveLinksetAndCapacity(mpc, linksetDataDirectory, loadScale, params.caseName);

    % run cascade simulations
    numUnsuccessfulRuns = 0;
    parfor jj = 0:numSamples-1
        % scale loading and generation values
        mpc_scaled = scaleLoadGenProfile(mpc, loadScale);

        % get random nodes that fail initially
        initialFailureIndices = initialFailureIndicesAll(:,jj+1);

        % run power failure cascade process
        kk = jj;
        saveLocation = [dataDirectory '/data_sample' num2str(kk) '.mat'];
        startTime =  datestr(now);
        success = powerCascade(mpc_scaled, initialFailureIndices, saveLocation, enableSaveData, flowType); 
        endTime =  datestr(now);
        totalTime_seconds = totalTime_seconds + etime(datevec(datenum(endTime)), datevec(datenum(startTime)));
        totalRuns = totalRuns + 1;

        % check if successful and display progress
        if(success ~= 1); numUnsuccessfulRuns = numUnsuccessfulRuns + 1; continue; end
        % disp(['Progress = ' num2str(jj+1) '/' num2str(numSamples)])
    end
    
end
disp(params)
disp(['Done - Saved ' num2str(numSamples-numUnsuccessfulRuns) ' cascade samples successfully. ' num2str(numUnsuccessfulRuns) ' runs were skipped.'])
disp(1000*totalTime_seconds/totalRuns)
