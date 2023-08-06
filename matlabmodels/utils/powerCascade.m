function success = powerCascade(mpc, initialFailureIndices, saveLocation, enableSaveData, flowType)

isOverLoaded = 1; dataValues = []; currentFailureIndices = initialFailureIndices;
while(isOverLoaded)
    % perform line failures (initial-failure or capacity-failure)
    mpc = removeInvalidLines(mpc, currentFailureIndices);
    
    % load shedding in each island and find power flow values
    [mpc, success] = shedLoadAndSolvePF(mpc, flowType);
    if(success ~= 1); return; end
    
    % record and save power values
    dataValues = appendSaveValues(mpc, dataValues);
    
    % update the failure indices for next time step
    [currentFailureIndices, isOverLoaded] = getOverLoadedLines(mpc);
end

if(enableSaveData == 1)
    saveData(saveLocation, dataValues)
end

end