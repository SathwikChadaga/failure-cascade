function dataValues = appendSaveValues(mpc, dataValues)

    powerInjected_consume = mpc.bus(:,3);
    
    powerInjected_supply  = zeros(size(powerInjected_consume));
    powerInjected_supply(mpc.gen(:,1)) = mpc.gen(:,2);
    
    powerFlowValues = zeros([mpc.numInitialLines, 1]);
    powerFlowValues(mpc.branch(:,18)) = mpc.branch(:,14);
    
    activeLines = zeros([mpc.numInitialLines, 1]);
    activeLines(mpc.branch(:,18)) = 1;
    
    if(isempty(dataValues))
        dataValues.powerConsumeData  = powerInjected_consume;
        dataValues.powerSuppliedData = powerInjected_supply;
        dataValues.powerFlowData     = powerFlowValues;
        dataValues.activeLinesData   = activeLines;
    else
        if(size(powerInjected_consume, 1) ~= size(dataValues.powerConsumeData))
            disp('a')
        end
        dataValues.powerConsumeData  = [dataValues.powerConsumeData powerInjected_consume];
        dataValues.powerSuppliedData = [dataValues.powerSuppliedData powerInjected_supply];
        dataValues.powerFlowData     = [dataValues.powerFlowData powerFlowValues];
        dataValues.activeLinesData   = [dataValues.activeLinesData activeLines];
    end
end
