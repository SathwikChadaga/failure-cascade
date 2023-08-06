function saveData(saveLocation, dataValues)

powerConsumeData  = dataValues.powerConsumeData;
powerSuppliedData = dataValues.powerSuppliedData;
powerFlowData     = dataValues.powerFlowData;
activeLinesData   = dataValues.activeLinesData;

save(saveLocation, 'powerConsumeData', 'powerSuppliedData', 'powerFlowData', 'activeLinesData')
end