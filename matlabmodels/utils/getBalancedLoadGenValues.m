function [loadVals, genVals] = getBalancedLoadGenValues(loadVals, genVals)

% if load is higher, shed some load
if sum(loadVals) > sum(genVals)
    if sum(genVals) > 0
        % if total generation is non-zero, perform load shedding
        loadVals = loadVals*sum(genVals)/sum(loadVals);
    else
        % if total generation is zero, then set both generation and
        % loading values to zero
        genVals = 0;
        loadVals = 0;
    end
    % if generation is higher, shed some generation
else
    if sum(loadVals) > 0
        % if total load is non-zero, perform generation shedding
        genVals = genVals*sum(loadVals)/sum(genVals);
    else
        % if total load is zero, then set both generation and load
        % values to zero
        genVals = 0;
        loadVals = 0;
    end
end

end