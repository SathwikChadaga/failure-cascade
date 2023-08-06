function [result, success] = shedLoadAndSolvePF(mpc, flowType)

% make a graph from mpc and get components of disconnected subgraphs
[islandLabels, numIslands] = getIslandLabels(mpc);

% perform load shedding if the graph was broken into more than one parts in
% the previous time step
result = mpc;
if numIslands > 1
    % intializing a temporary variable to hold the results later
    mpcTempResult = [];
    
    % enquire load shedding in each subgraph one by one
    for currentIslandLabel = 1:numIslands
        % if(islandSizes(currentIslandLabel) <= 1); continue; end
        
        % for this subgraph component, get indices, loading, and generation
        island = getIsland(mpc, islandLabels, currentIslandLabel);
        
        % perform load or generation shedding for this island
        mpc = performLoadGenShedding(mpc, island);
        
        % get the model corresponding to current component's indices
        mpcCurrentIsland = getMpcIsland(mpc, island);
        
        % solve power flow values
        [mpcCurrentIslandResult, success] = solveIslandPowerFlow(mpcCurrentIsland, flowType);
        if(success ~= 1); return; end
        
        % add current results to the main temporary results
        mpcTempResult = appendMpcFromIslands(mpcTempResult, mpcCurrentIslandResult);

    end
    
    % After all components are done, resort the results
    result.bus = sortrows(mpcTempResult.bus, 1);
    result.gen = sortrows(mpcTempResult.gen, 1);
    result.branch = sortrows(mpcTempResult.branch, 18);
    
else
    % if number of components is not more than 1, then directly apply PF
    % equations and return the results
    if(strcmp(flowType, 'dc'))
        [result, success] = rundcpf_me(mpc);
    elseif(strcmp(flowType, 'ac'))
        [result, success] = runacpf_me(mpc);
    end
    result.branch = sortrows(result.branch, 18);
    if(success ~= 1); return; end
end

% handling a corner case where some PF values are nan
nanPfIndices = isnan(result.branch(:,14));
result.branch(nanPfIndices, 14) = 0;
result.branch(nanPfIndices, 16) = 0;

end