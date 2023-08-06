function mpcIsland = getMpcIsland(mpc, island)

% get the model corresponding to current component's indices
genNodeIndicesOfIsland = island.genNodeIndices;
allNodeIndicesOfIsland = island.allNodeIndices;
mpcIsland = mpc; % current mpc

% current mpc gen
mpcIsland.gen = mpc.gen(genNodeIndicesOfIsland,:); % current mpc gen
mpcIsland.gencost = mpc.gencost(genNodeIndicesOfIsland,:); % current mpc gencost

% current mpc bus
mpcIsland.bus = mpc.bus(allNodeIndicesOfIsland,:);
if size(find(mpcIsland.bus(:,2) == 3), 1) == 0 ...
        && size(mpcIsland.bus(:,2), 1) > 1
    index_slack_set = find(mpcIsland.bus(:,3) == min(mpcIsland.bus(:,3)));
    index_slack = index_slack_set(1,1);
    mpcIsland.bus(index_slack, 2) = 3;
end

% current mpc branch
branchIndicesOfIsland = [];
for tt = 1:size(mpc.branch,1)
    if size(find(mpcIsland.bus(:,1) == mpc.branch(tt,1)),1) > 0 ...
            && size(find(mpcIsland.bus(:,1) == mpc.branch(tt,2)),1) > 0
        branchIndicesOfIsland = [branchIndicesOfIsland; tt];
    end
end
mpcIsland.branch = mpc.branch(branchIndicesOfIsland,:);
end

