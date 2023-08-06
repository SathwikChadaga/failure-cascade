function [islandLabels, numIslands] = getIslandLabels(mpc)

currGraph    = digraph(mpc.branch(:,1)', mpc.branch(:,2)');
islandLabels = conncomp(currGraph, 'Type', 'weak');

% corner cace where last node(s) are disconnected
lastConnectedNode = max([mpc.branch(:,1); mpc.branch(:,2)]);
if(lastConnectedNode ~= size(mpc.bus, 1))
    extraLabels = (1:(size(mpc.bus, 1) - size(islandLabels, 2))) + max(islandLabels);
    islandLabels = [islandLabels extraLabels];
end

numIslands   = max(islandLabels);

end