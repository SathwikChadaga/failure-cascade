function island = getIsland(mpc, allIslandLabels, currentIslandLabel)
allNodeIndicesOfIsland = find(allIslandLabels == currentIslandLabel);
loadVals = mpc.bus(allNodeIndicesOfIsland,3);
genVals = [];
genNodeIndicesOfIsland = [];
for jj = 1:size(mpc.gen(:,1),1)
    if ismember(mpc.gen(jj,1), allNodeIndicesOfIsland)
        genVals = [genVals; mpc.gen(jj,2)];
        genNodeIndicesOfIsland = [genNodeIndicesOfIsland; jj];
    end
end

island = [];
island.allNodeIndices = allNodeIndicesOfIsland;
island.loadVals = loadVals;
island.genVals = genVals;
island.genNodeIndices = genNodeIndicesOfIsland;

end