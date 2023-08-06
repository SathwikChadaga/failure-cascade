function mpc = performLoadGenShedding(mpc, island)
loadVals = island.loadVals;
genVals = island.genVals;

[loadVals, genVals] = getBalancedLoadGenValues(loadVals, genVals);

mpc.bus(island.allNodeIndices, 3) = loadVals;
mpc.gen(island.genNodeIndices, 2) = genVals;
end