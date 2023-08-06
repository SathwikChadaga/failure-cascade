function runningMpc = appendMpcFromIslands(runningMpc, currentMpc)

if(isempty(runningMpc))
    runningMpc.bus    = [];
    runningMpc.gen    = [];
    runningMpc.branch = [];
end

runningMpc.bus    = [runningMpc.bus; currentMpc.bus];
runningMpc.gen    = [runningMpc.gen; currentMpc.gen];
runningMpc.branch = [runningMpc.branch; currentMpc.branch];

end