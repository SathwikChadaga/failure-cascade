function [mpcResult, success] = solveIslandPowerFlow(mpcIsland, flowType)
% if more than one node, run the PF model
if size(mpcIsland.bus,1)~=0 &&  size(mpcIsland.gen,1)~=0 && size(mpcIsland.branch,1)~=0
    % run the PF model
    if(strcmp(flowType, 'dc'))
        [mpcResult, success] = rundcpf_me(mpcIsland);
    elseif(strcmp(flowType, 'ac'))
        [mpcResult, success] = runacpf_me(mpcIsland);
    end
else
    % as it is a single node, switch it off?
    mpcResult = mpcIsland;
    mpcResult.gen(:,2) = 0;
    mpcResult.bus(:,3) = 0;
    mpcResult.branch(:,14) = 0;
    success = 1;
end
end