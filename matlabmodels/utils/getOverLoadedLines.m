function [invalidLines, isOverLoaded] = getOverLoadedLines(mpc)
    % compare power and capacity values to get overloaded line indices in the grid
    powerFlowValues = 0.5*(sqrt(mpc.branch(:,14).^2 + mpc.branch(:,15).^2) + sqrt(mpc.branch(:,16).^2+mpc.branch(:,17).^2));
    % powerFlowValues = abs(mpc.branch(:,14));
    capacityValues = mpc.branch(:,6);
    invalidLines = mpc.branch(powerFlowValues > capacityValues, 18);
    
    % see if there exists overloaded lines
    isOverLoaded = (size(invalidLines,1) ~= 0);
end