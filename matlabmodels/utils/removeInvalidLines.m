function mpc = removeInvalidLines(mpc, overLoadedLineIndices)
    % remove those branches that correspond to the given overloaded line indices
    for ii = 1:size(overLoadedLineIndices,1)
        mpc.branch(mpc.branch(:,18) == overLoadedLineIndices(ii), :) = [];
    end
end