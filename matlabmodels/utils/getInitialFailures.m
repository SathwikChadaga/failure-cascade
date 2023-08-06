function initialFailureIndicesAll = getInitialFailures(numSamples, numLines, numFailures)
    rng(1); 
    uniquenessTracker = eye(numLines, numLines);
    initialFailureIndicesAll = zeros(numFailures, numSamples);
    for jj = 1:numSamples
        % get random nodes that fail initially
        [initialFailureIndices, uniquenessTracker] ...
            = getNewUniqueInitialFailure(numLines, numFailures, uniquenessTracker);
        initialFailureIndicesAll(:,jj) = initialFailureIndices;
    end
end

function [failureIndices, uniquenessTracker] = getNewUniqueInitialFailure(numLines, numFailures, uniquenessTracker)
    failureIndices = randperm(numLines, numFailures)';
    if(sum(sum(uniquenessTracker)) == size(uniquenessTracker, 1)*size(uniquenessTracker, 2))
        disp('All unique initial failures already simulated.')
        return
    end
    while(uniquenessTracker(failureIndices(1), failureIndices(2)) == 1)
        failureIndices =  randperm(numLines, numFailures)';
    end
    uniquenessTracker(failureIndices(1), failureIndices(2)) = 1;
    uniquenessTracker(failureIndices(2), failureIndices(1)) = 1;

end