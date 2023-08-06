% Function to estimate the Pairwise Influence Matrices A11 and A01
% Sathwik Chadaga, Oct 25, 2021
% Edited from Xinyu's code
%
% Refer Xinyu's paper for expressions
% Vectorized the code by using the failure time values
% Faster compared to Xinyu's code  

function [A11, A01] = estimateA(cascade)
fprintf(['Started estimating A11 and A01 at ', datestr(now), '\n']);

numNodes = size(cascade,1); % number of nodes
A11 = zeros(numNodes); % the 1-1 transition probabilities
A01 = zeros(numNodes); % the 0-1 transition probabilities

% find top 2 max values
temp = cascade;
maxCascade1 = max(temp);
temp(temp == max(temp)) = -inf;
maxCascade2 = max(temp);

tau = cascade; % time steps at which nodes fail
maxIndices = ((tau == max(tau)).*(maxCascade1-maxCascade2 >= 2) == 1);
tau(maxIndices) = tau(maxIndices) - 1;

taudash = cascade;

% get transition probabilities for each node pairs (j,i)
for jj = 1:numNodes
%     if(rem(jj,10) == 0)
%         disp([num2str(jj) '/' num2str(numNodes)])
%     end
    parfor ii = 1:numNodes
        cji11 = sum(max(min(taudash(ii,:)-2, cascade(jj,:)-1),0));
        cji01 = sum(max(taudash(ii,:)-2 - max(min(taudash(ii,:)-2, cascade(jj,:)-1),0),0));
        cj1 = sum(min(tau(ii,:)-1, cascade(jj,:)-1));
        cj0 = sum(tau(ii,:)-1 - min(tau(ii,:)-1, cascade(jj,:)-1));
        
        if(cj1 == 0)
            A11(jj,ii) = 0;
        else
            A11(jj,ii) = cji11/cj1;
        end
        
        if(cj0 == 0)
            A01(jj,ii) = 0;
        else
            A01(jj,ii) = cji01/cj0;
        end
    end
end

fprintf(['Finished estimating A11 and A01 at ', datestr(now), '\n']);
end