% Funtion to estimate the Influence Matrix D
% Sathwik Chadaga, Oct 30, 2021
% Edited from Xinyu's code
%
% Refer Xinyu's paper for expressions
% Parallel for loops used for IPOPT
% Much faster because of the parallel for loops

function D = estimateD(A11, A01, cascade)
fprintf(['Started estimating D at ', datestr(now), '\n']);

numNodes   = size(cascade,1); % number of nodes
numSamples = size(cascade,2); % number of samples

%% Data modification
% get stateSeries and stateSizes
stateSeries = zeros(numNodes,25,numSamples);
stateSizes  = zeros(numSamples,1);
for ii = 1:numSamples
    % get stateSize
    maxInd       = max(cascade(:,ii)); % index of max entry
    tempCascade  = cascade(:,ii);
    tempCascade(cascade(:,ii)==maxInd) = -inf;
    secondMaxInd = max(tempCascade); % index of second max entry
    if maxInd-secondMaxInd >= 2
        currStateSize = secondMaxInd+1;
    else
        currStateSize = secondMaxInd;
    end

    % get stateSeries
    currStateSeries = ones(numNodes,currStateSize);
    for j = 1:currStateSize
        currStateSeries(cascade(:,ii)<=j,j) = 0;
    end
    stateSeries(:,1:currStateSize,ii) = currStateSeries;
    stateSizes(ii,1) = currStateSize;
end

% get stateSeriesSparse
stateSeriesSparse = cell(size(stateSeries,3));
parfor k=1:numSamples
    state_series_matrix  = reshape(stateSeries(:,1:stateSizes(k,1),k),[size(stateSeries,1),stateSizes(k,1)]);
    stateSeriesSparse{k} = state_series_matrix;
end

% get flag values failureExists
failureExists = zeros(numNodes,1);
parfor ii = 1:numSamples
    tempFailure = zeros(numNodes,1);
    tempFailure(cascade(:,ii)<stateSizes(ii,1) & cascade(:,ii)>1,1) = 1;
    failureExists = failureExists + tempFailure;
end
failureExists = (failureExists>0);

%% Estimate D using IPOPT
A_k = [ones(1,numNodes);zeros(1,numNodes)];
b_k = [1;0];

% set diagonal elements of A01 to 0;
A01(eye(size(A01))==1) = 0;

% intialize D matrix
estimatedD = rand(numNodes);
for ii=1:numNodes
    estimatedD(ii,ii) = 0;
    estimatedD(ii,:) = estimatedD(ii,:)/sum(estimatedD(ii,:));
end

% set the IPOPT options
options                             = [];
options.lb                          = zeros(numNodes,1);
options.ub                          = ones(numNodes,1);
options.cl                          = b_k;
options.cu                          = b_k;
options.ipopt.jac_c_constant        = 'yes';
options.ipopt.hessian_constant      = 'yes';
options.ipopt.jac_d_constant        = 'yes';
options.ipopt.hessian_approximation = 'limited-memory';
options.ipopt.mu_strategy           = 'adaptive';
options.ipopt.tol                   = 1e-5; %%%% TODO: try 1e-6 instead of 1e-7; might be faster
options.ipopt.print_level           = 0;

parfor kk = 1:numNodes
%     if(rem(kk,10) == 0)
%         disp([num2str(kk) '/' num2str(numNodes)])
%     end
    % skip if a row of D corresponding to a node never fails
    if failureExists(kk)
        % get values of state(t+1)
        state_tPlus1 = zeros(1, sum(stateSizes)-size(stateSizes,1));
        count_t      = 0;
        for q = 1:numSamples
            state_tPlus1(count_t + (1:(stateSizes(q)-1))) ...
                = stateSeriesSparse{q}(kk,2:stateSizes(q));
            count_t = count_t + stateSizes(q)-1;
        end
        
        % get values of state(t)
        A11_vec = A11(:,kk);
        A01_vec = A01(:,kk);
        state_t = zeros(numNodes, sum(stateSizes)-size(stateSizes,1));
        count_t = 0;
        for q = 1:numSamples
            A1_vec_rep = repmat(A11_vec,[1,(stateSizes(q)-1)]);
            A2_vec_rep = repmat(A01_vec,[1,(stateSizes(q)-1)]);
            state_t(:, count_t + (1:(stateSizes(q)-1))) ...
                = A1_vec_rep.*stateSeriesSparse{q}(:,1:(stateSizes(q)-1)) ...
                    + A2_vec_rep.*(1-stateSeriesSparse{q}(:,1:(stateSizes(q)-1)));
            count_t = count_t + stateSizes(q)-1;
        end
        
        % get state(ti)*state(tj) and state(ti+1)*state(tj+1)
        state_tProd      = state_t*state_t';
        state_tPlus1Prod = state_t*state_tPlus1';
        
        % define functions for IPOPT
        A_tmp       = A_k;
        A_tmp(1,kk) = 0;
        A_tmp(2,kk) = 1;

        funcs                   = [];
        funcs.objective         = @(coeff) estimateDCost(coeff,state_tPlus1,state_t,numSamples);
        funcs.constraints       = @(coeff) (A_tmp*coeff);
        funcs.gradient          = @(coeff) estimateDGradient(coeff,state_tPlus1Prod,state_tProd,numSamples);
        funcs.jacobian          = @(coeff) estimateDJacobian(coeff,A_tmp);
        funcs.jacobianstructure = @() sparse(A_tmp);
      
        % call IPOPT mex file
        xInIpopt = (estimatedD(kk,:))';
        [xOutIpopt, ~] = ipopt_auxdata(xInIpopt,funcs,options);
        estimatedD(kk,:) = xOutIpopt';
    end
end

% handle skipped rows of D corresponding to nodes that never fail, 
% set the row to [0,0,...,1,...,0], with 1 as the diagonal element
estimatedD(~failureExists, :) = 0;
estimatedD = estimatedD + diag(failureExists == 0);

% return estimated D
D = estimatedD;

fprintf(['Finished estimating D at ', datestr(now), '\n']);

end