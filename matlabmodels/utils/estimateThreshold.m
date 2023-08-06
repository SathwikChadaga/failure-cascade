function [epsilonVals, initialStates, finalStates] = estimateThreshold(cascadeLinks, A01, A11, D)
fprintf(['Started estimating threshold values at ', datestr(now), '\n']);

numTrainSamples = size(cascadeLinks,2);

%% Modify matrices - Xinyu's part
% ?
for ii = 1:size(A01,1)
    A01(ii,ii)=0;
end

P1 = A11';
P2 = A01';
numNodes = size(D,1);

% ?? calculate M_est, b_est, s(t+1) = M*s(t)+b
M_est = D.*(P1-P2);
sumTemp1 = zeros(numNodes,1);
M_1=D.*P1;
for ii=1:numNodes
    sumTemp1(ii,1)=sum(M_1(ii,:));
end
sumTemp2 = zeros(numNodes,1);
M_2=D.*P2;
for ii=1:numNodes
    sumTemp2(ii,1)=sum(M_2(ii,:));
end
b_est=zeros(numNodes,1);
for ii = 1:numNodes
    b_est(ii,1)=D(ii,:)*P2(ii,:)';
end

%% Modify cascade data - Xinyu's part
% load([fileBaseName '_Train.mat'], 'cascadeTrain')

% cascadeTrain to state_series_tensor
% cascadeTrain is a 2d matrix, its entry is the time step when the node failed
% state_series_tensor is a binary matrix with 3 dimensions (3rd dimension is the time step)
stateSizes = zeros(numTrainSamples,1);
weight_vector = zeros(numTrainSamples,1);
for ii = 1:numTrainSamples
    max_i=max(cascadeLinks(:,ii));
    tmp_cascadeTrain=cascadeLinks(:,ii);
    tmp_cascadeTrain(cascadeLinks(:,ii)==max_i)=0;
    submax_i=max(tmp_cascadeTrain);
    if max_i-submax_i>=2
        currStateSize = submax_i+1;
    else
        currStateSize = submax_i;
    end
    stateSizes(ii,1) = currStateSize;
    weight_vector(ii,1) = 1;
end

%% Start epsilon estimation
epsilonVals = zeros(numNodes, numTrainSamples);
initialStates = zeros(numNodes, numTrainSamples);
finalStates = zeros(numNodes, numTrainSamples);
Flag_opt=zeros(numNodes,numTrainSamples);

parfor kk = 1:numTrainSamples
    real_currCascade = cascadeLinks(:,kk);
    
    % determine the final state
    real_final_state=zeros(numNodes,1);
    max_real=max(real_currCascade(:,1));
    tmp_real=real_currCascade(:,1);
    tmp_real(real_currCascade(:,1)==max_real)=0;
    submax_real=max(tmp_real);
    if max_real-submax_real>=2
        real_final_state(real_currCascade(:,1)==max_real, 1)=1;
    end
    
    finalStates(:,kk)=real_final_state;
    
    % get intial state
    currState = ones(numNodes,1);
    tmp_rnd=find(real_currCascade==1);
    currState(tmp_rnd)=0;
    initialStates(:,kk) = currState;
    stateCascade = [];
    stateCascade = [stateCascade, currState];
    
    % simulate rest of the cascade sequence
    count=0;
    while count<stateSizes(kk,1)
        count=count+1;
        currState = M_est*currState + b_est;
        stateCascade = [stateCascade, currState];
    end
    
    % determine the threshold for each link ii
    for ii=1:numNodes
        if real_currCascade(ii) > 1 && real_final_state(ii)==0 % if link ii fails in between
            epsilonVals(ii,kk)=0.5*stateCascade(ii,real_currCascade(ii))+0.5*stateCascade(ii,real_currCascade(ii)-1);  %Here?
        end
        if real_currCascade(ii) == 1 % if link ii fails initially
            epsilonVals(ii,kk)=1;
        end
        if real_final_state(ii) == 1 % if link ii never fails
            epsilonVals(ii,kk)=stateCascade(ii,real_currCascade(ii))*0.8;
            Flag_opt(ii,kk)=1;
        end
    end
end

fprintf(['Finished estimating threshold values at ', datestr(now), '\n']);

end