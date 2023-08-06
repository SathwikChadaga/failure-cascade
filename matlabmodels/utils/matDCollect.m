clear

addpath('../top')

% get param struct
myParams = getParams();
flowType          = myParams.flowType;                % ac or dc
caseName          = myParams.caseName;                % model name
totNumSamples     = myParams.totNumSamples;           % number of samples
dataFileTimeStamp = myParams.targetTimeStampForTrain; % file timestamp
loadScale         = myParams.loadScale;
generateScale     = myParams.generateScale;
numInitFailures   = myParams.numInitFailures;

% for diff load
loadRange = 0.5:0.1:1.5;
allDs = zeros(size(loadRange,2), 1710, 1710);
fileLocations = [];

for ii = 1:size(loadRange,2)
    currLoad = loadRange(ii);
    dataDirectory = [myParams.dataPath '/' flowType];
    dataDirectory = [dataDirectory '/' caseName];
    dataDirectory = [dataDirectory '/load' num2str(currLoad) 'Gen' num2str(currLoad)];
    dataDirectory = [dataDirectory '/initFails' num2str(numInitFailures)];
    dataDirectory = [dataDirectory '/samples' num2str(totNumSamples)];
    
    % if load data exists
    if exist(dataDirectory, 'dir')
        dataDirNow_diffload = dir(dataDirectory);
        latestD = [];
        % for diff data
        for jj = 1:size(dataDirNow_diffload,1)
            % skip if not a data folder
            if(~strcmp(dataDirNow_diffload(jj).name,'.') && ...
                    ~strcmp(dataDirNow_diffload(jj).name,'..') && ...
                    ~strcmp(dataDirNow_diffload(jj).name,'.DS_Store'))
                
                dataDirNow = [dataDirectory '/' dataDirNow_diffload(jj).name];
                
                % consider if D exists
                if(exist([dataDirNow '/D.mat']))
                    latestD = dataDirNow;
                end
            end
        end
        % read the latest D
        disp(['Reading D from ' latestD])
        load([latestD '/D.mat'], 'D');
        allDs(ii, :,:) = D;
        fileLocations = [fileLocations string(latestD)];
    end
end

% plots
for ii = 2:size(allDs,1)-1
    D = zeros(1710,1710);
    D = allDs(ii,:,:);
    D = permute(D, [2 3 1]);
    subplot(3,3,ii-1);
    spy(D>0.05)
    title(['D matrix (load/generation scaling = ' num2str(loadRange(ii)) ')'])
end

save('../plots/DversusLoad.mat',  'loadRange', 'allDs', 'fileLocations')
