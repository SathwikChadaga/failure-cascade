load('../plots/DversusLoadAzure.mat', 'loadRange', 'allDs', 'fileLocations');

% plots
for ii = 2:size(allDs,1)-1
    D = zeros(1710,1710);
    D = allDs(ii,:,:);
    D = permute(D, [2 3 1]);
    subplot(3,3,ii-1);
    spy(D>0.05)
    title(['D matrix (load/generation scaling = ' num2str(loadRange(ii)) ')'])
end