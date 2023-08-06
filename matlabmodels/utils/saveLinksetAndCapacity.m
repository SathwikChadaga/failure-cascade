function saveLinksetAndCapacity(mpc, linksetDataDirectory, loadScale, caseName)
linkSet = mpc.branch(:,1:2);
capacityValues = mpc.branch(:,6);
saveLocation = [linksetDataDirectory '/linksetAndCapacity' ...
    caseName 'load' num2str(loadScale,'%.2f') 'gen' num2str(loadScale,'%.2f') '.mat'];
save(saveLocation, 'linkSet', 'capacityValues');
end