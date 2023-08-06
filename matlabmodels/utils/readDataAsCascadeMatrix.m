function cascadeLinks = readDataAsCascadeMatrix(dataDirectory, numSamples)

cascadeLinks = [];
for ii = 1:round(0.9*numSamples)
    try 
        load([dataDirectory '/data_sample' num2str(ii-1) '.mat'], 'activeLinesData');
    catch
        disp('File not found')
        continue
    end
    activeLinesData = [activeLinesData activeLinesData(:,end)];
    cascadeLinks = [cascadeLinks sum(activeLinesData, 2)+1];
end

end