function mpc = scaleLoadGenProfile(mpc, loadScale)

scaleLimitLow = mpc.scaleLimitLow;
scaleLimitHigh = mpc.scaleLimitHigh;

loadVals = mpc.bus(:,3);
genVals = mpc.gen(:,2);

if(strcmp(loadScale, 'random'))
    loadScale = scaleLimitLow + (scaleLimitHigh - scaleLimitLow)*(rand());
%     loadScale = randsample([1.00:0.1:2.00], 1);
end

if(strcmp(loadScale, 'random-nonuniform'))
    if(rand() < 0)
        loadScale = scaleLimitLow +(scaleLimitHigh - scaleLimitLow)*(rand());
    else
        loadScale = scaleLimitLow +(scaleLimitHigh - scaleLimitLow)*(rand(size(loadVals)));
    end
end

if(strcmp(loadScale, 'non-uniform'))
    loadScale = mpc.knownScaling;
end

loadVals = loadScale.*loadVals;
genVals  = sum(loadVals)/sum(genVals)*genVals;

mpc.bus(:,3) = loadVals;
mpc.gen(:,2) = genVals;

end