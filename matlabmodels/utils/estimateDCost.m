% Objective function for estimation of D
function Y = estimateDCost(coeff, state_tPlus1, state_t, numSamples)

Y = sum((state_tPlus1 - coeff'*state_t).^2);
Y = Y/numSamples;

end