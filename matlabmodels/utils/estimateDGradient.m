% Gradient function for estimation of D
function gradient = estimateDGradient(coeff, state_tPlus1Prod, state_tProd, numSamples)

gradient = 2*(state_tProd*coeff - state_tPlus1Prod);
gradient = gradient/numSamples;

end