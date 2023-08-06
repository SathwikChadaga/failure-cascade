# failure-cascade
Power Failure Cascade Prediction using Machine Techniques. The routine is ordered as follows.

1. matlabmodels folder contains MATLAB models that generate data to train and test rest of the models. This folder also contains files to train the benchamark Influence model.
2. mlmodels folder contains files to train the ML models.
3. gnnmodel folder contains files to train the GNN model.
4. influencemodel folder contains  files to test the Influence model in Python.
5. results folder contains files to generate the results.

## matlabmodels folder

1. getParams.m This file contains all the simulation parameters. Edit this to set the required simulation parameters.
2. generateData.m This file uses the parameters defined in getParams.m and gereates cascade failure sequence samples.
3. calculateIMParams.m This file trains an Influence model for the given parameters in getParams.m file and saves the weights.
4. convert-data-to-numpy.ipnyb This notebook converts all the saved data from .mat to .npy. It is necessary to run this file before running other models implemented in Python.

## mlmodels folder

## gnnmodel

## influencemodel

## results

