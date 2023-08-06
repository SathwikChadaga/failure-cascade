# failure-cascade
Power Failure Cascade Prediction using Machine Techniques. The routine is ordered as follows.

1. matlabmodels folder contains MATLAB models that generate data to train and test rest of the models. This folder also contains files to train the benchamark Influence model.
2. gnnmodel folder contains files to train the GNN model.
3. mlmodels folder contains files to train the ML models.
4. influencemodel folder contains  files to test the Influence model in Python.
5. results folder contains files to generate the results.

## matlabmodels folder
Before running these files, MATPOWER toolbox needs to be installed. And its files should be placed in the matlabmodels/utils folder. 

After setting up the MATPOWER toolbox, the following files within the matlabmodels folder can be used for the following purposes:
1. getParams.m This file contains all the simulation parameters. Edit this to set the required simulation parameters.
2. generateData.m This file uses the parameters defined in getParams.m and gereates cascade failure sequence samples.
3. calculateIMParams.m This file trains an Influence model for the given parameters in getParams.m file and saves the weights.
4. convert-data-to-numpy.ipnyb This notebook converts all the saved data from .mat to .npy; this file saves the data in .npy format in gnnmodels/data folder. It is necessary to run this file before running other models implemented in Python.


## gnnmodel
Before running these files, cascade sequence data needs to generated and converted to .npy format and saved in the gnnmodel/data folder. Data generation can be done using matlabmodels/generateData.m file. Conversion to .npy format can be done using matlabmodels/convert-data-to-numpy.ipnyb file; this file saves the cascade data in .npy format in gnnmodels/data folder. The files in this folder also require wandb to perform hyperparameter sweep and logging the progress.

Once the data is generated properly, the files of this folder can be used for the following purposes:
1. gnn-cascade-89-random.ipnyb This notebook trains a GNN model for the IEEE89 dataset. If save is enabled, it saves the weights in the gnnmodel/weights folder.
2. gnn-cascade-118-random.ipnyb This notebook trains a GNN model for the IEEE118 dataset. If save is enabled, it saves the weights in the gnnmodel/weights folder.
3. gnn-cascade-89-random-sweep.ipnyb This notebook performs a hyperparameter sweep on the GNN model for the IEEE89 dataset. 
4. gnn-cascade-118-random.ipnyb This notebook performs a hyperparameter sweep on the GNN model for the IEEE118 dataset.

## mlmodels folder
Before running these files, cascade sequence data needs to generated and converted to .npy format and saved in the gnnmodel/data folder. The files in this folder also require wandb to perform hyperparameter sweep.

Once the data is generated properly, the files of this folder can be used for the following purposes:
1. top-ml.ipnyb This notebook trains ML models. If save is enabled, it saves the weights in the mlmodels/models folder.
3. top-ml-sweep.ipnyb This notebook performs a hyperparameter sweep on the ML models.

## influencemodel
Before running the files in this folder, influence model weights need to be calculated and converted to .npy format. Influence model weights can be learnt using matlabmodels/calculateIMParams.m file. Conversion to .npy format can be done using matlabmodels/convert-data-to-numpy.ipnyb file; this file saves the influence model weights in .npy format in influencemodel/data folder.

The influence-model.ipnyb file in this folder defines classes in Python to run the influence model in Python. These will be useful when generating results in the next step.

## results
This folder contains some files to generate results. Before running these files, all the previous codes should have been run properly so that the test data are generated and saved, and the model parameters are trained and saved. 
