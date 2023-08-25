# failure-cascade
Power Failure Cascade Prediction using Machine Techniques. The code is used to generate results in the following papers, please refer to them for a detailed explanation of the underlying models.
\[1\] X. Wu, D. Wu and E. Modiano, "Predicting Failure Cascades in Large Scale Power Systems via the Influence Model Framework," in IEEE Transactions on Power Systems, vol. 36, no. 5, pp. 4778-4790, Sept. 2021, doi: 10.1109/TPWRS.2021.3068409.

Files in different folders perform different tasks as summarized in the following list. The list is also ordered in the same order as the files are supposed to be run.

1. `./matlabmodels` folder contains MATLAB models that generate data to train and test rest of the models. This folder also contains files to train the benchamark Influence model. To know more about these models, refer to \[1\].
2. `./gnnmodel` folder contains files to train the GNN model. It also shows some preliminary results for the GNN model.
3. `./mlmodels` folder contains files to train the ML models. It also shows some preliminary results for the ML models.
4. `./influencemodel` folder contains  classes to test the Influence model in Python.
5. `./results` folder contains files to generate detailed results for all the above models.

## `./matlabmodels` folder
Before running these files, MATPOWER toolbox (https://matpower.org/) needs to be installed. And its files should be placed in the `./matlabmodels/utils` folder. 

After setting up the MATPOWER toolbox, the following files within the `./matlabmodels` folder can be used for the following purposes:
1. `./matlabmodels/getParams.m` This file contains all the simulation parameters. Edit this to set the required simulation parameters.
2. `./matlabmodels/generateData.m` This file uses the parameters defined in `./matlabmodels/getParams.m` and gereates cascade failure sequence samples.
3. `./matlabmodels/calculateIMParams.m` This file trains an Influence model for the given parameters in `./matlabmodels/getParams.m` file and saves the weights.
4. `./matlabmodels/convert-data-to-numpy.ipnyb` This notebook converts all the saved data from .mat to .npy; this file saves the data in .npy format in `./gnnmodel/data` folder. It is necessary to run this file before running other models implemented in Python.

The `./matlabmodels/utils` folder contains utility files necessary to run the data generation and learn the influence model weights.

## `./gnnmodel` folder
Before running these files, cascade sequence data needs to generated and converted to .npy format and saved in the `./gnnmodel/data` folder. Data generation can be done using `./matlabmodels/generateData.m` file. Conversion to .npy format can be done using `./matlabmodels/convert-data-to-numpy.ipnyb` file; this file saves the cascade data in .npy format in `./gnnmodels/data` folder. The files in this folder use the Pytorch library (https://pytorch.org/). The files in this folder also require wandb (https://wandb.ai/site) to perform hyperparameter sweep and logging the progress.

Once the data is generated properly, the files of this folder can be used for the following purposes:
1. `./gnnmodel/gnn-cascade-89-random.ipnyb` This notebook trains a GNN model for the IEEE89 dataset. If save is enabled, it saves the weights in the `./gnnmodel/weights` folder.
2. `./gnnmodel/gnn-cascade-118-random.ipnyb` This notebook trains a GNN model for the IEEE118 dataset. If save is enabled, it saves the weights in the `./gnnmodel/weights` folder.
3. `./gnnmodel/gnn-cascade-89-random-sweep.ipnyb` This notebook performs a hyperparameter sweep on the GNN model for the IEEE89 dataset. 
4. `./gnnmodel/gnn-cascade-118-random.ipnyb` This notebook performs a hyperparameter sweep on the GNN model for the IEEE118 dataset. 

The `./gnnmodel/utils` folder contains utility files necessary to train and test the GNN models.

## `./mlmodels` folder
Before running these files, cascade sequence data needs to generated and converted to .npy format and saved in the `./gnnmodel/data` folder. The files in this folder use the scikit-learn library (https://scikit-learn.org/stable/). The files in this folder also require wandb (https://wandb.ai/site) to perform hyperparameter sweep.

Once the data is generated properly, the files of this folder can be used for the following purposes:
1. `./mlmodels/top-ml.ipnyb` This notebook trains ML models. If save is enabled, it saves the weights in the `./mlmodels/models` folder.
2. `./mlmodels/top-ml-sweep.ipnyb` This notebook performs a hyperparameter sweep on the ML models.

The `./mlmodels/utils` folder contains utility files necessary to train and test the ML models.

## `./influencemodel` folder
Before running the files in this folder, influence model weights need to be calculated and converted to .npy format. Influence model weights can be learnt using `./matlabmodels/calculateIMParams.m` file. Conversion to .npy format can be done using `./matlabmodels/convert-data-to-numpy.ipnyb` file; this file saves the influence model weights in .npy format in `./influencemodel/data` folder.

The `./influencemodel/influence-model.ipnyb` file in this folder defines classes in Python to run the influence model in Python. These will be useful when generating results in the next step.

The `./influencemodel/utils` folder contains utility files necessary to test the influence model.

## `./results` folder
This folder contains some files to generate results. Before running these files, all the previous codes should have been run properly so that the test data are generated and saved, and the model parameters are trained and saved. The files in this folder use the classes and functions defined in all the other utility files present in `./gnnmodel/utils`, `./mlmodels/utils`, and `./influencemodel/utils`. So if one edits the file structure of these files, these utility files need to be added to the path accordingly. 
