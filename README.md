# Lying Graph Convolution
This is the officila repo of the paper:

*Lying Graph Convolution: Learning to Lie for Node Classification Tasks*


To execute an epxeriment:
1) Install all the libraries in `requirements.txt`;
2) Execute the command `python run.py --exp-config-file EXP_CONFIG --dataset-config-file DATASET_CONFIG` (check the file `run.py` to see other otional arguments);
   - `EXP_CONFIG` is the configuration file to create the model and to perform the model selection (see the folder `configs\exp`)
   - `DATASET_CONFIG` is the configuration file to load the dataset (see the folder `configs\datasets`)
