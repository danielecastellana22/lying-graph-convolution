# Lying Graph Convolution
This is the official repo of the paper:

*D. Castellana, "Lying Graph Convolution: a Network for Node Classification Inspired by Opinion Dynamics", 2024 International Joint Conference on Neural Networks (IJCNN), Yokohama, Japana, 2024*

To execute an epxeriment:
1) Install all the libraries in `requirements.txt`;
2) Execute the command `python run.py --exp-config-file EXP_CONFIG --dataset-config-file DATASET_CONFIG` (check the file `run.py` to see other otional arguments);
   - `EXP_CONFIG` is the configuration file to create the model and to perform the model selection (see the folder `configs\exp`)
   - `DATASET_CONFIG` is the configuration file to load the dataset (see the folder `configs\datasets`)
