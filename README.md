# COMS6200 Project

This repo is for Sem 2, 2022 group project - Machine Learning based Network Intrusion Detection.

***

## Group members (Team F)
<ul>
    <li>(Huy) Van Nhat Huy Nguyen <a href="mailto:s4571730@student.uq.edu.au">s4571730@student.uq.edu.au</a></li>
    <li>(Jason) Xurong Liang <a href="mailto:s4571850@student.uq.edu.au">s4571850@student.uq.edu.au</a></li>
    <li>Talia Garrett-Benson <a href="mailto:s4395592@student.uq.edu.au">s4395592@student.uq.edu.au</a> </li>
    <li>Emilie Aulie <a href="mailto:s4529780@student.uq.edu.au">s4529780@student.uq.edu.au</a> </li>
</ul>

***

## Info about each script/directory:
[proprocess_and_data_visualization.ipynb](./proprocess_and_data_visualization.ipynb) - notebook for raw dataset preprocess & visualization\
The preprocessed dataset may be found in [here](https://drive.google.com/drive/folders/1cnvofUhz84pMR0SztvfOYzIcpq1sR2VT?usp=sharing)\
[dataloader.py](./dataloader.py) - script that loads the preprocessed dataframe\
[evaluate.py](./evaluate.py) - script that contains functions for evaluation\
[ensemble_methods.py](./ensemble_methods.py) - script that used to run RF and Adaboost in comprehensive mode and imbalanced dataset mode\
[mlp_torch.py](./mlp_torch.py) - script that used to run MLP on comprehensive mode\
[mlp_torch_sampling.py](./mlp_torch_sampling.py) - script that used to run MLP to address imbalanced dataset problem\
[svm_dt.py](./svm_dt.py) - script that used to run SVM and decision tree classifiers in comprehensive mode and imbalanced dataset mode\
[knn.py](./knn.py) - script that used to run kNN classifier in comprehensive mode and imbalanced dataset mode\
[res](./res) - the directory where the computed results will be saved

***

## Usage
- Language: Python 3.9
- Require packages for non-MLP models: `pandas, scikit-learn, numpy, imblearn, platform`
- Require packages for MLP models: see [Note](#note) below
- For [ensemble_methods.py](./ensemble_methods.py), [mlp_torch.py](./mlp_torch.py) and [mlp_torch_sampling.py](./mlp_torch_sampling.py), input arguments need to be supplied to choose mode and hyperparameters settings.
- For [knn.py](./knn.py) and [svm_dt.py](./svm_dt.py), the hyperparameters need to be changed inside the script.

<!-- <ul>
    <li>For <a href="./ensemble_methods.py">ensemble_methods.py</a>,
<a href="./mlp_torch.py">mlp_torch.py</a> and <a href="./mlp_torch_sampling.py">mlp_torch_sampling.py</a>,
input arguments need to be supplied to choose mode and hyperparam settings.
    </li>
    <li>For <a href="./knn.py">knn.py</a> and  <a href="./svm_dt.py">svm_dt.py</a>,
    the hyperparameters need to be changed inside the script.
    </li>
</ul> -->

For those that require commandline arguments:\
e.g. Evaluate random forest's performance on dataset of all 3 types:
```shell
python3 ./ensemble_mathods.py --task full --classifier random_forest --res_dir ./res/
```
For argument help please refer to:
```shell
python3 ./ensemble_mathods.py -h
```

***

## Note
Conda virtual environment is recommended for MLP models due to dependencies of PyTorch. \
Instructions:
- Install Anaconda/Miniconda from [here](https://conda.io/projects/conda/en/latest/user-guide/install/download.html)
- Run `conda env create --file COMS6200_env.yml` to 
create the PyTorch environment of the project.
- Run `conda activate torch` to activate the environment.
- Run the MLP files inside the `torch` environment. 

Non-MLP models can be run normally without Conda.

***
<p align="center">Made with ❤ by <em>Team F</em></p>
