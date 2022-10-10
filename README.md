# COMS6200 Project

This repo is for Sem 2, 2022 group project - Machine Learning based Network Intrusion Detection.

***

## Group members (Team F)
<ul>
    <li>(Huy) Van Nhat Huy Nguyen <a href="mailto:s4571730@student.uq.edu.au">s4571730@student.uq.edu.au</a></li>
    <li>Talia Garrett-Benson <a href="mailto:s4395592@student.uq.edu.au">s4395592@student.uq.edu.au</a> </li>
    <li>(Jason) Xurong Liang <a href="mailto:s4571850@student.uq.edu.au">s4571850@student.uq.edu.au</a></li>
    <li>Emilie Aulie <a href="mailto:s4529780@student.uq.edu.au">s4529780@student.uq.edu.au</a> </li>
</ul>

***

## Info about each script/directory:
[proprocess_and_data_visualization.ipynb](./proprocess_and_data_visualization.ipynb) - notebook for raw dataset preprocess & visualization\
[dataloader.py](./dataloader.py) - script that loads the preprocessed dataframe\
[evaluate.py](./evaluate.py) - script that contains functions for evaluation\
[ensemble_methods.py](./ensemble_methods.py) - script that used to run RF and Adaboost\
[mlp_torch.py](./mlp_torch.py) - script that used to run MLP on all data frames\
[mlp_torch_sampling.py](./mlp_torch_sampling.py) - script that used to run MLP to address imbalanced dataset problem\
[svm_dt.py](./svm_dt.py) - script that used to run SVM and decision tree classifiers

***

## Example usage
Evaluate random forest's performance on dataset of all 3 types:
```shell
python3 ./ensemble_mathods.py --task full --classifier random_forest --res_dir ./res/
```
For argument help please refer to:
```shell
python3 ./ensemble_mathods.py -h
```

***
<p align="center">Made with ❤️by <em>Team F</em></p>