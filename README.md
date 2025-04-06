# simple-nn-without-torch

Codes for the HW1. Please follow the instructions in the README.md file to run the codes.

## Data Preparation

Use the following command to download the data:
```bash
cd data
bash download.sh
```

After downloading the data, extract the data from the tar.gz file.

## Training
Use the following command to train the model:
```bash
python main.py
```

By default, we use the following config and hyperparameters, which is the best result in the parameter search:
```python
# config & hyperparameters
seed=0
dim_hidden = [2048,2048]
activation_function = 'relu'
learning_rate = 0.1
lr_decay = 0.95
l2_reg = 1e-4
train_val_ratio = 0.8
epochs = 50
batch_size = 64
save_path = './saves'
```
The model parameter is saved in `model.pkl`, training history is saved in `training_log.csv`.

## Evaluation
We provide a model trained in the baidu cloud. You can download it and save it in the `saves` folder.
Use the following command to evaluate the model:
```bash
python test.py
```
Please modify the `save_path` in the `test.py` file to the path of the saved model if you want to use your own model. 

Use the `training_vis.ipynb` to plot the loss curve and accuracy curve.

## Parameter Search
Use the following command to search the best hyperparameters:
```bash
python search.py
```

By default, we use the following config and search space:
```python
# config & hyperparameters
seed=0
activation_function = 'relu'
train_val_ratio = 0.8
epochs = 50
batch_size = 64
save_path = './saves'
lr_decay = 0.95

# search parameters
Hidden = [256,512,1024,2048]
Ratio = [8,4,2,1]
LR = [0.1, 0.05, 0.01]
L2 = [1e-2, 1e-3, 1e-4, 1e-5]
```
Hidden is the dimension of the first layer, Ratio is the ratio of the first layer to the second layer, LR is the learning rate, L2 is the L2 regularization.

The search results are saved in the `search_result.txt` file and the best parameters are saved in the `best_params.txt` file.

## Parameter Visualization
Use the `pattern_vis.ipynb` to plot the pattern of the parameters. We provide the visualization of some of the first layer weights with their corresponding top activated images. We can observe some data patterns in the plots.