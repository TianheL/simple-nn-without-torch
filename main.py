from utils.dataset import Dataset
from utils.model import MLP
from utils.trainer import custom_trainer
import numpy as np
import os

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

save_path = os.path.join(save_path, f'hidden_{dim_hidden[0]}_{dim_hidden[1]}_{activation_function}_lr_{learning_rate}_decay_{lr_decay}_l2_{l2_reg}')
print(save_path)

np.random.seed(seed)

train_dataset = Dataset(type='train')
train_X, train_Y, val_X, val_Y = train_dataset.train_val_split(ratio=train_val_ratio)
test_dataset = Dataset(type='test')
test_X, test_Y = test_dataset.get_all_data()

# print('train_X.shape:', train_X.shape)
# print('train_Y.shape:', train_Y.shape)
# print('val_X.shape:', val_X.shape)
# print('val_Y.shape:', val_Y.shape)
# print('test_X.shape:', test_X.shape)
# print('test_Y.shape:', test_Y.shape)

model = MLP(input_size=3072, dim_hidden=dim_hidden, output_size=10, activation_function=activation_function)

trainer= custom_trainer(model, train_X, train_Y, val_X, val_Y, test_X, test_Y, batch_size=batch_size, epochs=epochs, lr=learning_rate, l2_reg=l2_reg, lr_decay=lr_decay, save_path=save_path)
trainer.train()

model.load_from_trained(os.path.join(save_path, 'model.pkl'))
model.forward(test_X)
acc = model.accuracy(test_Y)
print(f"test accuracy: {acc}")