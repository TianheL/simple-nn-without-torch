from utils.dataset import Dataset
from utils.model import MLP
from utils.trainer import custom_trainer
import numpy as np
import os

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

best_val_acc=0
best_dim_hidden=[]
best_learning_rate=0
best_l2_reg=0

with open('search_result.txt', 'w') as f:
    f.write('dim_hidden,learning_rate,l2_reg,val_acc\n')

for learning_rate in LR:
    for l2_reg in L2:
        for hidden in Hidden:
            for ratio in Ratio:
                dim_hidden = [hidden, hidden//ratio]
                save_path = os.path.join('./saves', f'hidden_{dim_hidden[0]}_{dim_hidden[1]}_{activation_function}_lr_{learning_rate}_decay_{lr_decay}_l2_{l2_reg}')
                print(save_path)

                np.random.seed(seed)

                train_dataset = Dataset(type='train')
                train_X, train_Y, val_X, val_Y = train_dataset.train_val_split(ratio=train_val_ratio)
                test_dataset = Dataset(type='test')
                test_X, test_Y = test_dataset.get_all_data()

                model = MLP(input_size=3072, dim_hidden=dim_hidden, output_size=10, activation_function=activation_function)

                trainer= custom_trainer(model, train_X, train_Y, val_X, val_Y, test_X, test_Y, batch_size=batch_size, epochs=epochs, lr=learning_rate, l2_reg=l2_reg, lr_decay=lr_decay, save_path=save_path)
                val_acc = trainer.train()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_dim_hidden = dim_hidden
                    best_learning_rate = learning_rate
                    best_l2_reg = l2_reg
                with open('search_result.txt', 'a') as f:
                    f.write('{},{},{},{}\n'.format(dim_hidden, learning_rate, l2_reg, val_acc))
                
                with open('best_params.txt', 'w') as f:
                    f.write('dim_hidden,learning_rate,l2_reg,val_acc\n')
                    f.write('{},{},{},{}\n'.format(best_dim_hidden, best_learning_rate, best_l2_reg, best_val_acc)) 