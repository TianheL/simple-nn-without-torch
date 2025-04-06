import numpy as np
from utils.dataloader import DataLoader
from tqdm import tqdm
import os

class custom_trainer:
    def __init__(self, model, train_X, train_Y, val_X, val_Y, test_X, test_Y, batch_size=512, epochs=10, lr=0.01, l2_reg=1e-3, lr_decay=0.95, save_path='./saves'):
        self.model = model
        self.train_X = train_X
        self.train_Y = train_Y
        self.val_X = val_X
        self.val_Y = val_Y
        self.test_X = test_X
        self.test_Y = test_Y
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.l2_reg = l2_reg
        self.lr_decay = lr_decay
        self.dataloader = DataLoader(self.train_X, self.train_Y, batch_size=self.batch_size)
        self.iter_per_epoch = self.dataloader.get_iter_num()
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        with open(os.path.join(self.save_path,'training_log.csv'), 'w') as f:
            f.write('epoch,train_loss,train_acc,val_loss,val_acc\n')
        
    def train(self):
        best_val_acc = 0
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        for epoch in range(self.epochs):
            self.dataloader.reset()
            progress_bar = tqdm(total=self.iter_per_epoch, desc='Epoch {}/{}'.format(epoch+1, self.epochs), unit='it')
            for i in range(self.iter_per_epoch):
                X, Y = self.dataloader.get_batch()
                self.model.forward(X)
                self.model.backward(Y)
                self.model.step(self.lr, self.l2_reg)

                progress_bar.update()
                progress_bar.refresh()
            
            self.model.forward(self.train_X)
            train_loss.append(self.model.cross_entropy(self.train_Y))
            train_acc.append(self.model.accuracy(self.train_Y))
            self.model.forward(self.val_X)
            val_loss.append(self.model.cross_entropy(self.val_Y))
            val_acc.append(self.model.accuracy(self.val_Y))
            print('Epoch: {}/{}'.format(epoch+1, self.epochs), 'LR: {:.4f}'.format(self.lr,), 'Train Loss: {:.4f}'.format(train_loss[-1]), 'Train Acc: {:.4f}'.format(train_acc[-1]), 'Val Loss: {:.4f}'.format(val_loss[-1]), 'Val Acc: {:.4f}'.format(val_acc[-1]))
            self.lr = self.lr * self.lr_decay
            with open(os.path.join(self.save_path,'training_log.csv'), 'a') as f:
                f.write('{},{},{},{},{}\n'.format(epoch+1, train_loss[-1], train_acc[-1], val_loss[-1], val_acc[-1]))
            if val_acc[-1] > best_val_acc:
                best_val_acc = val_acc[-1]
                self.model.save(os.path.join(self.save_path,'model.pkl'))
                with open(os.path.join(self.save_path,'best_epoch.txt'), 'w') as f:
                    f.write(str(epoch+1))
        return best_val_acc





        
