import numpy as np
import os
from matplotlib import pyplot as plt
import gzip
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class Dataset:
    def __init__(self, data_path='./data/cifar-10-python/cifar-10-batches-py', type='train'):
        self.label = ['airplane', 'automoblie','bird','cat','deer', 'dog','frog','horse','ship','truck']
        if type == 'train':
            self.X = np.zeros([50000,3072], dtype=np.uint8)
            self.Y = np.zeros([50000,10], dtype=np.uint8)
            FileList = ['data_batch_1',
                        'data_batch_2',
                        'data_batch_3',
                        'data_batch_4',
                        'data_batch_5']
        elif type == 'test':
            self.X = np.zeros([10000,3072], dtype=np.uint8)
            self.Y = np.zeros([10000,10], dtype=np.uint8)
            FileList = ['test_batch']

        index = 0
        for file in FileList:
            dict = unpickle(os.path.join(data_path,file))
            self.X[index:index + 10000, ...] = dict[b'data']
            
            self.Y[np.arange(index,index+10000), dict[b'labels']] = 1
            index += 10000
        self.X = self.X.reshape(-1, 3, 32, 32)
        self.X = self.X / 255.0
        self.X = (self.X - 0.5) / 0.5
        self.X = self.X.reshape(-1, 3072)

    def get_data(self, index=0, imshow=True):
        if imshow == True:
            plt.imshow(self.X[index].reshape([3,32,32]).transpose([1,2,0]))
            plt.show()
            print(self.label[np.where(self.Y[index]==1)[0][0]])
        return self.X[index], self.Y[index]
    
    def get_all_data(self): 
        return self.X, self.Y

    
    def train_val_split(self, ratio=0.8):
        idx = np.random.permutation(len(self.Y))
        self.X = self.X[idx]
        self.Y = self.Y[idx]
        self._num_train = int(len(self.Y) * ratio)
        return self.X[:self._num_train], self.Y[:self._num_train], self.X[self._num_train:], self.Y[self._num_train:]