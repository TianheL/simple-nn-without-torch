import numpy as np

class DataLoader:
    def __init__(self, X, Y, batch_size=512):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.num_samples = X.shape[0]
        self.num_batches = self.num_samples // self.batch_size
        self.index = np.arange(self.num_samples)
        self.reset()

    def reset(self):
        np.random.shuffle(self.index)
        self.curent_batch = 0
    def get_iter_num(self):
        return self.num_batches

    def get_batch(self):
        if self.curent_batch < self.num_batches:
            lower_index = self.curent_batch * self.batch_size
            upper_index = (self.curent_batch+1)*self.batch_size
            self.curent_batch += 1
            return self.X[self.index[lower_index:upper_index]], self.Y[self.index[lower_index:upper_index]]
        else:
            raise StopIteration

