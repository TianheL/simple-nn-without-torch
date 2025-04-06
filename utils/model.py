import numpy as np
import pickle
import math

class MLP:
    def __init__(self, input_size=3072, dim_hidden=[512,128], output_size=10, activation_function='relu'):
        self.input_size = input_size
        self.dim_hidden = dim_hidden
        self.output_size = output_size
        self.activation_function = activation_function
        self.num_layers = len(dim_hidden) + 1
        self.init_weights()
    
    def init_weights(self):
        self.weights = []
        self.biases = []

        # init fc1.weight
        fan_in = self.input_size
        a = math.sqrt(5)
        bound = math.sqrt(6 / ((1 + a ** 2) * fan_in))
        weight = np.random.uniform(-bound, bound, (self.input_size, self.dim_hidden[0]))
        self.weights.append(weight)

        # init fc1.bias
        bound_bias = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        bias = np.random.uniform(-bound_bias, bound_bias, self.dim_hidden[0])
        self.biases.append(bias)

        # init mid-layer weights and biases
        for i in range(1, self.num_layers - 1):
            fan_in = self.dim_hidden[i - 1]
            bound = math.sqrt(6 / ((1 + a ** 2) * fan_in))
            weight = np.random.uniform(-bound, bound, (self.dim_hidden[i - 1], self.dim_hidden[i]))
            self.weights.append(weight)

            bound_bias = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            bias = np.random.uniform(-bound_bias, bound_bias, self.dim_hidden[i])
            self.biases.append(bias)

        # init last-layer.weight
        fan_in = self.dim_hidden[-1]
        bound = math.sqrt(6 / ((1 + a ** 2) * fan_in))
        weight = np.random.uniform(-bound, bound, (self.dim_hidden[-1], self.output_size))
        self.weights.append(weight)

        # init last-layer.bias
        bound_bias = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        bias = np.random.uniform(-bound_bias, bound_bias, self.output_size)
        self.biases.append(bias)

    def act_func(self, X):
        if self.activation_function=='sigmoid':
            return 1/(1+np.exp(-X))
        elif self.activation_function=='relu':
            return np.maximum(X, 0)
        elif self.activation_function=='tanh':
            return np.tanh(X)
        else:
            raise NotImplementedError('activation function not implemented')
        
    def grad_act_func(self, output):
        if self.activation_function=='sigmoid':
            return output*(1-output)
        elif self.activation_function=='relu':
            return np.where(output>0, 1, 0)
        elif self.activation_function=='tanh':
            return 1-np.square(output)
        else:
            raise NotImplementedError('activation function not implemented')
    
    def softmax_funx(self, X):
        exp_X = np.exp(X-np.max(X, axis=1, keepdims=True))
        return exp_X/np.sum(exp_X, axis=1, keepdims=True)

    def forward(self, input_data):
        self.data = [input_data]
        for i in range(self.num_layers-1):
            self.data.append(self.act_func(np.dot(self.data[-1], self.weights[i])+self.biases[i]))
        self.pred = self.softmax_funx(np.dot(self.data[-1], self.weights[-1])+self.biases[-1])
        return self.pred
    
    def cross_entropy(self, labels):
        return -np.sum(labels*np.log(self.pred+1e-8))/self.pred.shape[0]
    
    def accuracy(self, labels):
        return np.mean(np.argmax(self.pred, axis=1)==np.argmax(labels, axis=1))

    def backward(self, labels):
        self.grad_weights = []
        self.grad_biases = []

        dZ = self.pred - labels
        dW = np.dot(self.data[-1].T, dZ) / labels.shape[0]
        db = np.sum(dZ, axis=0) / labels.shape[0]
        self.grad_weights.insert(0, dW)
        self.grad_biases.insert(0, db)

        for i in range(self.num_layers - 2, -1, -1):
            dZ = np.dot(dZ, self.weights[i + 1].T) * self.grad_act_func(self.data[i + 1])
            dW = np.dot(self.data[i].T, dZ) / labels.shape[0]
            db = np.sum(dZ, axis=0) / labels.shape[0]
            self.grad_weights.insert(0, dW)
            self.grad_biases.insert(0, db)

    def step(self, lr, l2_reg):
        for i in range(self.num_layers):
            self.weights[i] -= lr*(self.grad_weights[i] + l2_reg*self.weights[i])
            self.biases[i] -= lr*self.grad_biases[i]
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.weights, self.biases), f)

    def load_from_trained(self, path):
        with open(path, 'rb') as f:
            weights, biases = pickle.load(f)
            self.weights = weights
            self.biases = biases




