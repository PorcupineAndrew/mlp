import numpy as np
from scipy.special import expit
from utils import LOG_INFO


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor


class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, input):
        '''linear rectifier activation, input as u, output as
        f(u) = max(0, u). For best performance, use in-replace
        max method.

        Args
        --------
        input: numpy.array([dim for input layer])

        Returns
        --------
        output: numpy.array([dim for current layer])
        '''
        self._saved_for_backward(input)
        return np.maximum(input, 0, input)

    def backward(self, grad_output):
        '''linear rectifier gradient

        Args
        --------
        grad_output: numpy.array([dim for current layer])

        Returns
        --------
        output: numpy.array([dim for input layer])
        '''
        return (1 * (self._saved_tensor>0)) * grad_output


class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, input):
        '''sigmoid activation, input as u, output as
        f(u) = 1/(1 + exp(-u)). For best performance,
        use scipy.special.expit

        Args
        --------
        input: numpy.array([dim for input layer])

        Returns
        --------
        output: numpy.array([dim for current layer])
        '''
        output = expit(input)
        self._saved_for_backward(output)
        return output

    def backward(self, grad_output):
        '''sigmoid gradient

        Args
        --------
        grad_output: numpy.array([dim for current layer])

        Returns
        --------
        output: numpy.array([dim for input layer])
        '''
        sigmoid_value = self._saved_tensor
        return (sigmoid_value * (1-sigmoid_value)) * grad_output


class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        '''treat each input as a row vector and produce an 
        output vector by doing matrix multiplication with
        weight W and then adding bias b.

        Args
        --------
        input: numpy.array([dim for input layer])

        Returns
        --------
        output: numpy.array([dim for current layer])
        '''
        self._saved_for_backward(input)
        return np.dot(input, self.W) + self.b

    def backward(self, grad_output):
        '''linear gradient

        Args
        --------
        grad_output: numpy.array([dim for current layer])

        Returns
        --------
        output: numpy.array([dim for input layer])
        '''
        input = self._saved_tensor
        self.grad_W = np.array([input for i in range(out_num)]).T
        self.grad_b = np.ones(out_num)
        return np.dot(grad_output, self.W.T)

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
