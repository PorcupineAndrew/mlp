from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''calculate Euclidean loss value (square of Euclidean norm).
        For best performance, use numpy rather than scipy.

        Args
        --------
        input: output from previous layer (numpy.array)
        target: label in onehot-encoding (numpy.array)

        Returns
        --------
        loss: loss value
        '''
        return np.sum((a-b)**2) / 2

    def backward(self, input, target):
        '''calculate Euclidean loss gradient

        Args
        --------
        input: output from previous layer (numpy.array)
        target: label in onehot-encoding (numpy.array)

        Returns
        --------
        loss: loss gradient (numpy.array)
        '''
        return input - target


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''calculate softmax loss value

        Args
        --------
        input: output from previous layer (numpy.array)
        target: label in onehot-encoding (numpy.array)

        Returns
        --------
        loss: loss value
        '''
        input_exp = np.exp(input)
        return -np.dot(np.log(input_exp/np.sum(input_exp)), target)

    def backward(self, input, target):
        '''calculate softmax loss gradient

        Args
        --------
        input: output from previous layer (numpy.array)
        target: label in onehot-encoding (numpy.array)

        Returns
        --------
        loss: loss gradient (numpy.array)
        '''
        input_exp = np.exp(input)
        softmax_value = input_exp / np.sum(input_exp)
        return softmax_value*np.sum(target)-target
