from __future__ import division
import numpy as np
import sys


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
        return np.mean(np.sum((input-target)**2, axis=1)) / 2

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
        delta = input-target
        return delta / delta.shape[0]


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
        cross_entropy = np.log(input_exp/input_exp.sum(axis=1)[:,None])
        return -np.mean((cross_entropy*target).sum(axis=1))

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
        softmax_value = input_exp / input_exp.sum(axis=1)[:,None]
        return softmax_value*target.sum(axis=1)[:,None]-target
