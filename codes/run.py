#!/usr/bin/env python3
#-*- coding:utf-8 -*-
# **********************************************************************
# * Description   : training for one model
# * Last change   : 11:49:18 2019-10-04
# * Author        : Yihao Chen
# * Email         : chenyiha17@mails.tsinghua.edu.cn
# * License       : none
# **********************************************************************

from network import Network
from utils import LOG_INFO, createDir
from layers import Relu, Sigmoid, Linear
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import time

config = {
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'momentum': 0.01,
    'batch_size': 100,
    'max_epoch': 500,
    'disp_freq': 50,
    'test_epoch': 5,
    'plot_epoch': 100,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="config to modify", type=str, default="learning_rate:0.001 weight_decay:0.01 momentum:0.01")
    parser.add_argument("-arch", help="network architecture", type=str, default="Lin-784-10 Relu")
    parser.add_argument("-loss", help="loss type", type=str, default="Euclidean")
    parser.add_argument("-name", help="experiment name", type=str, default="default")
    args = parser.parse_args()

    # result dir
    result_dir = os.path.join("./result", args.name)
    plot_dir = os.path.join(result_dir, "plot")
    createDir(plot_dir, removeflag=1)

    # modify config
    for item in args.config.strip().split(" "):
        key, value = item.split(":")
        assert key in config, "parse error: "+item
        config[key] = type(config[key])(value)

    # build network
    model = Network()
    for item in args.arch.strip().split(" "):
        parts = item.split("-")
        if parts[0] == "Lin":
            model.add(Linear('linear', int(parts[1]), int(parts[2]), 0.01))
        elif parts[0] == "Relu":
            model.add(Relu('relu'))
        elif parts[0] == "Sigm":
            model.add(Sigmoid('sigm'))
        else:
            raise Exception("parse error: "+item)

    # get loss
    if args.loss == "Euclidean":
        loss = EuclideanLoss(name='loss')
    elif args.loss == "Softmax":
        loss = SoftmaxCrossEntropyLoss(name='loss')
    else:
        raise Exception("parse error: "+args.loss)

    # get data
    train_data, test_data, train_label, test_label = load_mnist_2d('data')

    # print info
    print("======================================================")
    for key, value in config.items():
        print(key+": "+str(value))
    print("network: "+args.arch)
    print("loss: "+args.loss)
    print("result dir: "+result_dir)
    print("======================================================")

    # plot init
    acc_line = []
    loss_line = []

    # training
    start_time = time.time()
    for epoch in range(config['max_epoch']):
        LOG_INFO('Training @ %d epoch...' % (epoch))
        loss_list, acc_list = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])

        if epoch % config['test_epoch'] == 0:
            LOG_INFO('Testing @ %d epoch...' % (epoch))
            test_net(model, loss, test_data, test_label, config['batch_size'])

        if epoch % config['plot_epoch'] == 0:
            LOG_INFO('Plot @ %d epoch...' % (epoch))
            loss_line.append((epoch, loss_list))
            acc_line.append((epoch, acc_list))

    print("======================================================")
    LOG_INFO('Testing @ final epoch...')
    test_net(model, loss, test_data, test_label, config['batch_size'])
    print("training time: %d seconds" % int(time.time()-start_time))


    # plot loss
    x = list([i+1 for i in range(len(loss_line))])
    plt.figure(figsize=(8,4))
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.legend(loc="best")
    for item in loss_line:
        plt.plot(x, item[1], linewidth=1, label="epoch "+str(item[0]))
    plt.title(args.name+" loss", fontweight=800)
    plt.savefig(os.path.join(plot_dir, args.name+"_loss.png"))

    # plot acc
    plt.figure(figsize=(8,4))
    plt.xlabel("iter")
    plt.ylabel("acc")
    plt.legend(loc="best")
    for item in acc_line:
        plt.plot(x, item[1], linewidth=1, label="epoch "+str(item[0]))
    plt.title(args.name+" accuracy", fontweight=800)
    plt.savefig(os.path.join(plot_dir, args.name+"_acc.png"))
