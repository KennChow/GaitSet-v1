# -*- coding: utf-8 -*-
# @Author  : admin
# @Time    : 2018/11/15
import os
from copy import deepcopy

import numpy as np

from .utils import load_data
from .model import Model



def initialize_data(config, train=False, test=False):
    print("Initializing data source...")
    print("cache:" + str((train or test)))
    train_source, test_source = load_data(**config['data'], cache=(train or test))  # type: DataSet
    gallery_source = None
    # print(train or test)  # False
    print("train:" + str(train))
    print("test:" + str(test))
    if train:
        print("Loading training data...")
        train_source.load_all_data()
    if test:
        print("Loading test data...")
        test_source.load_all_data()
    print("Data initialization complete.")
    return train_source, test_source, gallery_source

# self.gallery_source = gallery_source
def initialize_model(config, train_source, test_source, gallery_source):
    print("Initializing model...")
    data_config = config['data']
    model_config = config['model']
    model_param = deepcopy(model_config)
    model_param['train_source'] = train_source  # type: DataSet
    model_param['test_source'] = test_source  # type: DataSet
    model_param['gallery_source'] = gallery_source  # type: DataSet
    # model_param['probe_source'] = probe_source  # type: DataSet
    model_param['train_pid_num'] = data_config['pid_num']
    batch_size = int(np.prod(model_config['batch_size']))
    model_param['save_name'] = '_'.join(map(str,[
        model_config['model_name'],
        data_config['dataset'],
        data_config['pid_num'],
        data_config['pid_shuffle'],
        model_config['hidden_dim'],
        model_config['margin'],
        batch_size,
        model_config['hard_or_full_trip'],
        model_config['frame_num'],
    ]))


    m = Model(**model_param)
    print("Model initialization complete.")
    return m, model_param['save_name']


def initialization(config, train=False, test=False):
    print("Initialzing...")
    WORK_PATH = config['WORK_PATH']
    os.chdir(WORK_PATH)
    os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]
    # print(train)  # false
    # print(test)  # false
    train_source, test_source, gallery_source = initialize_data(config, train, test)  # type: DataSet
    return initialize_model(config, train_source, test_source, gallery_source)