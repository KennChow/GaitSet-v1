import os
from copy import deepcopy
from config import conf
from my_data_loader import load_data
import numpy as np
from model.model import Model
import torch
import torch.nn.functional as F
from model.utils import infer
import argparse

def cuda_dist(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
        1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    return dist


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--iter', default='80000', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 80000')
parser.add_argument('--batch_size', default='1', type=int,
                    help='batch_size: batch size for parallel test. Default: 1')
parser.add_argument('--cache', default=False, type=boolean_string,
                    help='cache: if set as TRUE all the test data will be loaded at once'
                         ' before the transforming start. Default: FALSE')
parser.add_argument('--probe_data', type=str)

opt = parser.parse_args()

def gallery_load_data():
    print("Initialzing...")
    WORK_PATH = conf['WORK_PATH']
    os.chdir(WORK_PATH)
    os.environ["CUDA_VISIBLE_DEVICES"] = conf["CUDA_VISIBLE_DEVICES"]
    # print(train)  # false
    # print(test)  # false

    # initialize_data(config):
    print("Initializing data source...")
    # print("path:")
    # print(**conf['gallery_data'])
    gallery_source = load_data(**conf['gallery_data'])  # type: DataSet
    probe_source = load_data(**conf['probe_data'])
    # print(train or test)  # False
    print("Loading gallery data...")
    gallery_source.load_all_data()
    print("Data initialization complete.")

    #initialize_model(config, train_source, test_source)
    print("Initializing model...")
    data_config = conf['gallery_data']
    model_config = conf['model']
    model_param = deepcopy(model_config)
    model_param['gallery_source'] = gallery_source  # type: DataSet
    model_param['probe_source'] = probe_source  # type: DataSet
    model_param['train_source'] = None  # type: DataSet
    model_param['test_source'] = None  # type: DataSet
    model_param['train_pid_num'] = 0  ##
    batch_size = int(np.prod(model_config['batch_size']))
    model_param['save_name'] = '_'.join(map(str,[
        model_config['model_name'],
        data_config['dataset'],
        data_config['pid_shuffle'],
        model_config['hidden_dim'],
        model_config['margin'],
        batch_size,
        model_config['hard_or_full_trip'],
        model_config['frame_num'],
    ]))
    m = Model(**model_param)
    print("Model initialization complete.")
    # return m, model_param['save_name']
    # initilization()结束

    gallery_source = m.transform_gallery('gallery', opt.batch_size)  # test --> np.concatenate(feature_list, 0), view_list, seq_type_list, label_list
    # print(gallery_source)
    feature, view, seq_type, label = gallery_source
    # print(label)
    # print("feature:")
    # print(type(feature))
    # for f in feature:
    #     print(f)

    probe_source = m.transform_gallery('probe', opt.batch_size)
    pfeature, pview, pseq_type, plabel = probe_source
    # print("pfeature:")
    # print(type(pfeature))
    # for pf in pfeature:
    #     print(pf)

    # return train_source, test_source

    dist = cuda_dist(feature, pfeature)
    # print(dist)
    # print(dist.shape)
    # print(dist < 0.1)  # tensor([[ True], [False], [False], [False], [ True], [ True], [False], [False], [ True],
    # [ True], [ True], [ True]], device='cuda:0')
    # print(torch.sort(dist, 0))
    idx = int(dist.argmin(0).cpu().numpy())  # ndarray
    # print(idx)
    # print(label)  # ['125', '125', '126', '126', '127', '127', '128', '128', '129', '129', '130', '130']
    print(label[idx])
    # print(dist.argmin(0))  # tensor([10], device='cuda:0')
    # print(min_index.cpu().numpy())  # ndarray

print(11)
gallery_load_data()
