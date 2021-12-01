import os
from copy import deepcopy
from datetime import datetime
import numpy as np
import argparse

from model.utils.evaluator import evaluation_my_dataset
from my_data_loader import load_data
from model.initialization import initialization
from model.utils import evaluation
from config import conf
from model.model import Model


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
opt = parser.parse_args()


# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0  # 当np.diag(array) 中array是一个1维数组时，结果形成一个以一维数组为对角线元素的矩阵,
                                                            # array是一个二维矩阵时，结果输出矩阵的对角线元素
    if not each_angle:  # bushimeigejiaodu jiuzhishuhcuyigeshuzhi
        result = np.mean(result)  # axis 不设置值，对 m*n 个数求均值，返回一个实数
    return result


# m = initialization(conf, test=opt.cache)[0]
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
# probe_source = load_data(**conf['probe_data'])
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
model_param['probe_source'] = None  # type: DataSet
model_param['train_source'] = None  # type: DataSet
model_param['test_source'] = None  # type: DataSet
model_param['train_pid_num'] = 0  ##
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

print('path' + str(	os.getcwd()))
# WORK_PATH = conf['WORK_PATH']
# print(WORK_PATH)
# print(WORK_PATH)
# os.chdir(WORK_PATH)

# load model checkpoint of iteration opt.iter
print('Loading the model of iteration %d...' % opt.iter)
m.load(opt.iter)






print('Transforming...')
time = datetime.now()
gallery = m.transform_gallery('gallery', opt.batch_size)  # 这里改成m.transform('probe', opt.batch_size)



###########
print('Evaluating...')
acc = evaluation_my_dataset(gallery, conf['data'])  # 修改划分形式
print('Evaluation complete. Cost:', datetime.now() - time)

# Print rank-1 accuracy of the best model
# e.g.
# ===Rank-1 (Include identical-view cases)===
# NM: 95.405,     BG: 88.284,     CL: 72.041
# print(type(acc))
# print(len(acc))  # 3
# print(acc.shape)  # (3,11,11,5)
# print(acc)
for i in range(1):
    # print(111111)

    print('===Rank-%d (Include identical-view cases)===' % (i + 1))
    print(np.mean(acc[0, :, :, i]))
    # print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
    #     np.mean(acc[0, :, :, i]),
    #     np.mean(acc[1, :, :, i]),
    #     np.mean(acc[2, :, :, i])))

# Print rank-1 accuracy of the best model，excluding identical-view cases
# e.g.
# ===Rank-1 (Exclude identical-view cases)===
# NM: 94.964,     BG: 87.239,     CL: 70.355
for i in range(1):
    # print(222222)

    print('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
    print(de_diag(acc[0, :, :, i]))
    # print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
    #     de_diag(acc[0, :, :, i]),
    #     de_diag(acc[1, :, :, i]),
    #     de_diag(acc[2, :, :, i]), ))

# Print rank-1 accuracy of the best model (Each Angle)
# e.g.
# ===Rank-1 of each angle (Exclude identical-view cases)===
# NM: [90.80 97.90 99.40 96.90 93.60 91.70 95.00 97.80 98.90 96.80 85.80]
# BG: [83.80 91.20 91.80 88.79 83.30 81.00 84.10 90.00 92.20 94.45 79.00]
# CL: [61.40 75.40 80.70 77.30 72.10 70.10 71.50 73.50 73.50 68.40 50.00]
np.set_printoptions(precision=2, floatmode='fixed')
# for i in range(1):
#     # print(333333)
#     print('===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
#     print('NM:', de_diag(acc[0, :, :, i], True))  # True shide shuchu le yihang
#     print('BG:', de_diag(acc[1, :, :, i], True))
#     print('CL:', de_diag(acc[2, :, :, i], True))