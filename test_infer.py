from datetime import datetime
import numpy as np
import argparse
####
from model.initialization import initialization
from model.utils import infer


from config import conf
# 121212

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


# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0  # 当np.diag(array) 中array是一个1维数组时，结果形成一个以一维数组为对角线元素的矩阵,
                                                            # array是一个二维矩阵时，结果输出矩阵的对角线元素
    if not each_angle:  # bushimeigejiaodu jiuzhishuhcuyigeshuzhi
        result = np.mean(result)  # axis 不设置值，对 m*n 个数求均值，返回一个实数
    return result


m = initialization(conf, test=opt.cache)[0]  # 这里已经加载好了数据集


# load model checkpoint of iteration opt.iter
print('Loading the model of iteration %d...' % opt.iter)
m.load(opt.iter)
print('Transforming...')
time = datetime.now()
test = m.transform('test', opt.batch_size)  # test --> np.concatenate(feature_list, 0), view_list, seq_type_list, label_list
print('Inferring...')
print(opt.probe_data)
infer(opt.probe_data, test, conf['data'])

# print('Evaluation complete. Cost:', datetime.now() - time)

# Print rank-1 accuracy of the best model
# e.g.
# ===Rank-1 (Include identical-view cases)===
# NM: 95.405,     BG: 88.284,     CL: 72.041
# print(type(acc))
# print(len(acc))  # 3
# print(acc.shape)  # (3,11,11,5)
# print(acc)

# for i in range(1):
#     # print(111111)
#     print('===Rank-%d (Include identical-view cases)===' % (i + 1))
#     print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
#         np.mean(acc[0, :, :, i]),
#         np.mean(acc[1, :, :, i]),
#         np.mean(acc[2, :, :, i])))

# Print rank-1 accuracy of the best model，excluding identical-view cases
# e.g.
# ===Rank-1 (Exclude identical-view cases)===
# NM: 94.964,     BG: 87.239,     CL: 70.355
# for i in range(1):
#     # print(222222)
#     print('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
#     print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
#         de_diag(acc[0, :, :, i]),
#         de_diag(acc[1, :, :, i]),
#         de_diag(acc[2, :, :, i]), ))

# Print rank-1 accuracy of the best model (Each Angle)
# e.g.
# ===Rank-1 of each angle (Exclude identical-view cases)===
# NM: [90.80 97.90 99.40 96.90 93.60 91.70 95.00 97.80 98.90 96.80 85.80]
# BG: [83.80 91.20 91.80 88.79 83.30 81.00 84.10 90.00 92.20 94.45 79.00]
# CL: [61.40 75.40 80.70 77.30 72.10 70.10 71.50 73.50 73.50 68.40 50.00]
# np.set_printoptions(precision=2, floatmode='fixed')
# for i in range(1):
#     # print(333333)
#     print('===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
#     print('NM:', de_diag(acc[0, :, :, i], True))  # True shide shuchu le yihang
#     print('BG:', de_diag(acc[1, :, :, i], True))
#     print('CL:', de_diag(acc[2, :, :, i], True))
#
