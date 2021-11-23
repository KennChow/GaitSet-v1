import os
import os.path as osp

import numpy as np


from model.utils.data_set import DataSet


def load_data(dataset_path, resolution, dataset, pid_shuffle, cache=True):
    print("cache in data_loader:")
    print(cache)
    seq_dir = list()
    view = list()
    seq_type = list()
    label = list()
    # print(dataset_path)  # /home/projects/output2
    print(dataset_path)
    # print(sorted(list(os.listdir(dataset_path))))  # ['001', '002', '003'...]
    for _label in sorted(list(os.listdir(dataset_path))):  # _label：001、002...
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        if dataset == 'CASIA-B' and _label == '005':
            continue
        label_path = osp.join(dataset_path, _label)
        # print("label_path")
        # print(label_path)  # /home/projects/output2/001...
        # print(sorted(list(os.listdir(label_path))))  # ['bg-01', 'bg-02', 'cl-01', 'cl-02'...]
        for _seq_type in sorted(list(os.listdir(label_path))):  # _seq_type:bg-01、bg-02...
            seq_type_path = osp.join(label_path, _seq_type)  # /home/projects/output2/001/bg-01、bg-02...
            # print(seq_type_path)  # /home/projects/output2/001/bg-01、bg-02...
            for _view in sorted(list(os.listdir(seq_type_path))):
                _seq_dir = osp.join(seq_type_path, _view)  # /home/projects/output2/124/nm-06/180
                seqs = os.listdir(_seq_dir)  # ['124-nm-06-180-097.png', '124-nm-06-180-059.png'...]
                if len(seqs) > 0:  # 步态视频帧数(照片数量)
                    seq_dir.append([_seq_dir])  # 每个人每个行走状态每个角度的集合  完整的路径
                    label.append(_label)  # 所有人的标签的集合
                    seq_type.append(_seq_type)  # 行走状态的集合
                    view.append(_view)  # 拍摄角度的集合

    # pid_fname = osp.join('partition', '{}_{}_{}.npy'.format(
    #     dataset, pid_num, pid_shuffle))  # partition/CASIA-B_73_False.npy
    # # LT:前74个人作为训练集 后50个人作为验证集
    # if not osp.exists(pid_fname):   # 生成CASIA-B_73_False.npy文件
    #     pid_list = sorted(list(set(label)))
    #     if pid_shuffle:
    #         np.random.shuffle(pid_list)
    #     pid_list = [pid_list[0:pid_num], pid_list[pid_num:]]
    #     os.makedirs('partition', exist_ok=True)
    #     np.save(pid_fname, pid_list)
    #
    # pid_list = np.load(pid_fname, allow_pickle=True)
    # # [list(['001', '002', '003', '004', '006', ... '074'])
    # #  list(['075', '076',...'123', '124'])]
    # train_list = pid_list[0]
    # test_list = pid_list[1]
    pid_list = sorted(list(set(label)))
    gallery_list = pid_list
    gallery_source = DataSet(
        [seq_dir[i] for i, l in enumerate(label) if l in gallery_list],
        [label[i] for i, l in enumerate(label) if l in gallery_list],
        [seq_type[i] for i, l in enumerate(label) if l in gallery_list],
        [view[i] for i, l in enumerate(label) if l in gallery_list],
        cache, resolution)
    aaa = [seq_dir[i] for i, l in enumerate(label) if l in gallery_list]
    print("aaa:")
    print(aaa)  # ['/home/projects/data_test/125/nm-01/072']
    # train_source = DataSet(
    #     [seq_dir[i] for i, l in enumerate(label) if l in train_list],
    #     [label[i] for i, l in enumerate(label) if l in train_list],
    #     [seq_type[i] for i, l in enumerate(label) if l in train_list],
    #     [view[i] for i, l in enumerate(label) if l in train_list],
    #     cache, resolution)
    # test_source = DataSet(
    #     [seq_dir[i] for i, l in enumerate(label) if l in test_list],
    #     [label[i] for i, l in enumerate(label) if l in test_list],
    #     [seq_type[i] for i, l in enumerate(label) if l in test_list],
    #     [view[i] for i, l in enumerate(label) if l in test_list],
    #     cache, resolution)
    # seq_dir_probe = '/home/projects/data_test'
    # probe_source = DataSet(
    #     [seq_dir[i] for i, l in enumerate(label) if l in gallery_list],
    #     [label[i] for i, l in enumerate(label) if l in gallery_list],
    #     [seq_type[i] for i, l in enumerate(label) if l in gallery_list],
    #     [view[i] for i, l in enumerate(label) if l in gallery_list],
    #     cache, resolution)

    print("测试集数量：" + str(len(gallery_source)))  # 5485
    # print(type(test_source))  # <class 'model.utils.data_set.DataSet'>
    return gallery_source  # , probe_source
