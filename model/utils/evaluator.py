import torch
import torch.nn.functional as F
import numpy as np


def cuda_dist(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
        1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    return dist


def evaluation(data, config):
    dataset = config['dataset'].split('-')[0]  # CASIA
    # print(dataset)
    feature, view, seq_type, label = data  # feature为特征向量
    # print(feature.shape)  # 5485*15872  5485个视频 每个视频的特征向量为15872个元素的向量
    # print("type(feature)：")
    # print(type(feature))  # <class 'numpy.ndarray'>
    # print(feature)
    label = np.array(label)
    # print("label:")
    # print(len(label))  # 5485  --> 说明一共有5485个视频
    # print(len(set(label)))  # 50 --> 验证集50个人
    view_list = list(set(view))  # 每个人的每种行走状态11个角度
    view_list.sort()
    view_num = len(view_list)

    # print("view_list:")
    # print(view_list)  #   # 11个 ['000', '018', '036', '054', '072', '090' ... ]
    # print("seq_type:")
    # print(list(set(seq_type)))  # 10个 ['cl-01', 'nm-01', 'nm-06', 'bg-02', 'nm-05', 'nm-04', 'bg-01', 'nm-03', 'cl-02', 'nm-02']

    sample_num = len(feature)
    # print("sample_num", end=" ")
    # print(sample_num)  # 5485
    #  这里修改一下划分形式  ************************
    probe_seq_dict = {'CASIA': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['00']]}
    gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']]}

    num_rank = 5
    acc = np.zeros(  [   len(probe_seq_dict[dataset]), view_num, view_num, num_rank  ]    )  # (3,11,11,5)

    # cnt = 0
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        # print("----1-----")
        # print(p)  # 0 1
        # print(probe_seq)  # ['nm-05', 'nm-06']  //   ['bg-01', 'bg-02']

        for gallery_seq in gallery_seq_dict[dataset]:
            # print("----2-----")
            # print(gallery_seq)  # ['nm-01', 'nm-02', 'nm-03', 'nm-04']  //   ['nm-01', 'nm-02', 'nm-03', 'nm-04']

            for (v1, probe_view) in enumerate(view_list):  # # 对应一个探针视频
                # print("----3-----")
                # print(v1)  # 0 1 2...
                # print(probe_view)  # 000 018 036...

                for (v2, gallery_view) in enumerate(view_list):  # 对应一个步态库视频

                    # print(str(probe_seq) + probe_view + " vs " + str(gallery_seq) + gallery_view)
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']000  100 vs 200
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']018
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']036
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']054
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']072
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']090
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']108
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']126
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']144
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']162
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']180
                    # cnt += 1  循环了363次
                    # print("----4-----")
                    # print(v2)  # 0 1 2...
                    # print(gallery_view)  # 000 018 036...

                    # codes above work to iter, down to calculate the distance
                    ################### ？？？？？？？？？？？？
                    # np.isin(a,b) 用于判定a中的元素在b中是否出现过，如果出现过返回True,否则返回False,
                    # 最终结果为一个形状和a一模一样的数组。
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])  # 看哪些行走类型以及角度出现过
                    # 用来挑出当前步态库中视频对应的特征向量  类似于子网掩码的作用
                    # print("gallery_seq:")
                    # print(gallery_seq)  # ['nm-01', 'nm-02', 'nm-03', 'nm-04']  4个
                    # print("[gallery_view]:")
                    # print([gallery_view])  # ['018'] .....  1个
                    # print("gseq_mask：")
                    # print(gseq_mask)  # [False False False ... False False False]
                    # print("np.sum(gseq_mask):")
                    # print(np.sum(gseq_mask))  # 200左右
                    # print(gseq_mask.shape)  # (5485,)
                    gallery_x = feature[gseq_mask, :]  # 2-D metric  当前步态库视频对应的特征向量
                    # 挑出步态库中的200个满足名字中包括'['nm-01', 'nm-02', 'nm-03', 'nm-04']000'的视频的特征向量
                    # print("gallery_x.shape：")
                    # print(gallery_x.shape)  # 200*15872
                    # print("gallery_x：")
                    # print(gallery_x)
                    gallery_y = label[gseq_mask]  # 1-D list
                    # 挑出步态库中的200个满足名字中包括'['nm-01', 'nm-02', 'nm-03', 'nm-04']000'的标签
                    # print("gallery_y.shape", end=" ")
                    # print(gallery_y.shape)  # 200,
                    # print("gallery_y", end=" ")
                    # print(gallery_y)


                    #################### ？？？？？？？？？？
                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                    # 用来挑出探针视频对应的特征向量
                    # print("pseq_mask：")
                    # print(pseq_mask)  # [False False False ... False False False]
                    # print("np.sum(pseq_mask):")
                    # print(np.sum(pseq_mask))  # 100左右
                    # print(pseq_mask.shape)  # (5485,)
                    probe_x = feature[pseq_mask, :]  # 2-D metric  当前探针视频对应的特征向量
                    # 挑出探针中的100个满足名字中包括'['nm-05', 'nm-06']000'的视频的特征向量
                    # print("probe_x.shape", end=" ")
                    # print(probe_x.shape)  # 100*15872
                    # print("probe_x", end=" ")
                    # print(probe_x)
                    probe_y = label[pseq_mask]  # 1-D list
                    # 挑出探针中的100个满足名字中包括'['nm-05', 'nm-06']000'的视频的标签(人)
                    # print("probe_y.shape", end=" ")
                    # print(probe_y.shape)  # 100,
                    # print("probe_y", end=" ")
                    # print(probe_y)

                    dist = cuda_dist(probe_x, gallery_x)  # tensor   找出每行距离最小的
                    # print("type(dist):")
                    # print(type(dist))  # <class 'torch.Tensor'>
                    # print("dist.shape:")
                    # print(dist.shape)  # torch.Size([100, 197])
                    # print(dist)
                    idx = dist.sort(1)[1].cpu().numpy()  # 取距离最短的  小-->大  取出来的应该是每行最近的对应的索引
                    # print(dist.sort(1)[0: 5])  这个才是距离
                    # print("idx:")
                    # print(type(idx))  # <class 'numpy.ndarray'>
                    # print(idx)
                    # print(idx.shape)  # (100, 197)
                    # print(dist.shape[0])  # 100

                    # print(np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0, 0))  # [80 82 85 86 90]
                    #                n行1列                              n行5列
                    # print("probe_y.shape:")
                    # print(probe_y.shape)  # (100,)
                    # print("gallery_y.shape:")
                    # print(gallery_y.shape)  # (200,)
                    # print(type(gallery_y))
                    # print(gallery_y)  # (200, 1)
                    #      （100，1）                          （100， 5）
                    a = np.reshape(probe_y, [-1, 1]) == gallery_y[  idx[:, 0:num_rank]  ]  # 这句可以用来推理？？
                    # print(gallery_y[idx[:, 0:num_rank]])  # (100,5)
                    # print("idx[:, 0:5]:")
                    # print(type(idx[:, 0:num_rank]))
                    # print(idx[:, 0:num_rank].shape)  # (100,5)
                    # print(idx[:, 0:num_rank])
                    # 打印100个标签
                    # print(probe_y)
                    # 打印最近最近的特征向量对应的标签
                    # print(np.reshape(gallery_y[idx[:, 0]], [1, -1]))

                    # print("probe_y:")
                    # print(probe_y)
                    # print("np.reshape(probe_y, [-1, 1]):")
                    # print(np.reshape(probe_y, [-1, 1]))  # (100,1)
                    #
                    # print("gallery_y:")
                    # print(gallery_y)
                    # print("gallery_y[idx[:, 0:num_rank]]:")
                    # print(gallery_y[idx[:, 0:num_rank]])  # (108,5)
                    # print("a:")
                    # print(a.shape)
                    # print(a)  # (100, 5)
                    b = np.cumsum(a, 1)  # axis = 1  行累加
                    # print("b:")
                    # print(b.shape)  # (100,5)  某一列大于0的个数
                    # print(b)
                    c = np.sum(b > 0, 0)  # axis = 0
                    # print("c:")
                    # print(c.shape)
                    # print(c)  # [80 82 85 86 90]  # 可以判定probe[i]就是90对应的那个gallery[j]?
                    acc[p, v1, v2, :] = np.round(c * 100 / dist.shape[0], 2)

                    # acc[p, v1, v2, :] = np.round(
                    #     np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                    #            0) * 100 / dist.shape[0], 2)
                    # print(acc[p, v1, v2, :])  # [80.82.85.86.90.]
                    # print(acc[p, v1, v2, :].shape)  # 5

                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']000  100 vs 200
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']018
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']036

    # print(cnt)
    return acc

def evaluation_my_dataset(data, config):
    dataset = config['dataset'].split('-')[0]  # CASIA
    feature, view, seq_type, label = data  # feature为特征向量
    # print(feature.shape)  # 5485*15872  5485个视频 每个视频的特征向量为15872个元素的向量
    # print("type(feature)：")
    # print(type(feature))  # <class 'numpy.ndarray'>
    # print(feature)
    label = np.array(label)
    # print("label:")
    # print(len(label))  # 5485  --> 说明一共有5485个视频
    # print(len(set(label)))  # 50 --> 验证集50个人
    view_list = list(set(view))  # 每个人的每种行走状态11个角度
    view_list.sort()
    view_num = len(view_list)

    # print("view_list:")
    # print(view_list)  #   # 11个 ['000', '018', '036', '054', '072', '090' ... ]
    # print("seq_type:")
    # print(list(set(seq_type)))  # 10个 ['cl-01', 'nm-01', 'nm-06', 'bg-02', 'nm-05', 'nm-04', 'bg-01', 'nm-03', 'cl-02', 'nm-02']

    # sample_num = len(feature)
    # print("sample_num", end=" ")
    # print(sample_num)  # 5485
    #  这里修改一下划分形式  ************************
    probe_seq_dict = {'CASIA': [['nm-03', 'nm-04']],
                      'OUMVLP': [['00']]}
    gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02']],
                        'OUMVLP': [['01']]}

    num_rank = 5
    acc = np.zeros(  [   len(probe_seq_dict[dataset]), view_num, view_num, num_rank  ]    )  # (3,11,11,5)

    # cnt = 0
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        # print("----1-----")
        # print(p)  # 0 1
        # print(probe_seq)  # ['nm-05', 'nm-06']  //   ['bg-01', 'bg-02']

        for gallery_seq in gallery_seq_dict[dataset]:
            # print("----2-----")
            # print(gallery_seq)  # ['nm-01', 'nm-02', 'nm-03', 'nm-04']  //   ['nm-01', 'nm-02', 'nm-03', 'nm-04']

            for (v1, probe_view) in enumerate(view_list):  # # 对应一个探针视频
                # print("----3-----")
                # print(v1)  # 0 1 2...
                # print(probe_view)  # 000 018 036...

                for (v2, gallery_view) in enumerate(view_list):  # 对应一个步态库视频

                    # print(str(probe_seq) + probe_view + " vs " + str(gallery_seq) + gallery_view)
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']000  100 vs 200
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']018
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']036
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']054
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']072
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']090
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']108
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']126
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']144
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']162
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']180
                    # cnt += 1  循环了363次
                    # print("----4-----")
                    # print(v2)  # 0 1 2...
                    # print(gallery_view)  # 000 018 036...

                    # codes above work to iter, down to calculate the distance
                    ################### ？？？？？？？？？？？？
                    # np.isin(a,b) 用于判定a中的元素在b中是否出现过，如果出现过返回True,否则返回False,
                    # 最终结果为一个形状和a一模一样的数组。
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])  # 看哪些行走类型以及角度出现过
                    # 用来挑出当前步态库中视频对应的特征向量  类似于子网掩码的作用
                    # print("gallery_seq:")
                    # print(gallery_seq)  # ['nm-01', 'nm-02', 'nm-03', 'nm-04']  4个
                    # print("[gallery_view]:")
                    # print([gallery_view])  # ['018'] .....  1个
                    # print("gseq_mask：")
                    # print(gseq_mask)  # [False False False ... False False False]
                    # print("np.sum(gseq_mask):")
                    # print(np.sum(gseq_mask))  # 200左右
                    # print(gseq_mask.shape)  # (5485,)
                    gallery_x = feature[gseq_mask, :]  # 2-D metric  当前步态库视频对应的特征向量
                    # 挑出步态库中的200个满足名字中包括'['nm-01', 'nm-02', 'nm-03', 'nm-04']000'的视频的特征向量
                    # print("gallery_x.shape：")
                    # print(gallery_x.shape)  # 200*15872
                    print("gallery_x：")
                    print(gallery_x)
                    gallery_y = label[gseq_mask]  # 1-D list
                    # 挑出步态库中的200个满足名字中包括'['nm-01', 'nm-02', 'nm-03', 'nm-04']000'的标签
                    # print("gallery_y.shape", end=" ")
                    # print(gallery_y.shape)  # 200,
                    print("gallery_y", end=" ")
                    print(gallery_y)


                    #################### ？？？？？？？？？？
                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                    # 用来挑出探针视频对应的特征向量
                    # print("pseq_mask：")
                    # print(pseq_mask)  # [False False False ... False False False]
                    # print("np.sum(pseq_mask):")
                    # print(np.sum(pseq_mask))  # 100左右
                    # print(pseq_mask.shape)  # (5485,)
                    probe_x = feature[pseq_mask, :]  # 2-D metric  当前探针视频对应的特征向量
                    # 挑出探针中的100个满足名字中包括'['nm-05', 'nm-06']000'的视频的特征向量
                    # print("probe_x.shape", end=" ")
                    # print(probe_x.shape)  # 100*15872
                    print("probe_x", end=" ")
                    print(probe_x)
                    probe_y = label[pseq_mask]  # 1-D list
                    # 挑出探针中的100个满足名字中包括'['nm-05', 'nm-06']000'的视频的标签(人)
                    # print("probe_y.shape", end=" ")
                    # print(probe_y.shape)  # 100,
                    print("probe_y", end=" ")
                    print(probe_y)

                    dist = cuda_dist(probe_x, gallery_x)  # tensor   找出每行距离最小的
                    # print("type(dist):")
                    # print(type(dist))  # <class 'torch.Tensor'>
                    # print("dist.shape:")
                    # print(dist.shape)  # torch.Size([100, 197])
                    # print(dist)
                    idx = dist.sort(1)[1].cpu().numpy()  # 取距离最短的  小-->大  取出来的应该是每行最近的对应的索引
                    # print(dist.sort(1)[0: 5])  这个才是距离
                    # print("idx:")
                    # print(type(idx))  # <class 'numpy.ndarray'>
                    # print(idx)
                    # print(idx.shape)  # (100, 197)
                    # print(dist.shape[0])  # 100

                    # print(np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0, 0))  # [80 82 85 86 90]
                    #                n行1列                              n行5列
                    # print("probe_y.shape:")
                    # print(probe_y.shape)  # (100,)
                    # print("gallery_y.shape:")
                    # print(gallery_y.shape)  # (200,)
                    # print(type(gallery_y))
                    # print(gallery_y)  # (200, 1)
                    #      （100，1）                          （100， 5）
                    a = np.reshape(probe_y, [-1, 1]) == gallery_y[  idx[:, 0:num_rank]  ]  # 这句可以用来推理？？
                    # print(gallery_y[idx[:, 0:num_rank]])  # (100,5)
                    # print("idx[:, 0:5]:")
                    # print(type(idx[:, 0:num_rank]))
                    # print(idx[:, 0:num_rank].shape)  # (100,5)
                    # print(idx[:, 0:num_rank])
                    # 打印100个标签
                    # print(probe_y)
                    # 打印最近最近的特征向量对应的标签
                    # print(np.reshape(gallery_y[idx[:, 0]], [1, -1]))

                    # print("probe_y:")
                    # print(probe_y)
                    # print("np.reshape(probe_y, [-1, 1]):")
                    # print(np.reshape(probe_y, [-1, 1]))  # (100,1)
                    #
                    # print("gallery_y:")
                    # print(gallery_y)
                    # print("gallery_y[idx[:, 0:num_rank]]:")
                    # print(gallery_y[idx[:, 0:num_rank]])  # (108,5)
                    # print("a:")
                    # print(a.shape)
                    # print(a)  # (100, 5)
                    b = np.cumsum(a, 1)  # axis = 1  行累加
                    # print("b:")
                    # print(b.shape)  # (100,5)  某一列大于0的个数
                    # print(b)
                    c = np.sum(b > 0, 0)  # axis = 0
                    # print("c:")
                    # print(c.shape)
                    # print(c)  # [80 82 85 86 90]  # 可以判定probe[i]就是90对应的那个gallery[j]?
                    acc[p, v1, v2, :] = np.round(c * 100 / dist.shape[0], 2)

                    # acc[p, v1, v2, :] = np.round(
                    #     np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                    #            0) * 100 / dist.shape[0], 2)
                    # print(acc[p, v1, v2, :])  # [80.82.85.86.90.]
                    # print(acc[p, v1, v2, :].shape)  # 5

                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']000  100 vs 200
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']018
                    # ['nm-05', 'nm-06']000 vs ['nm-01', 'nm-02', 'nm-03', 'nm-04']036

    # print(cnt)
    print(acc)
    return acc
