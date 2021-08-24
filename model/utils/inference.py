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


def infer(probe_data, data, config):
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

    sample_num = len(feature)
    # print("sample_num", end=" ")
    # print(sample_num)  # 5485

    ######## 修改中 ##############
    # print(probe_data)   # 078-nm-05-000
    # probe_seq = "nm-05"
    # probe_view = "000"
    # probe_label = "078"
    probe_seq = probe_data[4:9]
    probe_view = probe_data[10:13]
    probe_label = probe_data[0:3]
    pseq_mask = np.isin(seq_type, [probe_seq]) & np.isin(view, [probe_view]) & np.isin(label, [probe_label])  # 用来找出唯一对应的feature
    probe_x = feature[pseq_mask, :]  # 2-D metric  当前探针视频对应的特征向量  (1, 15872)
    # print("probe_x:")
    # print(probe_x)
    # probe_y = label[pseq_mask]  # 1-D list
    # a = np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]]  # 这句可以用来推理？？
    ######## 修改中 ##############


    # probe_seq_dict = {'CASIA': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
    #                   'OUMVLP': [['00']]}

    gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']]}

    num_rank = 5
    # acc = np.zeros(  [   len(probe_seq_dict[dataset]), view_num, view_num, num_rank  ]    )  # (3,11,11,5)

    # cnt = 0

    # for gallery_seq in gallery_seq_dict[dataset]:
    #     # print(gallery_seq)  # ['nm-01', 'nm-02', 'nm-03', 'nm-04']
    #     for (v2, gallery_view) in enumerate(view_list):  # 对应一个步态库视频   应该循环11次
    #         # (200, 15872)
    #         gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])  # 看哪些行走类型以及角度出现过
    #         gallery_x = feature[gseq_mask, :]  # 2-D metric  当前步态库视频对应的特征向量
    #         gallery_y = label[gseq_mask]  # 1-D list
    #         pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
    #         # probe_x = feature[pseq_mask, :]  # 2-D metric  当前探针视频对应的特征向量
    #         # probe_y = label[pseq_mask]  # 1-D list
    #
    #         dist = cuda_dist(probe_x, gallery_x)  # tensor   找出每行距离最小的
    #         # print(dist.sort(1)[0:5])
    #         idx = dist.sort(1)[1].cpu().numpy()  # (1, 200)
    #
    #         res_inf = gallery_y[  idx[:, 0:num_rank]  ] # (1, 5)
    #         print("探针视频：078-nm-05-000")
    #         print("根据"+str(gallery_seq)+"-"+str(gallery_view)+"推理结果前五名：")
    #         print(res_inf)
    #
    #         # a = np.reshape(probe_y, [-1, 1]) == gallery_y[  idx[:, 0:num_rank]  ]  # 这句可以用来推理？？
    #         # b = np.cumsum(a, 1)  # axis = 1  行累加
    #         # c = np.sum(b > 0, 0)  # axis = 0
    #         # acc[p, v1, v2, :] = np.round(c * 100 / dist.shape[0], 2)



    idx2 = []
    dist2 = []
    dic = {}
    total_order_idx = []
    # print("gallery_seq_dict[dataset]：")
    # print(gallery_seq_dict[dataset])
    # 遍历后50个人中的nm-01 ~ nm-04的视频
    for gallery_seq in gallery_seq_dict[dataset]:  # gallery_seq_dict[dataset]：[['nm-01', 'nm-02', 'nm-03', 'nm-04']]
        # print(gallery_seq)  # ['nm-01', 'nm-02', 'nm-03', 'nm-04']
        # for (v2, gallery_view) in enumerate(view_list):  # 对应一个步态库视频   应该循环11次
        # (200, 15872)
        # print(view_list)
        gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [view_list])  # 看哪些行走类型以及角度出现过
        print(len(gseq_mask))
        gallery_x = feature[gseq_mask, :]  # 2-D metric  当前步态库视频对应的特征向量
        gallery_y = label[gseq_mask]  # 1-D list
        # pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
        # probe_x = feature[pseq_mask, :]  # 2-D metric  当前探针视频对应的特征向量
        # probe_y = label[pseq_mask]  # 1-D list

        dist = cuda_dist(probe_x, gallery_x)  # tensor   找出每行距离最小的
        # print(type(dist))  # <class 'torch.Tensor'>
        # print(dist.size())  # torch.Size([1, 197])
        # print(dist)  #
        # print(dist.sort(1))  # 按照第一维排序  排序后才会显示对应的索引
        order_dist = dist.sort(1)[0].cpu().numpy()  # (1, 200)   获得的是排好序的距离列表
        idx = dist.sort(1)[1].cpu().numpy()  # (1, 200)  获得的是以距离排好序的对应的索引列表
        # print(len(order_dist[0]))  # 197
        # print(len(idx[0]))  # 197

        for i in range(len(idx[0])):
            dic[dist.sort(1)[0].cpu().numpy()[0][i]] = idx[0][i]


        idx2.extend(idx[0])
        dist2.extend(dist.sort(1)[0].cpu().numpy()[0])
        # print(idx)
        # print(dist.sort(1).cpu().numpy()[:, 0])
        # print(idx[0])  # [ 13  14  12  15 159 ... ]
        # print(idx[:, 0])  # [13]

        res_inf = gallery_y[idx[:, 0]]
        # print("探针视频：078-nm-05-000")
        # print("根据"+str(gallery_seq)+"-"+str(gallery_view)+"推理结果：")
        # print(res_inf)
        # print(gallery_y)

    # print(idx2)
    # dist2.sort()
    # print(dist2)
    # print(cnt)
    for k in sorted(dic):
        total_order_idx.append(dic[k])
        # print(k, dic[k], end=" ")
    # print(total_order_idx)
    print("根据探针视频"+probe_label+"-"+probe_seq+"-"+probe_view+"的推理结果前十位:")
    # print(gallery_y)
    # print(total_order_idx[0:10])
    print(gallery_y[total_order_idx[0:20]])
    return ""
