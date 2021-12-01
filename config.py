conf = {
    "WORK_PATH": "./work",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "data": {
        # 'dataset_path': "/home/projects/output2",
        # 'dataset_path': "/home/zc/Projects/GaitSet/home/projects/output",
        # 'dataset_path': "/home/zc/Projects/GaitSet/output",
        'dataset_path': "/workspace/projects/GaitSet/output",
        'resolution': '64',
        'dataset': 'CASIA-B',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_num': 73,  # 这里规定了测试集和训练集的边界
        'pid_shuffle': False,
    },
    "gallery_data": {
        'dataset_path': "/workspace/projects/GaitSet/data_gallery",
        'resolution': '64',
        'dataset': 'CASIA-B',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_num': 73,
        'pid_shuffle': False,
    },
    "probe_data": {
        'dataset_path': "/workspace/projects/GaitSet/data_probe",
        'resolution': '64',
        'dataset': 'CASIA-B',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_shuffle': False,
    },
    "model": {
        'hidden_dim': 256,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (8, 16),
        'restore_iter': 0,
        'total_iter': 2000,
        'margin': 0.2,
        'num_workers': 3,
        'frame_num': 30,
        'model_name': 'GaitSet',
    },
}
