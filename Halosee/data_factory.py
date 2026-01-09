import os

from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from Halosee.computing_scaler import check_and_build_scaler
from Halosee.data_loader import CustomAnomalyDataset, BatteryAnomalyDataset, MultiFileBatteryDataset, \
    SingleFileBatteryDataset
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, \
    Dataset_Custom, Dataset_PEMS, UCRAnomalyloader
from data_provider.data_loader_benchmark import CIDatasetBenchmark, \
    CIAutoRegressionDatasetBenchmark

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
    # 'UCRA': UCRAnomalyloader,
    'UCRA': CustomAnomalyDataset,
}


def data_provider(args, flag):
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'forecast':
        if args.use_ims:
            data_set = CIAutoRegressionDatasetBenchmark(
                root_path=os.path.join(args.root_path, args.data_path),
                flag=flag,
                input_len=args.seq_len,
                label_len=args.label_len,
                pred_len=args.output_len if flag == 'test' else args.pred_len,
                data_type=args.data,
                scale=True,
                timeenc=timeenc,
                freq=args.freq,
                stride=args.stride,
                subset_rand_ratio=args.subset_rand_ratio,
            )
        else:
            data_set = CIDatasetBenchmark(
                root_path=os.path.join(args.root_path, args.data_path),
                flag=flag,
                input_len=args.seq_len,
                pred_len=args.pred_len,
                data_type=args.data,
                scale=True,
                timeenc=timeenc,
                freq=args.freq,
                stride=args.stride,
                subset_rand_ratio=args.subset_rand_ratio,
            )
        print(flag, len(data_set))
        if args.use_multi_gpu:
            train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
            data_loader = DataLoader(data_set,
                                     batch_size=args.batch_size,
                                     sampler=train_datasampler,
                                     num_workers=args.num_workers,
                                     persistent_workers=True,
                                     pin_memory=True,
                                     drop_last=False,
                                     )
        else:
            data_loader = DataLoader(
                data_set,
                batch_size=args.batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=False)
        return data_set, data_loader

    elif args.task_name == 'anomaly_detection':
        drop_last = False

        # 单文件训练
        # data_set = BatteryAnomalyDataset(
        #     root_path=args.root_path,
        #     data_path=args.data_path,
        #     seq_len=args.seq_len,
        #     patch_len=args.patch_len,
        #     pred_len=args.pred_len,
        #     flag=flag,
        # )

        # 多文件训练
        data_set = MultiFileBatteryDataset(
            root_path=args.root_path,
            data_path=args.data_path,
            seq_len=args.seq_len,
            patch_len=args.patch_len,
            pred_len=args.pred_len,
            flag=flag,
            cut_data=args.cut_data,  # 是否进行测试集切分
        )
        print(f">>> 数据集信息 , flag:{flag}, 数据集长度(窗口):{len(data_set)}")

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True,
        )
        # 打印batch示例
        # for batch_data in data_loader:
        #     if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
        #         x_batch, y_batch = batch_data
        #     elif isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
        #         x_batch = batch_data[0]
        #         y_batch = batch_data[1]
        #     else:
        #         # 根据实际数据结构进行处理
        #         x_batch, y_batch = batch_data[0], batch_data[1]
        #     print(f">>> 数据加载器-数据batch形状：x={x_batch.shape}，y={y_batch.shape}")
        #     break
        return data_set, data_loader
    elif args.task_name == 'imputation':
        Data = data_dict[args.data]
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    else:
        raise NotImplementedError


def anomaly_data_provider(args, flag):
    if flag == 'test':
        shuffle_flag = False
        batch_size = 1  # bsz=1 for evaluation
    else:
        shuffle_flag = True
        batch_size = args.batch_size  # bsz for train and valid
    drop_last = False

    # --- 准备全局 Scaler ---
    scaler_save_path = args.scaler_save_path
    # 调用辅助函数
    global_scaler = check_and_build_scaler(args.root_path, scaler_save_path, args.data_path)

    # 多文件
    print("正在初始化 Dataset 列表...")
    dataset_list = []
    if args.data_path == "ALL":
        file_names = sorted([f for f in os.listdir(args.root_path) if f.endswith('.txt')])
    else:
        file_names = [args.data_path]

    valid_files_count = 0
    n_feature = 0
    for f_name in file_names:
        f_path = os.path.join(args.root_path, f_name)

        # 实例化单个 Dataset
        ds = SingleFileBatteryDataset(
            file_path=f_path,
            scaler=global_scaler,  # 传入全局scaler
            seq_len=args.seq_len,
            patch_len=args.patch_len,
            pred_len=args.pred_len,
            flag=flag,
            cut_data=args.cut_data
        )

        # 只有当文件有效（长度足够生成至少一个样本）时才加入
        if len(ds) > 0:
            dataset_list.append(ds)
            valid_files_count += 1
            # 获取特征数（取第一个有效文件的特征数即可）
            if valid_files_count == 1:
                n_feature = ds.n_features

    print(f"成功加载 {valid_files_count} 个文件的数据集。")
    if not dataset_list:
        raise RuntimeError("没有加载到任何有效的数据，请检查路径或数据长度。")

    # --- 使用 ConcatDataset 逻辑拼接 ---
    train_data = ConcatDataset(dataset_list)
    print(f"ConcatDataset 组装完成，总样本数: {len(train_data)}")

    # --- 构建 DataLoader ---
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle_flag,  # 关键：全局打乱，混合不同文件的数据
        num_workers=args.num_workers,
        drop_last=drop_last,  # 保留最后一个Batch
        pin_memory=True
    )

    return train_data, train_loader, n_feature
