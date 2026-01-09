import os

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class CustomAnomalyDataset(Dataset):
    def __init__(self, root_path, data_path, seq_len, patch_len, pred_len, flag="train"):
        self.root_path = root_path
        self.data_path = data_path
        self.seq_len = seq_len
        self.input_len = seq_len - patch_len
        self.patch_len = patch_len
        self.pred_len = pred_len
        self.flag = flag
        self.stride = 1 if self.flag == "train" else self.seq_len - 2 * self.patch_len
        self.dataset_file_path = os.path.join(self.root_path, self.data_path)
        data_list = []

        print("开始加载数据集，原始文件: ", self.data_path)

        # --- 原始数据读取逻辑保持不变 ---
        assert self.dataset_file_path.endswith('.txt')
        try:
            with open(self.dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    data_list.append(data_line)
            self.data = np.stack(data_list, 0)
        except ValueError:
            with open(self.dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line[0].split()])
            self.data = data_line
            self.data = np.expand_dims(self.data, axis=1)

        print("数据集大小：", self.data.shape)

        self.n_features = self.data.shape[1]
        print("特征数量：", self.n_features)

        # 归一化处理
        # 数据字段 [时间，车厢， vcb状态， 温度， 放电时长， 电压]
        # 做全局归一化
        # 归一化范围 [0, 1]
        self.scaler = StandardScaler()
        self.scaler.fit(self.data)

        # 核心修复：检查方差，如果方差为0（常数列），将 scale 设为 1，避免除以0
        # scaler.scale_ 存储的是标准差 (std)
        self.scaler.scale_ = np.where(self.scaler.scale_ == 0, 1.0, self.scaler.scale_)

        self.data = self.scaler.transform(self.data)

        print("数据集归一化完成")

        # 计算有效窗口数
        self.valid_windows = max(0, len(self.data) - self.seq_len - self.pred_len + 1)
        if self.valid_windows == 0:
            raise ValueError(f"❌ 有效窗口数为0！数据长度{len(self.data)} ≥ {self.seq_len}+{self.pred_len}-1")
        print(f"   - 有效滑动窗口：{self.valid_windows}（seq_len={seq_len}, pred_len={pred_len}）")

    def __len__(self):
        return self.valid_windows

    # def __getitem__(self, index):
    #     train_x = self.data[index: index + self.seq_len]
    #     train_y = self.data[index + self.seq_len: index + self.seq_len + self.pred_len]
    #     # print(train_x.shape, train_y.shape)
    #     return train_x, train_y

    def __getitem__(self, index):
        x = self.data[index: index + self.seq_len]
        return x, x


class BatteryAnomalyDataset(Dataset):
    def __init__(self, root_path, data_path, seq_len, patch_len, pred_len, flag="train"):
        self.root_path = root_path
        self.data_path = data_path
        self.seq_len = seq_len
        self.input_len = seq_len - patch_len
        self.patch_len = patch_len
        self.pred_len = pred_len
        self.flag = flag
        self.stride = 1 if self.flag == "train" else self.seq_len - 2 * self.patch_len
        self.dataset_file_path = os.path.join(self.root_path, self.data_path)
        self.train_ratio = 0.7
        data_list = []

        print("开始加载数据集，原始文件: ", self.data_path)

        # --- 原始数据读取逻辑保持不变 ---
        assert self.dataset_file_path.endswith('.txt')

        print("数据集文件：", self.dataset_file_path)
        try:
            with open(self.dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    data_list.append(data_line)
            self.data = np.stack(data_list, 0)
        except ValueError:
            with open(self.dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line[0].split()])
            self.data = data_line
            self.data = np.expand_dims(self.data, axis=1)

        self.n_features = self.data.shape[1]

        # 数据划分与标准化
        self.border = int(len(self.data) * self.train_ratio)
        self.scaler = StandardScaler()
        # 仅利用训练集拟合Scaler，防止数据泄露
        self.scaler.fit(self.data[:self.border])
        self.data = self.scaler.transform(self.data)
        if self.flag == "train":
            print("保存归一化参数到 scaler.pkl ...")
            joblib.dump(self.scaler, 'scaler.pkl')
            self.data = self.data[:self.border]
        else:
            self.data = self.data[self.border - self.patch_len:]

        print("数据集大小：", self.data.shape)

        self.start_indices = []
        for i in range(0, len(self.data) - self.seq_len + 1, self.stride):
            self.start_indices.append(i)

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start = self.start_indices[idx]
        x = self.data[start:start + self.seq_len]
        # x: [T, C]
        return torch.tensor(x, dtype=torch.float32)


class MultiFileBatteryDataset(Dataset):
    def __init__(self, root_path, data_path, seq_len, patch_len, pred_len, flag="train", cut_data=False):
        self.root_path = root_path
        self.data_path = data_path
        self.seq_len = seq_len
        self.input_len = seq_len - patch_len
        self.patch_len = patch_len
        self.pred_len = pred_len
        self.flag = flag
        self.stride = 1 if self.flag == "train" else self.seq_len - 2 * self.patch_len

        # --- 1. 处理多文件逻辑 ---
        if self.data_path == "ALL":
            file_names = sorted([f for f in os.listdir(self.root_path) if f.endswith('.txt')])
        else:
            file_names = [self.data_path]

        # --- 2. 修正 train_ratio 逻辑 ---
        # 如果 cut_data 为 True，说明我们需要对数据进行分割（前90%或后10%）
        # 无论 flag 是 train 还是 test，只要 cut_data=True，ratio 都应该是 0.9，这样 border 才能算对
        if cut_data:
            self.train_ratio = 0.9
        else:
            # 如果不剪切（全量推理），ratio 为 1.0
            self.train_ratio = 1.0

        self.pkl_file_path = r"D:\Work\LLM\GitHub\Large-Time-Series-Model\Halosee\scaler_all_2.pkl"

        self.scaler_params = None

        # 用于临时存储每个文件处理后的 numpy 数组
        all_files_data = []

        print("开始加载数据集，原始文件夹: ", self.root_path)
        print(f"待处理文件列表 (共{len(file_names)}个): {file_names}")

        for file_name in file_names:
            current_file_path = os.path.join(self.root_path, file_name)

            # --- 原有读取逻辑保留，改为针对 current_file_path 操作 ---
            data_list = []  # 重置当前文件的 list
            try:
                with open(current_file_path, "r", encoding='utf-8') as f:
                    for line in f.readlines():
                        line = line.strip('\n').split(',')
                        data_line = np.stack([float(i) for i in line])
                        data_list.append(data_line)
                file_data = np.stack(data_list, 0)  # 当前文件的 array
            except ValueError:
                # 处理单行空格分割的情况
                with open(current_file_path, "r", encoding='utf-8') as f:
                    for line in f.readlines():
                        line = line.strip('\n').split(',')
                        data_line = np.stack([float(i) for i in line[0].split()])
                file_data = data_line
                file_data = np.expand_dims(file_data, axis=1)

            # 将处理好的当前文件数据加入总列表
            all_files_data.append(file_data)

        # --- 合并所有文件数据 ---
        # 将所有文件的由 [T, C] 组成的列表在时间维度(axis=0)上合并
        # --- 4. 合并数据 ---
        self.data = np.concatenate(all_files_data, axis=0)
        print("合并后数据集大小：", self.data.shape)

        self.n_features = self.data.shape[1]

        # 计算分割线
        self.border = int(len(self.data) * self.train_ratio)

        # --- 5. 归一化与数据切分 (核心修改部分) ---
        if self.flag == "train":
            self.scaler = StandardScaler()
            # 训练集：利用前 90% 拟合
            fit_border = self.border if cut_data else len(self.data)
            self.scaler.fit(self.data[:fit_border])

            print(f"保存归一化参数到 {self.pkl_file_path} ...")
            joblib.dump(self.scaler, self.pkl_file_path)

            # 全量转换
            self.data = self.scaler.transform(self.data)

            # 如果需要剪切，只保留前 90%
            if cut_data:
                self.data = self.data[:self.border]

        else:
            # 测试/推理模式
            if os.path.exists(self.pkl_file_path):
                self.scaler = joblib.load(self.pkl_file_path)
            else:
                raise FileNotFoundError(f"归一化参数文件 {self.pkl_file_path} 不存在！")

            # 步骤 A: 必须先对【所有数据】进行归一化，防止切片后漏掉
            self.data = self.scaler.transform(self.data)

            # 步骤 B: 归一化后，再根据需求进行切片
            if cut_data:
                # 场景：测试集验证（取后10%）
                # 回退 patch_len 是为了保证滑动窗口在边界处能生成完整的序列
                self.data = self.data[self.border - self.patch_len:]
            else:
                # 场景：全量推理（新文件），保留所有数据
                # falg=text cut_data=False
                pass

        # 获取归一化参数
        self.scaler_params = self.scaler.get_params()

        print("合并后数据集总大小：", self.data.shape, "self.seq_len: ", self.seq_len, "self.stride: ", self.stride)

        self.start_indices = []
        for i in range(0, len(self.data) - self.seq_len + 1, self.stride):
            self.start_indices.append(i)

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start = self.start_indices[idx]
        # start = self.start_indices[idx]
        x = self.data[start:start + self.seq_len]
        # x: [T, C]
        return torch.tensor(x, dtype=torch.float32)


class SingleFileBatteryDataset(Dataset):
    def __init__(self, file_path, scaler, seq_len, patch_len, pred_len, flag="train", cut_data=False):
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.pred_len = pred_len
        self.flag = flag
        self.stride = 1 if self.flag == "train" else self.seq_len - 2 * self.patch_len
        self.file_path = file_path

        # --- 1. 读取单个文件 (保留你原有的容错逻辑) ---
        data_list = []
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    data_list.append(data_line)
            self.data = np.stack(data_list, 0)
        except ValueError:
            # 处理空格分割的情况
            with open(file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line[0].split()])
            self.data = data_line
            if self.data.ndim == 1:
                self.data = np.expand_dims(self.data, axis=1)

        self.n_features = self.data.shape[1]

        # --- 2. 归一化 (关键修改) ---
        # 必须使用外部传入的 scaler，保证所有文件缩放尺度一致
        if scaler is not None:
            self.data = scaler.transform(self.data)
        else:
            raise ValueError("Error: Scaler is None. 必须传入全局Scaler。")

        # --- 3. 数据切分 (Train/Test Split) ---
        # 计算分割线 (前90%训练，后10%测试)
        self.border = int(len(self.data) * 0.9)

        if cut_data:
            if self.flag == "train":
                self.data = self.data[:self.border]
            else:
                # 测试模式：保留后10%，并回退 patch_len 保证边界完整
                self.data = self.data[self.border - self.patch_len:]
        else:
            # 如果不切分（全量使用），保持原样
            pass

        # --- 4. 生成索引 ---
        self.start_indices = []
        # 只有当数据长度大于序列长度时才能生成样本
        if len(self.data) > self.seq_len:
            for i in range(0, len(self.data) - self.seq_len + 1, self.stride):
                self.start_indices.append(i)

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start = self.start_indices[idx]
        x = self.data[start: start + self.seq_len]
        return torch.tensor(x, dtype=torch.float32)

