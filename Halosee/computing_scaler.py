import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

def check_and_build_scaler(root_path, save_path, data_path="ALL"):
    """
    检查是否存在scaler文件，不存在则遍历所有数据计算并保存。
    返回加载好的 scaler 对象。
    """
    if os.path.exists(save_path):
        print(f"Loading existing scaler from {save_path}")
        scaler = joblib.load(save_path)
        return scaler

    print(">>> 全局 Scaler 未找到，开始计算（这可能需要一点时间）...")

    # 获取文件列表
    if data_path == "ALL":
        file_names = sorted([f for f in os.listdir(root_path) if f.endswith('.txt')])
    else:
        file_names = [data_path]

    all_data_list = []

    print(f"正在读取 {len(file_names)} 个文件以计算均值和方差...")
    for f_name in file_names:
        current_file_path = os.path.join(root_path, f_name)
        # 简化的读取逻辑，仅用于计算scaler
        try:
            with open(current_file_path, "r", encoding='utf-8') as f:
                # 为了节省内存，可以每隔几行采样，或者全量读取
                # 这里演示全量读取，如果内存爆了，可以改为采样
                temp_list = []
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    temp_list.append([float(i) for i in line])
                file_data = np.array(temp_list)
        except:
            # 简单的fallback读取
            with open(current_file_path, "r", encoding='utf-8') as f:
                line = f.readline().strip('\n').split(',')  # 只读第一行判断维度
                # 这里为了简单，如果格式复杂建议复用Dataset读取逻辑
                continue

        all_data_list.append(file_data)

    # 拼接用于fit的数据
    concat_data = np.concatenate(all_data_list, axis=0)
    print(f"Scaler 拟合数据形状: {concat_data.shape}")

    scaler = StandardScaler()
    # 假设前90%是训练数据，我们只用这部分来Fit Scaler，防止测试集泄露
    train_border = int(len(concat_data) * 0.9)
    scaler.fit(concat_data[:train_border])

    # 保存
    joblib.dump(scaler, save_path)
    print(f"全局 Scaler 已保存至: {save_path}")

    return scaler