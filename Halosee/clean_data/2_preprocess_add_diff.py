# 增加下降速率

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt




def process_file(file_path, save_path):
    try:
        df = pd.read_csv(file_path, header=None)
    except Exception:
        df = pd.read_csv(file_path, header=None, sep='\s+')

    # 计算电压的下降速率
    voltage_series = df.iloc[:, VOLTAGE_COL_INDEX] # 假设电压在第3列(index 2)

    # 1. 计算一阶差分 (瞬间变化)
    # 结果类似: [0, 0, -1, 0, 0, 0]
    # diff_series = voltage_series.diff().fillna(0)
    diff_series = voltage_series.diff().rolling(window=10).mean().fillna(0)

    # 2. 【关键修改】计算滑动窗口均值 (趋势特征)
    # window=6 表示看过去6个时间步的平均变化
    # 如果 seq_len=96，取 6-12 左右比较合适，能捕捉局部趋势
    # 物理含义：最近 6 个时间单位内，平均每一步下降了多少
    # 公式 ：(当前值 - 前一个值) / 前一个值
    # 10个窗口的平均值
    trend_feature = diff_series.rolling(window=10, min_periods=1).mean().fillna(0)

    # 3. 增强处理：因为平均后数值变小了，可以乘一个系数放大，让模型更容易看清
    trend_feature = trend_feature * 10

    # 4. 拼接到最后一列
    df['Voltage_Trend'] = trend_feature

    # 保存
    df.to_csv(save_path, header=False, index=False)


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录: {OUTPUT_DIR}")

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt') or f.endswith('.csv')]
    print(f"找到 {len(files)} 个文件，开始处理...")

    for file_name in tqdm(files):
        input_path = os.path.join(INPUT_DIR, file_name)
        output_path = os.path.join(OUTPUT_DIR, file_name)

        process_file(input_path, output_path)

    print("\n>>> 处理完成！")
    print(f">>> 新数据已保存在: {OUTPUT_DIR}")
    print(">>> 新数据的列数应为: 5")

    # === 验证环节：画一张图看看效果 ===
    sample_file = os.path.join(OUTPUT_DIR, files[0])
    verify_visualization(sample_file)


def verify_visualization(file_path):
    """画出电压和电压变化率的对比图，确认逻辑是否正确"""
    df = pd.read_csv(file_path, header=None)
    voltage = df.iloc[:, 2].values
    diff = df.iloc[:, -1].values  # 最后一列

    # 为了画图清晰，只取前500个点
    limit = 1000

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(voltage[:limit], label='Voltage (Original)', color='blue')
    plt.legend()
    plt.title("Original Voltage")

    plt.subplot(2, 1, 2)
    plt.plot(diff[:limit], label='Voltage Diff (New Feature)', color='red')
    # 画一条 0 线
    plt.axhline(0, linestyle='--', color='gray', alpha=0.5)
    plt.legend()
    plt.title("Voltage Rate of Change (First Derivative)")

    plt.tight_layout()
    plt.show()
    print(">>> 已生成验证图片，请检查：当电压下降时，红线是否变负？")


if __name__ == "__main__":
    # ================= 配置区 =================
    # 原始数据目录
    INPUT_DIR = r"D:\Work\LLM\GitHub\Large-Time-Series-Model\csv_data\3_txt_data"
    # 输出数据目录 (建议新建一个文件夹，不要覆盖原文件，以防万一)
    OUTPUT_DIR = r"D:\Work\LLM\GitHub\Large-Time-Series-Model\csv_data\4_dataset"

    # 电压所在的列索引 (从0开始数)
    VOLTAGE_COL_INDEX = 3
    main()