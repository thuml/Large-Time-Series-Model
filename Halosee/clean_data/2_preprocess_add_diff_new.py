import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def process_file(file_path, save_path, col_index):
    try:
        # 尝试读取，如果不带表头
        df = pd.read_csv(file_path, header=None)
    except Exception:
        # 处理空格分隔的情况
        df = pd.read_csv(file_path, header=None, sep='\s+')

    # 获取电压列
    voltage_series = df.iloc[:, col_index]

    # === 优化逻辑 ===
    # 1. 计算一阶差分 (当前时刻 - 上一时刻)
    # diff < 0 表示电压下降，diff > 0 表示电压回升
    raw_diff = voltage_series.diff().fillna(0)

    # 2. 计算滑动窗口均值 (趋势特征)
    # window=10: 看过去10个点的平均变化率，消除瞬间噪声
    # min_periods=1: 保证刚开始的数据也有值
    trend_feature = raw_diff.rolling(window=10, min_periods=1).mean().fillna(0)

    # 3. 增强系数
    # 因为平均后的数值很小（例如 -0.0001），乘以系数让模型更容易捕捉权重
    trend_feature = trend_feature * 100

    # 4. 拼接到最后一列
    df['Voltage_Trend'] = trend_feature

    # 保存 (去除header和index，保持纯数据格式)
    df.to_csv(save_path, header=False, index=False)


def verify_visualization(file_path, col_index):
    """验证可视化：对比原始电压 vs 计算出的趋势"""
    df = pd.read_csv(file_path, header=None)

    # 动态获取列，避免写死索引导致画错
    voltage = df.iloc[:, col_index].values
    trend = df.iloc[:, -1].values  # 刚才生成的最后一列

    # 为了画图清晰，只取前 2000 个点 (根据你的数据长度调整)
    limit = 2000
    if len(voltage) < limit:
        limit = len(voltage)

    plt.figure(figsize=(12, 8))

    # 子图1：原始电压
    plt.subplot(2, 1, 1)
    plt.plot(voltage[:limit], label='Raw Voltage', color='blue', linewidth=1.5)
    plt.title(f"Original Voltage (Column {col_index})")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 子图2：计算出的趋势
    plt.subplot(2, 1, 2)
    plt.plot(trend[:limit], label='Calculated Trend (Rate of Change)', color='red', linewidth=1.2)
    # 画一条 0 线，方便看正负
    plt.axhline(0, linestyle='--', color='black', alpha=0.7)

    plt.title("Voltage Trend (Negative = Dropping)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # 保存图片而不是单纯显示
    save_img_path = file_path.replace('.csv', '_check.png').replace('.txt', '_check.png')
    plt.savefig(save_img_path)
    print(f">>> 验证图片已保存至: {save_img_path}")
    print(">>> 观察指南：当蓝线（电压）开始往下掉时，红线应该立刻变为负数（掉到0线以下）。")
    # plt.show() # 如果需要弹窗可以取消注释


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录: {OUTPUT_DIR}")

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt') or f.endswith('.csv')]
    print(f"找到 {len(files)} 个文件，开始处理...")

    # 处理所有文件
    for file_name in tqdm(files):
        input_path = os.path.join(INPUT_DIR, file_name)
        output_path = os.path.join(OUTPUT_DIR, file_name)
        process_file(input_path, output_path, VOLTAGE_COL_INDEX)

    print("\n>>> 数据处理完成！")

    # === 验证环节 ===
    # 选取第一个生成的文件进行画图验证
    if len(files) > 0:
        sample_file = os.path.join(OUTPUT_DIR, files[0])
        verify_visualization(sample_file, VOLTAGE_COL_INDEX)


if __name__ == "__main__":
    # ================= 配置区 =================
    # 原始数据目录
    INPUT_DIR = r"D:\Work\LLM\GitHub\Large-Time-Series-Model\csv_data\3_txt_data_trend"
    # 输出数据目录
    OUTPUT_DIR = r"D:\Work\LLM\GitHub\Large-Time-Series-Model\csv_data\4_dataset_trend"

    # 【重要】请确认电压在第几列？(0代表第1列，2代表第3列，3代表第4列)
    # 假设你的电压在第4列，这里填3；如果在第3列，这里填2
    VOLTAGE_COL_INDEX = 3

    main()