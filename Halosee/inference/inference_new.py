"""
inference_server_final.py
版本：v2.0 (适配 checkpoint 内置阈值)
功能：加载模型权重与内置阈值 -> 批量推理 -> 状态机报警 -> 生成可视化报表
"""
import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

from Halosee.data_factory import data_provider, anomaly_data_provider
from Halosee.inference.args import InferenceConfig
from Halosee.train.exp_anomaly_detection import Custom_Exp_Anomaly_Detection


# ================= 配置区 =================
# 结果保存根目录
# RESULT_DIR_BASE = "./result/2394_2/"


class InferenceServer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0' if args.use_gpu and torch.cuda.is_available() else 'cpu')

        # 1. 准备输出目录
        # 根据当前时间或数据名创建子文件夹，避免覆盖
        data_name = os.path.basename(args.root_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = os.path.join(RESULT_DIR_BASE, f"{data_name}_{timestamp}")
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        # 2. 初始化模型结构
        self.exp = Custom_Exp_Anomaly_Detection(args)
        self.model = self.exp.model.to(self.device)

        # 3. 加载权重与阈值 (核心修改)
        if os.path.exists(args.checkpoint_path):
            print(f">>> 加载Checkpoint: {os.path.basename(args.checkpoint_path)}")
            # 加载整个字典
            checkpoint = torch.load(args.checkpoint_path, map_location=self.device, weights_only=False)

            # A. 加载模型参数
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 兼容旧版本直接保存模型的情况
                self.model.load_state_dict(checkpoint)

            # B. 提取内置阈值
            if 'global_mse_threshold' in checkpoint:
                self.builtin_threshold = checkpoint['global_mse_threshold']
                print(f" >>> 成功加载内置阈值 (95%分位数): {self.builtin_threshold:.6f}")
            else:
                print(" >>> 警告: Checkpoint 中未找到 'global_mse_threshold'，使用默认值 0.1")
                self.builtin_threshold = 0.1

            # self.exp.scaler = checkpoint['scaler']
            #
            # # # [关键修改] 获取训练时的 scaler
            # # if 'scaler' in checkpoint:
            # #     self.train_scaler = checkpoint['scaler']
            # #     print("✅ 已加载训练集 Scaler，将用于测试集归一化")
            # # else:
            # #     raise ValueError("Checkpoint 中缺失 'scaler'，无法进行正确的推理！")

            # if self.args.scaler is not None:
            #     self.exp.scaler = self.args.scaler
            # else:
            #     raise ValueError("Error: Scaler is None. 必须传入全局Scaler。")

        else:
            raise FileNotFoundError(f"找不到模型权重文件: {args.checkpoint_path}")

        self.model.eval()
        # self.criterion = torch.nn.MSELoss(reduction='none')
        self.criterion = torch.nn.MSELoss()

        # 4. 设定报警策略
        # 逻辑：微调算出的阈值是 95分位数 (Warning)，为了区分严重故障，我们设定 Critical = 2 * Warning
        self.TH_WARN = self.builtin_threshold
        self.TH_CRIT = self.builtin_threshold * 2.0
        self.TRIGGER_STEPS = 3  # 连续多少个点超过阈值才报警

        self.abnormal_counter = 0
        print(f">>> 报警策略就绪: Warning(>={self.TH_WARN:.6f}), Critical(>={self.TH_CRIT:.6f})")

    def run(self):
        print(f">>> 读取推理数据: {self.args.root_path}")
        # 注意：这里务必确保 test 模式下的数据预处理(Scaler)与训练时一致
        # dataset, dataloader = data_provider(self.args, flag='test')
        dataset, dataloader, n_features = anomaly_data_provider(self.args, flag='test')

        # 数据容器
        full_gt, full_pred, full_scores = [], [], []  # 推理结果容器 分别是 真实值、预测值、分数
        alarms = []  # 报警信息

        print(">>> 开始推理...")
        with torch.no_grad():
            for i, batch_x in enumerate(tqdm(dataloader)):
                batch_x = batch_x.to(self.device)

                # 前向传播
                if self.args.use_ims:
                    # Patch模式处理
                    x = batch_x[:, :-self.args.patch_len, :]
                    outputs = self.model(x, None, None, None)
                    batch_x_input = batch_x[:, self.args.patch_len:-self.args.patch_len, :]
                else:
                    # 普通模式
                    outputs = self.model(batch_x, None, None, None)
                    batch_x_input = batch_x  # [batch_size, seq_Len, n_features]

                # 计算 Loss & Score
                # loss = self.criterion(outputs, batch_x_input)  # [batch_size, seq_Len, n_features]
                # score_map = torch.mean(loss, dim=-1)  # [batch_size, seq_Len] -> 每个时间点的MSE

                # mse_map = (outputs - batch_x_input) ** 2  # [batch_size, seq_Len, n_features]
                # score_map = mse_map.mean(dim=-1)  # [batch_size, seq_Len] -> 每个时间点的MSE
                mse_map = (outputs - batch_x_input) ** 2
                global_mse = mse_map.mean(dim=(1, 2))  # [B]  与训练一致

                # === 数据收集策略：取滑窗的最后一个点 ===
                # 假设 stride=1，取最后一个点可以拼接成完整的连续时间序列
                last_gt = batch_x_input[:, -1, :].cpu().numpy()  # [batch_size, seq_Len] -> 每个时间点的真实值
                last_pred = outputs[:, -1, :].cpu().numpy()  # [batch_size, seq_Len] -> 每个时间点的预测值
                # last_score = score_map[:, -1].cpu().numpy()  # [batch_size] -> 每个时间点的分数
                last_score = global_mse.cpu().numpy()

                full_gt.append(last_gt)
                full_pred.append(last_pred)
                full_scores.append(last_score)

                # === 实时状态机报警逻辑 ===
                # 遍历当前 batch 中的每一个样本（通常 batch size 这里的 index 代表不同时间窗口）
                # print(f"[DEBUG] 当前 batch: {i}, 样本: {last_score}")
                for idx, score in enumerate(last_score):
                    # global_idx = i * self.args.batch_size + idx  # 估算的全局时间步索引
                    global_idx = i * (self.args.seq_len - 2 * self.args.patch_len) + idx  # 估算的全局时间步索引

                    print(
                        f"[DEBUG] 时间步: {global_idx}, 状态: {self.abnormal_counter}, 阈值: {self.TH_WARN:.6f}, 评分: {score:.6f}")

                    # 状态机计数
                    if score > self.TH_WARN:
                        self.abnormal_counter += 1
                    else:
                        self.abnormal_counter = 0

                    # 触发报警
                    if self.abnormal_counter >= self.TRIGGER_STEPS:
                        level = "CRITICAL" if score > self.TH_CRIT else "WARNING"

                        # 获取关键特征值用于日志 (假设 idx 2=电压, idx 4=趋势)
                        vol = last_gt[idx, 3] if last_gt.shape[1] > 3 else 0
                        trend = last_gt[idx, 5] if last_gt.shape[1] > 5 else 0

                        alarm_msg = {
                            "id": global_idx,
                            "time": datetime.now().strftime("%H:%M:%S"),  # 模拟实时时间
                            "level": level,
                            "score": float(f"{score:.6f}"),
                            "voltage": float(f"{vol:.2f}"),
                            "trend": float(f"{trend:.4f}"),
                            "threshold": self.TH_WARN
                        }
                        alarms.append(alarm_msg)

                    # level = "CRITICAL" if score > self.TH_CRIT else "WARNING"
                    #
                    # # 获取关键特征值用于日志 (假设 idx 2=电压, idx 4=趋势)
                    # vol = last_gt[idx, 3] if last_gt.shape[1] > 3 else 0
                    # trend = last_gt[idx, 5] if last_gt.shape[1] > 5 else 0
                    #
                    # alarm_msg = {
                    #     "id": global_idx,
                    #     "time": datetime.now().strftime("%H:%M:%S"),  # 模拟实时时间
                    #     "level": level,
                    #     "score": float(f"{score:.6f}"),
                    #     "voltage": float(f"{vol:.2f}"),
                    #     "trend": float(f"{trend:.4f}"),
                    #     "threshold": self.TH_WARN
                    # }
                    # alarms.append(alarm_msg)

        # === 结果后处理 ===
        self.gt_all = np.concatenate(full_gt, axis=0)
        self.pred_all = np.concatenate(full_pred, axis=0)
        self.score_all = np.concatenate(full_scores, axis=0)

        print(f"\n>>> 推理完成. 总样本数: {len(self.score_all)}")
        print(f">>> 触发报警次数: {len(alarms)}")

        self._save_results(alarms)

    def _save_results(self, alarms):
        # 1. 保存报警日志
        if alarms:
            alarm_path = os.path.join(self.result_dir, "alarm_logs.csv")
            pd.DataFrame(alarms).to_csv(alarm_path, index=False)
            print(f"📝 报警日志已保存: {alarm_path}")
        else:
            print("🎉 本次推理未发现持续异常。")

        # 2. 保存全量数据 CSV (用于复盘分析)
        print("💾正在保存全量分析数据...")
        df_detail = pd.DataFrame()
        # 动态获取特征列名
        n_feats = self.gt_all.shape[1]
        feat_names = ["time_value", "vcb_status", "Tem", "Voltage", "time", "rate"]  # 示例列名，按需修改

        for k in range(n_feats):
            col_name = feat_names[k] if k < len(feat_names) else f"Feat{k}"
            df_detail[f'{col_name}_True'] = self.gt_all[:, k]
            df_detail[f'{col_name}_Pred'] = self.pred_all[:, k]

        df_detail['Anomaly_Score'] = self.score_all
        df_detail['Threshold'] = self.TH_WARN

        csv_path = os.path.join(self.result_dir, "detailed_metrics.csv")
        df_detail.to_csv(csv_path, index=False)

        # 3. 生成可视化
        self._plot_visualization()

    def _plot_visualization(self):
        print("📊 正在生成可视化图表...")

        # 自动截取前 5000 个点，避免图表太密集看不清
        # display_len = min(len(self.score_all), 5000)
        # display_len = max(len(self.score_all), 500)
        display_len = max(len(self.score_all), 150)

        # 定义绘图风格
        plt.style.use('seaborn-v0_8-whitegrid')

        # === 图1: 电压 (Feature 2) 重构对比 ===
        plt.figure(figsize=(15, 6))
        plt.plot(self.gt_all[:display_len, 2], label='Ground Truth (Voltage)', color='#1f77b4', linewidth=1.5,
                 alpha=0.8)
        plt.plot(self.pred_all[:display_len, 2], label='Prediction', color='#ff7f0e', linewidth=1, linestyle='--')
        plt.title(f"Voltage Reconstruction (First {display_len} steps)", fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(self.result_dir, "vis_voltage_recon.png"))
        plt.close()

        # === 图2: 异常分与报警线 ===
        fig, ax = plt.subplots(figsize=(15, 6))
        # 绘制异常分
        ax.plot(self.score_all[:display_len], label='Anomaly Score', color='black', linewidth=0.8)
        # 绘制阈值线
        ax.axhline(self.TH_WARN, color='#d62728', linestyle='--', linewidth=2,
                   label=f'Warning Thresh ({self.TH_WARN:.4f})')
        # 填充超出阈值的区域
        ax.fill_between(range(display_len), self.TH_WARN, self.score_all[:display_len],
                        where=(self.score_all[:display_len] >= self.TH_WARN),
                        color='red', alpha=0.3, interpolate=True)

        ax.set_title("Anomaly Score & Alerts", fontsize=14)
        ax.set_ylabel("MSE Loss")
        ax.legend(loc='upper left')
        plt.savefig(os.path.join(self.result_dir, "vis_anomaly_score.png"))
        plt.close()

        print(f">>> 所有结果已保存至: {self.result_dir}")


if __name__ == "__main__":
    configs = InferenceConfig()
    RESULT_DIR_BASE = configs.result_dir_path

    # 启动推理
    server = InferenceServer(configs)
    server.run()
