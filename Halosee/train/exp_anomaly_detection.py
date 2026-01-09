import datetime

import swanlab
import torch.multiprocessing
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from Halosee.data_factory import data_provider, anomaly_data_provider
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_basic import Exp_Basic
from utils.tools import adjust_learning_rate, visual

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# 建议：如果可以在 main 函数中 init 更好，这里为了保持结构不动，保留在此
# 这里的 config 稍后会在类中被 args 覆盖更新
try:
    if not swanlab.get_run():  # 防止重复初始化
        swanlab.init(
            # 设置将记录此次运行的项目信息
            project="xudainchi0108",
            workspace="DawsonPeres",
            # 跟踪超参数和运行元数据
            config={
                "learning_rate": 0.02,
                "architecture": "CNN",
                "dataset": "CIFAR-100",
                "epochs": 10
            }
        )
except Exception as e:
    print(f"SwanLab init warning: {e}")


def visual(true, preds=None, name='./pic/test.pdf'):
    folder = os.path.dirname(name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.figure(figsize=(12, 6))
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=1)
    plt.plot(true, label='GroundTruth (Voltage)', linewidth=1, alpha=0.7)
    plt.legend()
    plt.title("Anomaly Detection Result")
    plt.savefig(name, bbox_inches='tight')
    plt.close()


class Custom_Exp_Anomaly_Detection(Exp_Anomaly_Detection):
    def __init__(self, args):
        # args.enc_in = 5
        # args.c_out = 5
        super(Custom_Exp_Anomaly_Detection, self).__init__(args)
        self.THRESHOLD_PERCENTILE = 95

        # === SwanLab 修改点 1: 更新配置 ===
        # 将实际运行的 args 参数更新到 SwanLab 面板中
        try:
            swanlab.config.update(vars(args))
        except Exception as e:
            print(f"SwanLab config update failed: {e}")

    def _get_data(self, flag):
        data_set, data_loader, n_features = anomaly_data_provider(self.args, flag)
        return data_set, data_loader, n_features

    def calculate_global_mse_threshold(self, normal_loader):
        """仅计算全局MSE的95分位数阈值"""
        self.model.eval()
        all_global_mse = []

        # with torch.no_grad():
        #     for batch_x, batch_y in normal_loader:
        #         batch_x = batch_x.to(self.device, non_blocking=True)
        #         batch_y = batch_y.to(self.device, non_blocking=True)

        with torch.no_grad():
            for batch_data in normal_loader:  # 修改这里，接收整个batch
                # 检查batch_data的结构
                if isinstance(batch_data, (list, tuple)):
                    if len(batch_data) >= 2:
                        batch_x, batch_y = batch_data[0], batch_data[1]
                    else:
                        batch_x = batch_data[0]
                        batch_y = batch_x  # 如果只有输入，异常检测通常输入和目标是相同的
                else:
                    # 如果是单个tensor，假设输入和目标相同
                    batch_x = batch_data
                    batch_y = batch_x

                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                # # 生成辅助输入
                # x_mark_enc, x_dec, x_mark_dec = self.get_timer_aux_inputs(
                #     batch_x, self.args.seq_len, self.args.pred_len, n_features
                # )
                # # 模型推理
                # if self.args.use_amp and torch.cuda.is_available():
                #     with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                #         outputs = self.model(x_enc=batch_x, x_mark_enc=x_mark_enc, x_dec=x_dec, x_mark_dec=x_mark_dec)
                # else:
                #     outputs = self.model(x_enc=batch_x, x_mark_enc=x_mark_enc, x_dec=x_dec, x_mark_dec=x_mark_dec)

                outputs = self.model(batch_x, None, None, None)

                if outputs.shape[1] > self.args.pred_len:
                    outputs = outputs[:, -self.args.pred_len:, :]



                # 计算全局MSE（n_feature列整体平均）
                mse_per_sample = torch.mean((outputs - batch_y) ** 2, dim=(1, 2)).cpu().numpy()
                all_global_mse.extend(mse_per_sample)

        # 计算95分位数
        global_mse_mean = np.mean(all_global_mse)
        global_mse_std = np.std(all_global_mse)
        global_mse_threshold = np.percentile(all_global_mse, self.THRESHOLD_PERCENTILE)

        # 打印阈值结果
        print(f"\n========== 全局MSE阈值（95分位数）==========")
        print(f"📌 全局MSE均值：{global_mse_mean:.6f}")
        print(f"📌 全局MSE标准差：{global_mse_std:.6f}")
        print(f"📌 95分位数阈值：{global_mse_threshold:.6f}")
        print("============================================")

        return global_mse_threshold

    def finetune(self, setting):
        args = self.args
        print(f"开始进行模型微调，数据文件：{args.data_path}，模型：{args.model}，设备：{args.devices}")
        print(args)
        # self.args.cut_data = True
        train_data, train_loader, n_feature = self._get_data(flag='train')

        # 模型初始化
        self.device = torch.device(f'cuda:{args.gpu}' if args.use_gpu and torch.cuda.is_available() else 'cpu')

        self.model = self.model.to(self.device)

        # 改为实际数据的列数，也就是shape[1]
        # n_feature = train_data.n_features

        # 验证设备
        for name, param in self.model.named_parameters():
            if param.device != self.device:
                param.data = param.data.to(self.device)

        self.criterion = nn.MSELoss().to(self.device)
        self.optimizer = Adam(
            self.model.parameters(),
            # lr=args.learning_rate,
            lr=args.learning_rate,
            weight_decay=args.weight_decay if args.use_weight_decay else 0
        )
        # 优化器调度器配置
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=args.decay_fac,
            patience=args.patience // 2
        )

        # self.model.train()

        print("开始进行模型微调")
        print(f"   - 输入形状：(batch_size={args.batch_size}, seq_len={args.seq_len}, n_features={n_feature})")
        print(f"   - 输出形状：(batch_size={args.batch_size}, pred_len={args.pred_len}, n_features={n_feature})")
        print(f"   - 损失函数：MSE（{n_feature}列全特征重构误差）")

        self.model.train()
        train_loss = []
        best_loss = float('inf')
        best_epoch = 0
        best_model_state = None

        # 开始微调
        for epoch in range(args.train_epochs):
            self.model.train()
            epoch_loss = []

            epoch_time = time.time()
            for i, batch_x in enumerate(train_loader):
                self.optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)  # Shape: [Batch, Seq, 3]

                # 前向传播
                # 参数含义：
                # 第二个参数 x_mark_enc:
                # 形状: (batch_size, seq_len, n_time_features)
                # 用途: 输入序列的时间标记特征（如小时、星期等）
                # 第三个参数 x_dec:
                # 形状: (batch_size, pred_len, n_features)
                # 用途: 解码器输入序列（主要用于预测任务）
                # 当前设为 None 表示不是预测任务，而是重构任务
                # 第四个参数 x_mark_dec:
                # 形状: (batch_size, pred_len, n_time_features)
                # 用途: 解码器输入的时间标记特征
                # 当前设为 None 表示不使用解码器时间标记

                # 为何设为 None ：
                # 在异常检测任务中，模型只需要重构输入序列
                # batch_x，不需要额外的时间标记或解码器输入
                # 当前任务是自监督学习，目标是让模型学会重建正常模式，因此这些额外输入都设为 None

                # x_mark_enc = torch.arange(args.seq_len, dtype=torch.float32).repeat(args.batch_size, n_feature, 1).permute(0, 2, 1)
                # x_mark_enc = x_mark_enc.to(batch_x.device)
                #
                # outputs = self.model(batch_x, x_mark_enc, None, None)

                # batch_y_target = batch_y.to(outputs.device)
                #
                # loss = self.criterion(outputs, batch_y_target)

                outputs = self.model(batch_x, None, None, None)

                # outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e4, neginf=-1e4)
                # batch_x = torch.nan_to_num(batch_x, nan=0.0)

                loss = self.criterion(outputs, batch_x)

                # 反向传播
                self.optimizer.zero_grad(set_to_none=True)  # 清空梯度，减少显存碎片

                loss.backward()
                self.optimizer.step()

                # === SwanLab 修改点 2: 记录 Step 级别的 Loss ===
                # 每隔一定步数记录一次，避免日志过大，这里设为每10步或每步
                if i % 10 == 0:
                    swanlab.log({
                        "train/batch_loss": loss.item()
                    })

                epoch_loss.append(loss.item())

            # 计算本轮平均Loss
            avg_loss = np.mean(epoch_loss)
            train_loss.append(avg_loss)
            print(datetime.datetime.now())

            # 计算 Epoch 级别的指标
            avg_train_loss = np.average(train_loss)
            cost_time = time.time() - epoch_time
            current_lr = self.optimizer.param_groups[0]['lr']

            # === SwanLab 修改点 3: 记录 Epoch 级别的指标 ===
            swanlab.log({
                "train/epoch_loss": avg_train_loss,
                "train/learning_rate": current_lr,
                "train/epoch_time": cost_time,
                "epoch": epoch + 1
            })

            print(f">>> Epoch: {epoch + 1:2d}, Cost Time: {time.time() - epoch_time:.2f}s , Epoch Loss: {avg_loss:.6f}, LR: {self.optimizer.param_groups[0]['lr']:.7f}")

            # 保存最佳模型（仅Loss下降时保存，减少IO）
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch + 1
                best_model_state = self.model.state_dict().copy()
                print(f" >>> 刷新最佳Loss：{best_loss:.6f}（第{best_epoch}轮）")

            # 学习率调整
            self.scheduler.step(avg_loss)

        # 保存模型
        model_path = './checkpoints/' + setting + '/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # 计算全局MSE 95分位数阈值
        print(f"\n🔍 计算全局MSE 95分位数阈值")
        # self.args.cut_data = True
        normal_data, normal_loader, n_feature = self._get_data('train')
        global_mse_threshold = self.calculate_global_mse_threshold(normal_loader)

        print(f"全局MSE {self.THRESHOLD_PERCENTILE}分位数阈值：{global_mse_threshold:.6f}")

        # 保存模型和阈值
        os.makedirs(args.checkpoints, exist_ok=True)
        model_path = os.path.join(args.checkpoints, setting)
        save_dict = {
            'model_state_dict': best_model_state,
            'best_loss': best_loss,
            'best_epoch': best_epoch,
            'n_features': n_feature,
            'seq_len': args.seq_len,
            'pred_len': args.pred_len,
            # 'scaler': train_data.scaler,
            # 'scaler_params': train_data.scaler_params,
            'train_loss': train_loss,
            'global_mse_threshold': global_mse_threshold,
            'threshold_percentile': self.THRESHOLD_PERCENTILE,
            'early_stopped': best_epoch < args.train_epochs,
            # 'sliding_stride': args.sliding_stride,  # 保存滑窗参数
            # 'sample_ratio': args.sample_ratio  # 保存采样参数
        }
        torch.save(save_dict,  os.path.join(model_path, 'checkpoint.pth'))

        print("模型已保存。")
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader, n_feature = self._get_data(flag='test')

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()

        # [关键修改1] reduction='none' 保留每个点的误差，用于后续分析
        self.anomaly_criterion = nn.MSELoss(reduction='none').to(self.device)

        input_list = []
        output_list = []
        score_list = []  # 存储每条样本的异常分

        print("开始测试推理...")
        with torch.no_grad():
            for i, batch_x in enumerate(tqdm(test_loader)):
                batch_x = batch_x.float().to(self.device)

                # 推理
                if self.args.use_ims:
                    outputs = self.model(batch_x[:, :-self.args.patch_len, :], None, None, None)
                    batch_x = batch_x[:, self.args.patch_len:-self.args.patch_len, :]
                    outputs = outputs[:, :-self.args.patch_len, :]
                else:
                    outputs = self.model(batch_x, None, None, None)

                # [关键修改2] 收集绘图数据：不再只取第0个，而是拉平整个Batch
                # 注意：如果数据量太大导致内存溢出，可以加 if i % 10 == 0 采样
                # 这里取第2列(index 1)作为电压展示
                input_list.append(batch_x[:, :, 1].detach().cpu().numpy().reshape(-1))
                output_list.append(outputs[:, :, 1].detach().cpu().numpy().reshape(-1))

                # [关键修改3] 正确的异常分计算逻辑
                # loss shape: [Batch, Seq, Features]
                raw_loss = self.anomaly_criterion(batch_x, outputs)

                # 1. 对特征维度求均值 -> [Batch, Seq] (每个时间点的误差)
                loss_per_time = torch.mean(raw_loss, dim=-1)

                # 2. 对时间序列求均值 -> [Batch] (每条样本的平均误差)
                # 这样我们才能知道哪个样本（哪段电池放电过程）是异常的
                loss_per_sample = torch.mean(loss_per_time, dim=-1)

                # 将分数转为numpy存入列表
                scores = loss_per_sample.detach().cpu().numpy()

                # 记录索引，方便后续反查是哪一批数据
                start_idx = i * self.args.batch_size
                for idx, sc in enumerate(scores):
                    score_list.append((start_idx + idx, sc))

        # === 绘图数据整合 ===
        # 将列表中的数组拼接成一条长的一维数组
        input_viz = np.concatenate(input_list)
        output_viz = np.concatenate(output_list)

        # 限制绘图长度，防止PDF生成过慢或内存溢出（例如只画前10000个点）
        viz_limit = 10000
        if len(input_viz) > viz_limit:
            print(f"绘图数据过大，仅截取前 {viz_limit} 个点进行可视化...")
            input_viz = input_viz[:viz_limit]
            output_viz = output_viz[:viz_limit]

        data_short_name = os.path.basename(self.args.data_path).split('.')[0]
        file_path = os.path.join(folder_path, f"{data_short_name}_test_result.pdf")

        print(f"Saving visualization to: {file_path}")
        visual(input_viz, output_viz, file_path)

        # === 统计 Top Anomalies ===
        # 按分数从高到低排序
        score_list.sort(key=lambda x: x[1], reverse=True)

        print("-" * 30)
        print("Anomaly Detection Results (Top 5 Anomalous Samples):")
        for k in range(min(5, len(score_list))):
            sample_idx, sc = score_list[k]
            print(f"Rank {k + 1}: Sample Index {sample_idx}, MSE Score: {sc:.6f}")
        print("-" * 30)

        # 保存结果
        filename = 'anomaly_result.csv'
        max_score = score_list[0][1] if len(score_list) > 0 else 0
        threshold_99 = np.percentile([s[1] for s in score_list], 99) if len(score_list) > 0 else 0

        # 写入CSV
        with open(filename, 'a') as f:
            f.write(f"{self.args.data_path},{max_score},{threshold_99}\n")

        # 保存完整分数数组
        np.save(os.path.join(folder_path, 'anomaly_scores.npy'), np.array(score_list))

        return
