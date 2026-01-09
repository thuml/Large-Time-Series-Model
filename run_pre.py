import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist

from Halosee.train.exp_anomaly_detection import Custom_Exp_Anomaly_Detection
# from Halosee.train.exp_anomaly_detection_swanlab import Custom_Exp_Anomaly_Detection
from exp.exp_forecast import Exp_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_imputation import Exp_Imputation
from utils.tools import HiddenPrints

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='大规模时间序列模型')

    # 基础配置
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='任务名称，可选值：[forecast, imputation, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='运行模式：1-训练，0-测试')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='模型标识符')
    parser.add_argument('--model', type=str, required=True, default='Timer',
                        help='模型名称，可选值：[Timer, TrmEncoder]')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')

    # 数据加载器配置
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='数据集类型')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='数据文件的根目录路径')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='数据文件名')
    parser.add_argument('--features', type=str, default='M',
                        help='预测任务类型：M-多变量预测多变量，S-单变量预测单变量，MS-多变量预测单变量')
    parser.add_argument('--target', type=str, default='OT', help='在S或MS任务中要预测的目标特征')
    parser.add_argument('--freq', type=str, default='h',
                        help='时间特征编码的频率，可选值：[s-秒级, t-分钟级, h-小时级, d-天级, b-工作日, w-周级, m-月级]，也可使用更详细的频率如15min或3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='模型检查点保存位置')
    parser.add_argument('--inverse', action='store_true', help='是否对输出数据进行逆变换', default=False)

    # 模型定义
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数量')
    parser.add_argument('--e_layers', type=int, default=2, help='编码器层数')
    parser.add_argument('--d_layers', type=int, default=1, help='解码器层数')
    parser.add_argument('--d_ff', type=int, default=2048, help='前馈网络维度')
    parser.add_argument('--factor', type=int, default=1, help='注意力因子')
    parser.add_argument('--distil', action='store_false',
                        help='是否在编码器中使用蒸馏机制，使用此参数表示不使用蒸馏',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='丢弃率')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='时间特征编码方式，可选值：[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='激活函数类型')
    parser.add_argument('--output_attention', action='store_true', help='是否输出编码器中的注意力权重')

    # 优化配置
    parser.add_argument('--num_workers', type=int, default=10, help='数据加载器的工作进程数')
    parser.add_argument('--itr', type=int, default=1, help='实验重复次数')
    parser.add_argument('--train_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='训练输入数据的批次大小')
    parser.add_argument('--patience', type=int, default=3, help='早停耐心值')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='优化器学习率')
    parser.add_argument('--des', type=str, default='test', help='实验描述')
    parser.add_argument('--loss', type=str, default='MSE', help='损失函数类型')
    parser.add_argument('--lradj', type=str, default='type1', help='学习率调整策略')
    parser.add_argument('--use_amp', action='store_true', help='是否使用自动混合精度训练', default=False)

    # GPU配置
    parser.add_argument('--use_gpu', type=bool, default=True, help='是否使用GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备ID')
    parser.add_argument('--use_multi_gpu', action='store_true', help='是否使用多GPU', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='多GPU设备ID列表')

    parser.add_argument('--stride', type=int, default=1, help='滑动窗口步长')
    parser.add_argument('--ckpt_path', type=str, default='', help='模型检查点文件路径')
    parser.add_argument('--finetune_epochs', type=int, default=10, help='微调训练轮数')
    parser.add_argument('--local_rank', type=int, default=0, help='本地进程排名（分布式训练）')

    parser.add_argument('--patch_len', type=int, default=24, help='输入序列长度（补丁长度）')
    parser.add_argument('--subset_rand_ratio', type=float, default=1, help='子集随机采样比例')
    parser.add_argument('--data_type', type=str, default='custom', help='数据类型')

    parser.add_argument('--decay_fac', type=float, default=0.75, help='衰减因子')

    # 余弦衰减配置
    parser.add_argument('--cos_warm_up_steps', type=int, default=100, help='余弦退火热身步数')
    parser.add_argument('--cos_max_decay_steps', type=int, default=60000, help='余弦退火最大衰减步数')
    parser.add_argument('--cos_max_decay_epoch', type=int, default=10, help='余弦退火最大衰减轮数')
    parser.add_argument('--cos_max', type=float, default=1e-4, help='余弦退火最大学习率')
    parser.add_argument('--cos_min', type=float, default=2e-6, help='余弦退火最小学习率')

    # 权重衰减配置
    parser.add_argument('--use_weight_decay', type=int, default=0, help='是否使用权重衰减')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减系数')

    # 自回归配置
    parser.add_argument('--use_ims', action='store_true', help='是否使用迭代多步预测', default=False)
    parser.add_argument('--output_len', type=int, default=96, help='输出序列长度')
    parser.add_argument('--output_len_list', type=int, nargs="+", help="输出长度列表（多尺度预测）")

    # 训练测试配置
    parser.add_argument('--train_test', type=int, default=1, help='是否同时进行训练和测试')
    parser.add_argument('--is_finetuning', type=int, default=1, help='是否为微调模式')

    # 预测任务配置
    parser.add_argument('--seq_len', type=int, default=96, help='输入序列长度')
    parser.add_argument('--label_len', type=int, default=48, help='起始标记长度')
    parser.add_argument('--pred_len', type=int, default=96, help='预测序列长度')

    # 填补任务配置
    parser.add_argument('--mask_rate', type=float, default=0.25, help='掩码比例（缺失值比例）')

    args = parser.parse_args()
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_multi_gpu:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "64209")
        hosts = int(os.environ.get("WORLD_SIZE", "8"))  # 节点数量
        rank = int(os.environ.get("RANK", "0"))  # 节点ID
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        gpus = torch.cuda.device_count()  # 每个节点的GPU数量
        args.local_rank = local_rank
        print(
            'IP地址: {}, 端口: {}, 主机数: {}, 全局排名: {}, 本地排名: {}, GPU数量: {}'.format(ip, port, hosts, rank, local_rank, gpus))
        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts, rank=rank)
        print('分布式进程组初始化完成')
        torch.cuda.set_device(local_rank)

    if args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        # Exp = Exp_Anomaly_Detection
        Exp = Custom_Exp_Anomaly_Detection
    elif args.task_name == 'forecast':
        Exp = Exp_Forecast
    else:
        raise ValueError('未找到任务名称')

    with HiddenPrints(int(os.environ.get("LOCAL_RANK", "0"))):
        print('实验参数:')
        print(args)
        if args.is_finetuning:
            for ii in range(args.itr):
                # 实验设置记录
                # setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
                setting = '{}_{}_{}_{}_{}'.format(
                    args.task_name,
                    args.model_id,
                    args.model,
                    args.data,
                    # args.features,
                    # args.seq_len,
                    # args.label_len,
                    # args.pred_len,
                    # args.patch_len,
                    # args.d_model,
                    # args.n_heads,
                    # args.e_layers,
                    # args.d_layers,
                    # args.d_ff,
                    # args.factor,
                    # args.embed,
                    # args.distil,
                    # args.des,
                    ii)
                setting += datetime.now().strftime("%y-%m-%d_%H-%M-%S")

                exp = Exp(args)  # 初始化实验
                print('>>>>>>>开始训练 : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.finetune(setting)

                print('>>>>>>>开始测试 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
                torch.cuda.empty_cache()
        else:
            ii = 0
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                ii)

            setting += datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            exp = Exp(args)  # 初始化实验
            print('>>>>>>>开始测试 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
            torch.cuda.empty_cache()