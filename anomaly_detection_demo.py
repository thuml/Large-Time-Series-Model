import os
import subprocess

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# os.makedirs('./test_results', exist_ok=True)

# 配置参数
model_name = "Timer"
ckpt_path = "checkpoints/Timer_anomaly_detection_1.0.ckpt"
seq_len = 96
d_model = 256
d_ff = 512
e_layers = 4
patch_len = 16
pred_len = 0
# seq_len=96
# d_model=256
# d_ff=512
# e_layers=4
# patch_len=96
# pred_len=96
subset_rand_ratio = 0.01
dataset_dir = "./dataset/xudianchi/txt"
# dataset_dir = "./dataset/UCR_Anomaly_FullData"

# 遍历数据集目录中的所有文件
for file_path in os.listdir(dataset_dir):
    data = file_path
    print(f"开始执行训练，使用数据集目录: {dataset_dir} 下的文件: {data}...")
    cmd = [
        "python", "-u", "run_ch.py",
        "--task_name", "anomaly_detection",
        "--is_training", "1",
        "--root_path", "./dataset/xudianchi/txt",
        # "--root_path", "./dataset/UCR_Anomaly_FullData",
        "--data_path", data,
        "--model_id", f"UCRA_{data}",
        # "--ckpt_path", ckpt_path,
        "--model", model_name,
        "--data", "UCRA",
        "--features", "M",
        "--seq_len", str(seq_len),
        "--d_model", str(d_model),
        "--d_ff", str(d_ff),
        "--patch_len", str(patch_len),
        "--pred_len", str(pred_len),
        "--e_layers", str(e_layers),
        "--train_test", "0",
        "--batch_size", "128",
        "--subset_rand_ratio", str(subset_rand_ratio),
        "--train_epochs", "10",
        "--num_workers", "0",
        "--subset_rand_ratio", "0",
    ]

    # 执行命令
    subprocess.run(cmd)
