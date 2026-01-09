import os
import subprocess

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 配置参数
model_name = "Timer"
ckpt_path = "checkpoints/Timer_anomaly_detection_1.0.ckpt"
seq_len = 96
d_model = 256
d_ff = 512
e_layers = 4
patch_len = 16
pred_len = 96
subset_rand_ratio = 0.01
dataset_dir = r"csv_data\4_dataset\normal"
scaler_save_path = r"Halosee\scaler_all_0108.pkl"


# --- 直接指定 data_path 为特殊标记 "ALL" ---
# 只启动一个进程，训练一个包含所有数据的模型
data = "ALL"

cmd = [
    "python", "-u", "run_ch.py",
    "--task_name", "anomaly_detection",
    "--is_training", "1",
    "--root_path", dataset_dir,  # 保持 root_path 指向文件夹
    "--data_path", data,  # 传递 "ALL" 作为文件名，后续在 Dataset 中处理
    "--model_id", f"UCRA_Combined_26Files",  # 修改 model_id 以区分
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
    "--batch_size", "512",
    # "--subset_rand_ratio", str(subset_rand_ratio),
    "--train_epochs", "10",
    "--num_workers", "0",
    "--subset_rand_ratio", "0",
    "--scaler_save_path", scaler_save_path,
    "--is_finetuning", "0",
]

# 执行命令
print(f"开始执行训练，使用数据集目录: {dataset_dir} 下的所有文件...")
subprocess.run(cmd)
