import datetime


class InferenceConfig:
    def __init__(self):

        # --- 基础参数 ---
        self.is_training = 0
        self.seed = 0
        self.checkpoints = './checkpoints'
        self.inverse =  False
        self.itr = 1
        self.train_epochs = 10
        self.patience = 3
        self.learning_rate = 0.0001
        self.des = 'inference'
        self.loss = 'MSE'
        self.lradj = 'type1'
        self.use_amp = False
        self.stride = 1
        self.ckpt_path = ''
        self.finetune_epochs = 10
        self.local_rank = 0
        self.data_type = 'custom'
        self.decay_fac = 0.75
        # 余弦衰减配置
        self.cos_warm_up_steps = 100
        self.cos_max_decay_steps = 60000
        self.cos_max_decay_epoch = 10
        self.cos_max = 1e-4
        self.cos_min = 2e-6
        self.use_weight_decay = 0
        self.weight_decay = 0.01
        self.output_len = 96
        self.train_test = 1
        self.is_finetuning = 1
        self.mask_rate=0.25

        # --- 核心模型参数 (必须与训练一致) ---
        self.task_name = 'anomaly_detection'
        self.model_id = 'Inference_Run'
        self.model = 'Timer'  # 您的模型名称
        self.enc_in = 5  # 输入特征数
        self.c_out = 5  # 输出特征数
        self.d_model = 256
        self.n_heads = 8
        self.e_layers = 4
        self.d_layers = 1
        self.d_ff = 512
        self.factor = 1
        self.distil = True
        self.dropout = 0.1
        self.embed = 'timeF'
        self.activation = 'gelu'
        self.output_attention = False
        self.patch_len = 16  # Timer 特有
        self.use_ims = False

        # --- 数据加载参数 (推理时调整) ---
        self.today_ = datetime.datetime.now().strftime("%Y_%m_%d")
        self.data = 'UCRA'  # 数据集类型名称 (触发对应的 Dataset 类)
        # self.root_path = r'\dataset\xudianchi\train\normal_diff'  # 推理数据所在的文件夹
        self.data_path = 'ALL'  # 推理文件名 (如果您的Dataset支持读取文件夹，这里可以是文件名或'ALL')
        self.features = 'M'
        self.target = 'OT'
        self.freq = 'h'
        self.seq_len = 96  # 必须与训练一致
        self.label_len = 48  # 异常检测通常不需要 label_len
        self.pred_len = 96  # 异常检测通常不需要 pred_len
        self.batch_size = 128  # 推理时的 batch size，可以设大一点
        self.num_workers = 0  # 避免多进程开销，推理推荐设为0
        self.subset_rand_ratio = 1.0  # 必须使用 100% 数据进行推理

        # --- 运行参数 ---
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0'

        # --- 权重路径 ---
        # 训练好的模型路径
        self.cut_data = False
        self.scaler_save_path = r'D:\Work\LLM\GitHub\Large-Time-Series-Model\Halosee\scaler_all_0108.pkl'

        # 数据集路径
        self.root_path = r"D:\Work\LLM\GitHub\Large-Time-Series-Model\dataset\xudianchi\train\abnormal_0108\0238_1_1"
        # 权重路径 (必须是包含 global_mse_threshold 的新权重)
        self.checkpoint_path = r'D:\Work\LLM\GitHub\Large-Time-Series-Model\checkpoints\anomaly_detection_UCRA_Combined_26Files_Timer_UCRA_026-01-08_14-08-16\checkpoint.pth'
        # 结果路径
        self.result_dir_path = "./result/0108/0238_1_1/"
