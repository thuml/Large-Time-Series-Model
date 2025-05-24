# Timer (Large Time-Series Model)

This repo provides official code, datasets, and checkpoints for [Timer: Generative Pre-trained Transformers Are Large Time Series Models](https://arxiv.org/abs/2402.02368). [[Poster]](https://cloud.tsinghua.edu.cn/f/91da8a3d06984f209461/), [[Slides]](https://cloud.tsinghua.edu.cn/f/b766629dbc584a4e8563/).

# Updates

:triangular_flag_on_post: **News** (2025.5) [Sundial](https://arxiv.org/abs/2502.00816), a family of generative time series foundation models has been accepted as **ICML 2025 Spotlight** (Top 2.6%). Get your first zero-shot predictions in one second! [[GitHub]](https://github.com/thuml/Sundial), [[HuggingFace]](https://huggingface.co/thuml/sundial-base-128m). 

:triangular_flag_on_post: **News** (2025.2) We release an open codebase [OpenLTM](https://github.com/thuml/OpenLTM), which contains a simple pipeline to pre-train customized large time-series models :)

:triangular_flag_on_post: **News** (2024.12) Timer-XL for unified forecasting is accepted as  [ICLR 2025](https://arxiv.org/abs/2410.04803). We released a pre-trained model on **260B time points** [[Performance]](./figures/zeroshot_result.png) [[Checkpoint]](https://huggingface.co/thuml/timer-base-84m) [[Quickstart]](./examples/quickstart_zero_shot.ipynb).

:triangular_flag_on_post: **News** (2024.10) We release the pre-training dataset UTSD on [HuggingFace](https://huggingface.co/datasets/thuml/UTSD) or you can use the numpy format [UTSD](https://cloud.tsinghua.edu.cn/f/93868e3a9fb144fe9719/) and this [dataloader](https://github.com/thuml/OpenLTM/blob/main/data_provider/data_loader.py).

:triangular_flag_on_post: **News** (2024.5) Accepted by ICML 2024, a [camera-ready version](https://arxiv.org/abs/2402.02368) of **31 pages**.

:triangular_flag_on_post: **News** (2024.2) Releasing model checkpoints and code for fine-tuning.

## Introduction

**Tim**e Series Transfor**mer** (Timer) is a Generative Pre-trained Transformer for general time series analysis.
<p align="center">
<img src="./figures/abilities.png" alt="" align=center />
</p>


## Zero-Shot Forecasting

We provide the checkpoint to make predictions without training samples. See our [HuggingFace Repo](https://huggingface.co/thuml/timer-base-84m) for more information.

> Example

```
import torch
from transformers import AutoModelForCausalLM

# load pretrain model
model = AutoModelForCausalLM.from_pretrained('thuml/timer-base-84m', trust_remote_code=True)

# prepare input
batch_size, lookback_length = 1, 2880
seqs = torch.randn(batch_size, lookback_length)

# generate forecast
prediction_length = 96
normed_output = model.generate(normed_seqs, max_new_tokens=prediction_length)

print(output.shape)
```

## Model Adaption

For developers interested in fine-tuning large time-series models or pre-training on customized datasets, please refer to [OpenLTM](https://github.com/thuml/OpenLTM), which includes the implementations and checkpoint of large time-series models.

For developers interested in applying large time-series models on other time series analysis tasks (e.g., imputation and anomaly detection), we provide example scripts [here](./scripts/README.md).

## Datasets

We collect Unified Time Series Datasets (UTSD), which encompass well-curated time series to facilitate the research on large time-series models. Our dataset is released in [HuggingFace](https://huggingface.co/datasets/thuml/UTSD).

<p align="center">
<img src="./figures/utsd.png" alt="" align=center />
</p>

###  Usage

You can access the data from HuggingFace and load the data in the style of [TSLib](https://github.com/thuml/Time-Series-Library):

```bash
# huggingface-cli login
# export HF_ENDPOINT=https://hf-mirror.com 

python ./scripts/UTSD/download_dataset.py

# dataloader
python ./scripts/UTSD/utsdataset.py
```

If you meet troubles when accessing the data, you can also download UTSD in numpy from [[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/93868e3a9fb144fe9719/) and use ```UTSD_Npy``` dataloader from [[OpenLTM]](https://github.com/thuml/OpenLTM/blob/main/data_provider/data_loader.py).


## Introduction

### Unified Pre-training

To pre-train on heterogeneous time series, we propose **single-series sequence (S3)**, reserving series variations into the unified 1D context. Further, we convert forecasting, imputation, and anomaly detection into a **unified generative task**.

<p align="center">
<img src="./figures/pretrain_adaptation.png" align=center />
</p>

### Model Architecture

We evaluate various candidate backbones and eventually adopt the **decoder-only Transformer**, which provides notable **generalization performance** and **flexibility** that accommodate varying-length time series.

<p align="center">
<img src="./figures/architecture.png" align=center />
</p>


## Performance

Timer achieves **state-of-the-art** performance in [zero-shot forecasting](./figures/zeroshot_result.png) and few-shot adaptation.

<p align="center">
<img src="./figures/performance.png" align=center />
</p>

## Scalability

By scaling, Timer achieves notable performance improvement. Currently, we provide the base version containing 84M parameters that is pre-trained on 260B time points, which supports a maximum context length of 2880.

<p align="center">
<img src="./figures/scale.png" alt="300" align=center />
</p>

## Subsequent Works

### ICLR 2025
 We proposed [Timer-XL](https://arxiv.org/abs/2410.04803) for unified time series forecasting.  It can be used for **task-specific training** or **scalable pre-training**, handling **arbitrary-length** and **any-variable** time series [[Repo]](https://github.com/thuml/Timer-XL).  

<p align="center">
<img src="./figures/timer-xl.png" alt="300" align=center />
</p>

### ICML 2025 
We proposed [Sundial](https://arxiv.org/abs/2502.00816), a family of **generative** time series foundation models, which is pre-trained on **a trillion** (10^12) time points. The model can be applied for **point** and **probabilistic** forecasting, making **zero-shot** predictions.  

<p align="center">
<img src="./figures/sundial.png" alt="300" align=center />
</p>

## Citation

If you find this repo helpful, please cite our paper. 

```
@inproceedings{liutimer,
  title={Timer: Generative Pre-trained Transformers Are Large Time Series Models},
  author={Liu, Yong and Zhang, Haoran and Li, Chenyu and Huang, Xiangdong and Wang, Jianmin and Long, Mingsheng},
  booktitle={Forty-first International Conference on Machine Learning}
}

@article{liu2024timer,
  title={Timer-XL: Long-Context Transformers for Unified Time Series Forecasting},
  author={Liu, Yong and Qin, Guo and Huang, Xiangdong and Wang, Jianmin and Long, Mingsheng},
  journal={arXiv preprint arXiv:2410.04803},
  year={2024}
}
```

## Contributors

If you have any questions or want to use the code, feel free to contact:
* Yong Liu (liuyong21@mails.tsinghua.edu.cn)
* Guo Qin (qinguo24@mails.tsinghua.edu.cn)
* Haoran Zhang (zhang-hr24@mails.tsinghua.edu.cn)
* Chenyu Li (lichenyu20@mails.tsinghua.edu.cn)
