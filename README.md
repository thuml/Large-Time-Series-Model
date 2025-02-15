> [!NOTE]
> We release a open codebase [**OpenLTM**](https://github.com/thuml/OpenLTM) to explore the design philosophy of large time-series models, which contains a simple pipeline to train large time-series models :)


# Timer (Large Time-Series Model)

This repo provides official code, datasets and checkpoints for [Timer: Generative Pre-trained Transformers Are Large Time Series Models](https://arxiv.org/abs/2402.02368). [[Poster]](https://cloud.tsinghua.edu.cn/f/91da8a3d06984f209461/), [[Slides]](https://cloud.tsinghua.edu.cn/f/b766629dbc584a4e8563/).

# Updates

:triangular_flag_on_post: **News** (2024.12) Timer is enhanced with [subsequent work (ICLR 2025)](https://arxiv.org/abs/2410.04803) and pre-trained on **260B time points**. Checkpoint is now available: [[HuggingFace]](https://huggingface.co/thuml/timer-base-84m) [[Benchmark]](https://cdn-uploads.huggingface.co/production/uploads/64fbe24a2d20ced4e91de38a/VAfuvvqBALLvQUXYJPZJx.png). An example of zero-shot forecasting is provided [here](./examples/quickstart_zero_shot.ipynb).

:triangular_flag_on_post: **News** (2024.10) We release numpy format of [UTSD](https://cloud.tsinghua.edu.cn/f/93868e3a9fb144fe9719/). An easier and more efficient dataloader can be found [here](https://github.com/thuml/OpenLTM/blob/main/data_provider/data_loader.py).

:triangular_flag_on_post: **News** (2024.6) Pre-training dataset (UTSD) is available in [HuggingFace](https://huggingface.co/datasets/thuml/UTSD). Dataloader is also contained.

:triangular_flag_on_post: **News** (2024.5) Accepted by ICML 2024, a [camera-ready version](https://arxiv.org/abs/2402.02368) of **31 pages**.

:triangular_flag_on_post: **News** (2024.2) Releasing model checkpoints and code for fine-tuning.

## Introduction

**Tim**e Series Transfor**mer** (Timer) is a Generative Pre-trained Transformer for general time series analysis.
<p align="center">
<img src="./figures/abilities.png" alt="" align=center />
</p>


## Zero-Shot Forecasting
We provide the checkpoint to make predictions without training samples. See our [HuggingFace Repo](https://huggingface.co/thuml/timer-base-84m) for the detialed information and usage.

> A inference example (**minimal dependencies required**): 

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

There's indeed room for improvement in this small model. We are actively working around it and are glad to see constructive suggestions and noteworthy cases :)

## Datasets

We collect Unified Time Series Datasets (UTSD), which encompass well-curated time series to facilitate the research on large time-series models. Our dataset is released in [HuggingFace](https://huggingface.co/datasets/thuml/UTSD).

<p align="center">
<img src="./figures/utsd.png" alt="" align=center />
</p>

###  Usage

You can access and load UTSD in the style of [TSLib](https://github.com/thuml/Time-Series-Library) based on the following steps:

```bash
# huggingface-cli login
# export HF_ENDPOINT=https://hf-mirror.com 

python ./scripts/UTSD/download_dataset.py

# dataloader
python ./scripts/UTSD/utsdataset.py
```

## For Developers 

For developers interest in large model adaptation, we provide fine-tuning code based on [non-HuggingFace checkpoints](https://drive.google.com/drive/folders/15oaiAl4OO5gFqZMJD2lOtX2fxHbpgcU8?usp=drive_link), which is a smaller version of Timer developed in the [TSLib](https://github.com/thuml/Time-Series-Library) style.

> [!NOTE]
>  We recommend using [checkpoints on HuggingFace](https://huggingface.co/thuml/timer-base-84m) for model evaluation (e.g., zero-shot forecasting). However, it is not compatiable with the following fine-tuning code (but we are working on it :)
> 
> 

### Supported Tasks

> **[Forecasting](./scripts/forecast/README.md)**: We provide all scripts for few-shot forecasting in this repo.

> **[Imputation](./scripts/imputation/README.md)**:  We propose segment-level imputation, which is more challenging than point-level imputation.

> **[Anomaly Detection](scripts/anomaly_detection/README.md)**: We provide new benchmarks of predictive anomaly detection on [UCR Anomaly Archive](https://arxiv.org/pdf/2009.13807).

We provide the README files illustrating each task under the folder ```./scripts/```.



### Code for Fine-tuning 

1. Use Python 3.10 and install necessary dependencies.

```
pip install -r requirements.txt
```

2. Put downstream datasets from [Google Drive](https://drive.google.com/file/d/1yffcQBcMLasQcT7cdotjOVcg-2UKRarw/view?usp=drive_link) and [Baidu Drive](https://pan.baidu.com/s/1KLwxB0Au-rxpmgY0yu2d3w?pwd=6k73) under the folder ```./dataset/```.

3. Put the checkpoint from [Google Drive](https://drive.google.com/drive/folders/15oaiAl4OO5gFqZMJD2lOtX2fxHbpgcU8?usp=drive_link) and [Baidu Drive](https://pan.baidu.com/s/1Wj_1_qMgyLNLOSUFZK3weg?pwd=r8i1) under the folder ```./checkpoints/```.

4. Train and evaluate the model. We provide the above tasks under the folder ```./scripts/```.

```bash
# forecasting
bash ./scripts/forecast/ECL.sh

# segement-level imputation
bash ./scripts/imputation/ECL.sh

# anomaly detection
bash ./scripts/anomaly_detection/UCR.sh
```

### Train on Custom Dataset

To fine-tune on your time series dataset, you can try out the following steps:

1. The key is to reload the customized dataloader and load the pre-trained checkpoint (See ```./scripts/``` folder).
2. ```CIDatasetBenchmark```/```CIAutoRegressionDatasetBenchmark``` in the ```data_provider``` folder can train and evaluate models in direct / iterative multi-step mode.


## Approach

### Pre-training and Adaptation

To pre-train on heterogeneous time series, we propose **single-series sequence (S3)**, reserving series variations into the unified 1D context. Further, we convert forecasting, imputation, and anomaly detection into a **unified generative task**.

<p align="center">
<img src="./figures/pretrain_adaptation.png" align=center />
</p>

### Model Architecture

We evaluate various candidate backbones and eventually adopt the **decoder-only Transformer**, which provides notable **generalization performance** and **length-flexibility** that accommodate various time series.

<p align="center">
<img src="./figures/architecture.png" align=center />
</p>


## Performance

Timer achieves **state-of-the-art** performance in [zero-shot forecasting](https://cdn-uploads.huggingface.co/production/uploads/64fbe24a2d20ced4e91de38a/VAfuvvqBALLvQUXYJPZJx.png), general time series analysis, and present the pre-training benefit on few-shot scenarios.

<p align="center">
<img src="./figures/performance.png" align=center />
</p>

## Scalability

By scaling, Timer achieves notable performance improvement. Currently, we provide the base version containing 84M paramaters that is pre-trained on 260B time points, which supports a **maximum context length of 2880**.

<p align="center">
<img src="./figures/scale.png" alt="300" align=center />
</p>

## Futher Improvement

We enhanced Timer by this [paper](https://arxiv.org/abs/2410.04803) with **longer context** and **TimeAttention**.

<p align="center">
<img src="./figures/timer-xl.png" alt="300" align=center />
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
