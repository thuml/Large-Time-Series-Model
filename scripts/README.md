## Other Tasks

We provide example scripts for few-shot forecasting, imputation and anomaly detection. The checkpoint is pre-trained and fine-tuned using [TSLib](https://github.com/thuml/Time-Series-Library).


### Supported Tasks

> **[Forecasting](./scripts/forecast/README.md)**: We provide scripts for few-shot forecasting.

> **[Imputation](./scripts/imputation/README.md)**:  We adopt segment-level imputation, which is more challenging than point-level imputation.

> **[Anomaly Detection](scripts/anomaly_detection/README.md)**: We build a benchmark using [UCR Anomaly Archive](https://arxiv.org/pdf/2009.13807). The task aims to predict normal future series and detect anomalies in advance.

We provide the README files illustrating each task under the folder ```./scripts/```.


### Code for Fine-tuning 

1. Use Python 3.10 and install necessary dependencies.

```
pip install -r requirements.txt
```

2. Put downstream datasets from [Google Drive](https://drive.google.com/file/d/1yffcQBcMLasQcT7cdotjOVcg-2UKRarw/view?usp=drive_link) or [Baidu Drive](https://pan.baidu.com/s/1KLwxB0Au-rxpmgY0yu2d3w?pwd=6k73) under the folder ```./dataset/```.

3. Put the checkpoint from [Google Drive](https://drive.google.com/drive/folders/15oaiAl4OO5gFqZMJD2lOtX2fxHbpgcU8?usp=drive_link) or [Baidu Drive](https://pan.baidu.com/s/1Wj_1_qMgyLNLOSUFZK3weg?pwd=r8i1) under the folder ```./checkpoints/```.

4. Train and evaluate the model. We provide the above tasks under the folder ```./scripts/```.

```bash
# forecasting
bash ./scripts/forecast/ECL.sh

# segement-level imputation
bash ./scripts/imputation/ECL.sh

# anomaly detection
bash ./scripts/anomaly_detection/UCR.sh
```