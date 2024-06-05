# Time Series Imputation

Imputation is ubiquitous in real-world applications, aiming to fill corrupted time series based on partially observed data. However, while various machine learning algorithms and simple linear interpolation can effectively cope with the corruptions randomly happening at the point level, real-world corruptions typically result from prolonged monitor shutdowns and require a continuous period of recovery. Consequently, imputation can be ever challenging when attempting to recover a span of time points encompassing intricate series variations.


## Dataset

We establish a comprehensive imputation benchmark, which includes 11 datasets with 4 mask ratios $\{12.5\%, 25.0\%, 37.5\%, 50.0\%\}$ each. We provide the download links: [Google Drive](https://drive.google.com/file/d/1yffcQBcMLasQcT7cdotjOVcg-2UKRarw/view?usp=drive_link) and [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/6bc31f9a003b4d75a10b/).

<p align="center">
<img src="../../figures/forecast_dataset.png" alt="" align=center />
</p>

## Task Description

In this task, we conduct the segment-level imputation. Each time series is divided into several segments and each segment has the possibility of being completely masked. We randomly mask segments as zeros except for the first segment, ensuring that the first part is observed by the model. By introducing masked tokens during adaptation, Timer generates imputations with the previous context and assembles them with the observed part. We take the MSE of the masked segments as the measurement to evaluate the imputation performance. 

## Scripts

```bash
model_name=Timer
ckpt_path=checkpoints/Timer_imputation_1.0.ckpt
d_model=256
d_ff=512
e_layers=4
patch_len=24

# set data scarcity ratio
for subset_rand_ratio in 0.05 0.2 1
do
# set mask rate of imputation
for mask_rate in 0.125 0.25 0.375 0.5
do
python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --model_id electricity_mask_$mask_rate \
  --mask_rate $mask_rate \
  --model $model_name \
  --ckpt_path $ckpt_path \
  --data_path electricity.csv \
  --data custom \
  --features M \
  --seq_len 192 \
  --label_len 0 \
  --pred_len 192 \
  --patch_len $patch_len \
  --e_layers $e_layers \
  --factor 3 \
  --train_test 0 \
  --batch_size 16 \
  --d_model $d_model \
  --d_ff $d_ff \
  --des 'Exp' \
  --itr 1 \
  --use_ims \
  --subset_rand_ratio $subset_rand_ratio \
  --learning_rate 0.001
done
done
```
