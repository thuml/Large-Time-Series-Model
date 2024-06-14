#!/bin/sh

export CUDA_VISIBLE_DEVICES=0

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
  --pred_len 192 \   # not used in imputation
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