#!/bin/sh

model_name=Timer
seq_len=672
label_len=576
pred_len=96 # the length of a single autoregressive roll
output_len=96 # the length of forecasting
segment_len=96
ckpt_path=checkpoints/Timer_67M_4G.pt
subset_rand_ratio=0.1

python run.py \
  --task_name large_fewshot_forecast \
  --is_training 0 \
  --ckpt_path $ckpt_path \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --data custom \
  --model_id Weather_{$subset_rand_ratio} \
  --model $model_name \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --output_len $output_len \
  --des 'Exp' \
  --batch_size 2048 \
  --learning_rate 3e-5 \
  --num_workers 4 \
  --segment_len $segment_len \
  --subset_rand_ratio $subset_rand_ratio \
  --itr 1
