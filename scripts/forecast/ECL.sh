#!/bin/sh

model_name=Timer
seq_len=672     # the lookback length
label_len=576
pred_len=96     # the forecast length by an autoregression
output_len=96   # the length to be forecasted
segment_len=96    
ckpt_path=checkpoints/Timer_67M_4G.pt
subset_rand_ratio=0.1   # downstream data scarcity

python run.py \
  --task_name large_fewshot_forecast \
  --is_training 0 \
  --ckpt_path $ckpt_path \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --data custom \
  --model_id ECL_{$subset_rand_ratio} \
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