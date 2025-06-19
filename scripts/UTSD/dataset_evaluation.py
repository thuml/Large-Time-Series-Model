from arch.unitroot import ADF
from scipy.stats import entropy
import numpy as np
import torch
import argparse
from datasets import load_from_disk


def adf_evaluator(x):
    return ADF(x).stat


def forecastability_evaluator(x, seq_len=256):
    x = torch.tensor(x).squeeze() # L
    forecastability_list = []
    for i in range(max(x.shape[0]-seq_len, 0) // seq_len + 1):
        start_idx = i * seq_len
        end_idx = min(start_idx + seq_len, x.shape[0])
        window = x[start_idx:end_idx]
        amps = torch.abs(torch.fft.rfft(window))
        amp = torch.sum(amps)
        forecastability = 1 - entropy(amps/amp, base=len(amps))
        forecastability_list.append(forecastability)
    np_forecastability_list = np.array(forecastability_list)
    # replace nan with 1
    np_forecastability_list[np.isnan(np_forecastability_list)] = 1
    return np.mean(np_forecastability_list)


def save_log(path, content):
    with open(path, 'a') as f:
        f.write(content)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Evaluation')
    parser.add_argument('--root_path', type=str, required=True, help='Root path of the dataset, e.g. ./data/bdg-2_bear')
    parser.add_argument('--log_path', type=str, required=False, default='log.txt', help='Path to save the log file')
    args = parser.parse_args()
    print("Evaluate dataset at ", args.root_path)
    
    dataset = load_from_disk(args.root_path)
    print(dataset)
    series_list = dataset['target']
    
    if not isinstance(series_list[0][0], list):
        series_list = [series_list]
    
    time_point_list = []
    adf_stat_list = []
    forecastability_list = []
    
    for i in range(len(series_list)):
        for j in range(len(series_list[i])):
            try:
                series = series_list[i][j]
                # fill missing value with 0 for evaluation
                series = [0 if np.isnan(x) else x for x in series]
                adf_stat = adf_evaluator(series)
                forecastability = forecastability_evaluator(series)
                forecastability_list.append(forecastability)
                adf_stat_list.append(adf_stat)
                time_point_list.append(len(series))
            except Exception as e:
                save_log(args.log_path, f'Error: {args.root_path} {i} {j}\n'+str(e)+'\n')
                continue
    
    time_point_list = np.array(time_point_list)
    adf_stat_list = np.array(adf_stat_list)
    forecastability_list = np.array(forecastability_list)
    
    time_points = np.sum(time_point_list)
    weighted_adf = np.sum(adf_stat_list * time_point_list) / time_points
    weighted_forecastability = np.sum(forecastability_list * time_point_list) / time_points
    
    print("Weighted ADF:", weighted_adf)
    print("Weighted Forecastability:", weighted_forecastability)
    print("Total Time Points:", time_points)
    print("Finish evaluation ", args.root_path)
    save_log(args.log_path, f"root_path: {args.root_path}\n Weighted ADF: {weighted_adf}\n Weighted Forecastability: {weighted_forecastability}\n Total Time Points: {time_points}\n\n")