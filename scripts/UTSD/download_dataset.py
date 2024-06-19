# if you want to download the dataset, you can run this script:
# '''python download_dataset.py'''

# if you meet with some network problems, you can set the mirror site before running the script:
# export HF_ENDPOINT=https://hf-mirror.com

import datasets

ds = datasets.load_dataset("thuml/UTSD", "UTSD-1G")
# ds = datasets.load_dataset("thuml/UTSD", "UTSD-2G")
# ds = datasets.load_dataset("thuml/UTSD", "UTSD-4G")
# ds = datasets.load_dataset("thuml/UTSD", "UTSD-12G")

# the dataset have not been divided into train, test, and val splits
# therefore, ds['train'] contains all the time series
# you can split them by yourself, or use our default split as train:val=9:1 in '''utsdataset.py'''
all = ds['train']

# print the total number of time series
print(f'total {len(all)} single-variate series')

# each item is a single-variate series containing: 
# 1. dataset name (item_id)
# 2. start time (start)
# 3. end time (end)
# 4. sampling frequecy (freq)
# 5. time series values (target)
# timestampes are optional since some datasets are irregular and may not have 

# see https://huggingface.co/datasets/thuml/UTSD/viewer for more details`
print(all[0].keys())

# you can access the time series values by item['target']
num_timepoints = len(all[0]['target'])
print(f'the first time series containing {num_timepoints} time points')

# or generate the timestamps by item['start'], item['end'], and item['freq']
