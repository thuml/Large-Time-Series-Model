import datasets
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


"""
All single-variate series in UTSD are divided into (input-output) windows with a uniform length based on S3.
"""
class UTSDataset(Dataset):
    def __init__(self, subset_name=r'UTSD-1G', flag='train', split=0.9,
                 input_len=None, output_len=None, scale=True, stride=1):
        self.input_len = input_len
        self.output_len = output_len
        self.seq_len = input_len + output_len
        assert flag in ['train', 'val']
        assert split >= 0 and split <=1.0
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.scale = scale
        self.split = split
        self.stride = stride

        self.data_list = []
        self.n_window_list = []

        self.subset_name = subset_name
        self.__read_data__()

    def __read_data__(self):
        dataset = datasets.load_dataset("thuml/UTSD", self.subset_name, split='train')
        # split='train' contains all the time series, which have not been divided into splits, 
        # you can split them by yourself, or use our default split as train:val = 9:1
        print('Indexing dataset...')
        for item in tqdm(dataset):
            self.scaler = StandardScaler()
            data = item['target']
            data = np.array(data).reshape(-1, 1)
            num_train = int(len(data) * self.split)
            border1s = [0, num_train - self.seq_len]
            border2s = [num_train, len(data)]

            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.scale:
                train_data = data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data)
                data = self.scaler.transform(data)

            data = data[border1:border2]
            n_window = (len(data) - self.seq_len) // self.stride + 1
            if n_window < 1:
                continue

            self.data_list.append(data)
            self.n_window_list.append(n_window if len(self.n_window_list) == 0 else self.n_window_list[-1] + n_window)


    def __getitem__(self, index):
        # you can wirte your own processing code here
        dataset_index = 0
        while index >= self.n_window_list[dataset_index]:
            dataset_index += 1

        index = index - self.n_window_list[dataset_index - 1] if dataset_index > 0 else index
        n_timepoint = (len(self.data_list[dataset_index]) - self.seq_len) // self.stride + 1

        s_begin = index % n_timepoint
        s_begin = self.stride * s_begin
        s_end = s_begin + self.seq_len
        p_begin = s_end
        p_end = p_begin + self.output_len
        seq_x = self.data_list[dataset_index][s_begin:s_end, :]
        seq_y = self.data_list[dataset_index][p_begin:p_end, :]

        return seq_x, seq_y

    def __len__(self):
        return self.n_window_list[-1]


# See ```download_dataset.py``` to download the dataset first
if __name__ == '__main__':
    # dataset = UTSDataset(subset_name=r'UTSD-1G', input_len=672, output_len=0, flag='train')
    dataset = UTSDataset(subset_name=r'UTSD-1G', input_len=720, output_len=96, flag='train')
    print(f'total {len(dataset)} time series windows (sentence)')
