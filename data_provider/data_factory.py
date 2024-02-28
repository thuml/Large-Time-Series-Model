import os

from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, Dataset_S3

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
}

def data_provider(args, flag):
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
    else:
        shuffle_flag = True

    if args.task_name == 'large_fewshot_forecast':
        data_set = Dataset_S3(
            root_path=os.path.join(args.root_path, args.data_path),
            flag=flag,
            input_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.output_len if flag == 'test' else args.pred_len,
            data_type=args.data,
            scale=True,
            timeenc=timeenc,
            freq=args.freq,
            stride=args.stride,
            subset_rand_ratio=args.subset_rand_ratio,
        )

        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=False)

        return data_set, data_loader
    else:
        raise ValueError('Task Not Implemneted')
