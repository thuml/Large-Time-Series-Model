import argparse
import random
from datetime import datetime

import numpy as np
import torch

from exp.exp_large_fewshot_forecast import Exp_Large_Fewshot_forecast

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Timer')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='large_fewshot_forecast',
                        help='task name, options:[large_fewshot_forecast large_fewshot_imputation large_fewshot_anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Timer', help='model name, options: [Timer]')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='the lookback length')
    parser.add_argument('--label_len', type=int, default=48, help='the label length')
    parser.add_argument('--pred_len', type=int, default=96, help='the forecast length by an autoregression')
    parser.add_argument('--output_len', type=int, default=96, help='the length to be forecasted')
    parser.add_argument('--segment_len', type=int, default=24, help='the patch length')

    # model define
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--stride', type=int, default=1, help='stride')
    parser.add_argument('--ckpt_path', type=str, default='', help='ckpt file')
    parser.add_argument('--finetune_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--subset_rand_ratio', type=float, default=1, help='mask ratio')
    parser.add_argument('--data_type', type=str, default='custom', help='data_type')

    # weight decay
    parser.add_argument('--use_weight_decay', type=int, default=0, help='use_post_data')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--decay_fac', type=float, default=0.75)

    args = parser.parse_args()
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.task_name == 'large_fewshot_forecast':
        Exp = Exp_Large_Fewshot_forecast
    elif args.task_name == 'large_fewshot_imputation':
        raise NotImplementedError
    elif args.task_name == 'large_fewshot_anomaly_detection':
        raise NotImplementedError
    else:
        raise NotImplementedError

    print('Args in experiment:')
    print(args)
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_pl{}_eb{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.segment_len,
            args.embed,
            args.des,
            ii)
        setting += datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        print(setting)

        exp = Exp(args)  # set experiments

        exp.finetune(setting)
        exp.test(setting)

        torch.cuda.empty_cache()
