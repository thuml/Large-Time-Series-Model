import torch.multiprocessing

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import adjust_learning_rate, visual

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                if self.args.use_ims:
                    # backward overlapping parts between outputs and inputs
                    outputs = self.model(batch_x[:, :-self.args.patch_len, :], None, None, None)
                    batch_x = batch_x[:, self.args.patch_len:, :]
                else:
                    # input and output are completely aligned
                    outputs = self.model(batch_x, None, None, None)

                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def finetune(self, setting):

        train_data, train_loader = self._get_data(flag='train')
        # vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch_x in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                if self.args.use_ims:
                    # backward overlapping parts between outputs and inputs
                    outputs = self.model(batch_x[:, :-self.args.patch_len, :], None, None, None)
                    batch_x = batch_x[:, self.args.patch_len:, :]
                else:
                    # input and output are completely aligned
                    outputs = self.model(batch_x, None, None, None)

                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # vali_loss = self.vali(vali_data, vali_loader, criterion)
            if self.args.train_test:
                test_loss = self.vali(test_data, test_loader, criterion)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        return self.model

    def find_border(self, input_string):
        parts = input_string.split('_')

        if len(parts) < 3:
            return None

        border1_str = parts[-2]
        border2_str = parts[-1]
        if '.' in border2_str:
            border2_str = border2_str[:border2_str.find('.')]

        try:
            border1 = int(border1_str)
            border2 = int(border2_str)
            return border1, border2
        except ValueError:
            return None

    def find_border_number(self, input_string):
        parts = input_string.split('_')

        if len(parts) < 3:
            return None

        border_str = parts[-3]

        try:
            border = int(border_str)
            return border
        except ValueError:
            return None

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        score_list = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        border_start = self.find_border_number(self.args.data_path)
        border1, border2 = self.find_border(self.args.data_path)

        token_count = 0
        if self.args.use_ims:
            rec_token_count = (self.args.seq_len - 2 * self.args.patch_len) // self.args.patch_len
        else:
            rec_token_count = self.args.seq_len // self.args.patch_len

        input_list = []
        output_list = []
        with torch.no_grad():
            for i, batch_x in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruct the input sequence and record the loss as a sorted list
                if self.args.use_ims:
                    outputs = self.model(batch_x[:, :-self.args.patch_len, :], None, None, None)
                    batch_x = batch_x[:, self.args.patch_len:-self.args.patch_len, :]
                    outputs = outputs[:, :-self.args.patch_len, :]
                else:
                    outputs = self.model(batch_x, None, None, None)

                input_list.append(batch_x[0, :, -1].detach().cpu().numpy())
                output_list.append(outputs[0, :, -1].detach().cpu().numpy())
                for j in range(rec_token_count):
                    # criterion
                    token_start = j * self.args.patch_len
                    token_end = token_start + self.args.patch_len
                    score = torch.mean(self.anomaly_criterion(batch_x[:, token_start:token_end, :],
                                                              outputs[:, token_start:token_end, :]), dim=-1)
                    score = score.detach().cpu().numpy()
                    score = np.mean(score)
                    score_list.append((token_count, score))
                    token_count += 1

        input = np.concatenate(input_list, axis=0).reshape(-1)
        output = np.concatenate(output_list, axis=0).reshape(-1)
        half_patch_len = self.args.patch_len // 2
        input = input[border1 - border_start - half_patch_len:border2 - border_start + half_patch_len]
        output = output[border1 - border_start - half_patch_len:border2 - border_start + half_patch_len]
        data_path = os.path.join('./test_results/UCR/', setting, self.args.data_path[:self.args.data_path.find('.')])
        file_path = data_path + '.pdf'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        visual(input, output, file_path)

        score_list.sort(key=lambda x: x[1], reverse=True)

        def is_overlap(index):
            start = index * self.args.patch_len + border_start
            end = start + self.args.patch_len
            if border1 <= start <= border2 or border1 <= end <= border2 or start <= border1 and end >= border2:
                return True
            else:
                return False

        # find the first overlap token's index
        topk = 0
        for i, (index, score) in enumerate(score_list):
            if is_overlap(index):
                topk = i
                break
        print("score_list: ", score_list)
        print('topk:', topk + 1)
        filename = 'ucr768_' + self.args.model + '.csv'
        results = [self.args.data_path, topk + 1, len(score_list)] + score_list
        with open(filename, 'a') as f:
            f.write(','.join([str(result) for result in results]) + '\n')

        return
