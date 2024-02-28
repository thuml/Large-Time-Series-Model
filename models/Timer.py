import torch
from torch import nn


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2402.02368.pdf
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        if configs.ckpt_path != '' and configs.ckpt_path.endswith('.pt'):
            self.timer = torch.jit.load(configs.ckpt_path)
        else:
            raise NotImplementedError

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.timer(x_enc, x_mark_enc, x_dec, x_mark_dec)
