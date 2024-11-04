import torch
from torch import nn

from models import TimerBackbone


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.ckpt_path = configs.ckpt_path
        self.patch_len = configs.patch_len
        self.stride = configs.patch_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.layers = configs.e_layers
        self.n_heads = configs.n_heads
        self.dropout = configs.dropout

        self.output_attention = configs.output_attention

        self.backbone = TimerBackbone.Model(configs)
        # Decoder
        self.decoder = self.backbone.decoder
        self.proj = self.backbone.proj
        self.enc_embedding = self.backbone.patch_embedding


        if self.ckpt_path != '':
            if self.ckpt_path == 'random':
                print('loading model randomly')
            else:
                print('loading model: ', self.ckpt_path)
                if self.ckpt_path.endswith('.pth'):
                    self.backbone.load_state_dict(torch.load(self.ckpt_path))
                elif self.ckpt_path.endswith('.ckpt'):
                    sd = torch.load(self.ckpt_path, map_location="cpu")["state_dict"]
                    sd = {k[6:]: v for k, v in sd.items()}
                    self.backbone.load_state_dict(sd, strict=True)

                else:
                    raise NotImplementedError

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, M = x_enc.shape

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1) # [B, M, T]
        dec_in, n_vars = self.enc_embedding(x_enc) # [B * M, N, D]

        # Transformer Blocks
        dec_out, attns = self.decoder(dec_in) # [B * M, N, D]
        dec_out = self.proj(dec_out) # [B * M, N, L]
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2) # [B, T, M]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev + means
        if self.output_attention:
            return dec_out, attns
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, M = x_enc.shape
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1) # [B, M, T]
        dec_in, n_vars = self.enc_embedding(x_enc) # [B * M, N, D]

        # Transformer Blocks
        dec_out, attns = self.decoder(dec_in) # [B * M, N, D]
        dec_out = self.proj(dec_out) # [B * M, N, L]
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2) # [B, T, M]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev + means
        return dec_out

    def anomaly_detection(self, x_enc):
        B, L, M = x_enc.shape

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1) # [B, M, T]
        dec_in, n_vars = self.enc_embedding(x_enc) # [B * M, N, D]

        # Transformer Blocks
        dec_out, attns = self.decoder(dec_in) # [B * M, N, D]
        dec_out = self.proj(dec_out) # [B * M, N, L]
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2) # [B, T, M]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev + means
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out  # [B, T, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, T, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, T, D]

        raise NotImplementedError

