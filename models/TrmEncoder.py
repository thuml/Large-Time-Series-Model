import torch
import torch.nn as nn

from models import TrmEncoderBackbone


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # [B, M, D, N]
        x = self.flatten(x) # [B, M, D * N]
        x = self.linear(x) # [B, M, S]
        x = self.dropout(x) # [B, M, S]
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.input_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.stride = configs.patch_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.layers = configs.e_layers
        self.n_heads = configs.n_heads
        self.dropout = configs.dropout
        self.ckpt_path = configs.ckpt_path

        self.output_attention = configs.output_attention

        self.backbone = TrmEncoderBackbone.Model(configs)

        self.encoder = self.backbone.encoder
        self.head = self.backbone.head
        self.patch_embedding = self.backbone.patch_embedding


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
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1) # [B, M, T]
        enc_out, n_vars = self.patch_embedding(x_enc) # [B * M, N, D]
        # Encoder
        enc_out, attns = self.encoder(enc_out) # [B * M, N, D]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])) # [B, M, N, D]
        enc_out = enc_out.permute(0, 1, 3, 2) # [B, M, D, N]

        # Decoder
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1) # [B, S, M]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev + means

        if self.output_attention:
            return dec_out, attns
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
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
        enc_out, n_vars = self.patch_embedding(x_enc) # [B * M, N, D]
        # Encoder
        enc_out, attns = self.encoder(enc_out) # [B * M, N, D]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])) # [B, M, N, D]
        enc_out = enc_out.permute(0, 1, 3, 2) # [B, M, D, N]

        # Decoder
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1) # [B, S, M]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev + means

        return dec_out

    def anomaly_detection(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1) # [B, M, T]
        enc_out, n_vars = self.patch_embedding(x_enc) # [B * M, N, D]
        # Encoder
        enc_out, attns = self.encoder(enc_out) # [B * M, N, D]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])) # [B, M, N, D]
        enc_out = enc_out.permute(0, 1, 3, 2) # [B, M, D, N]

        # Decoder
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1) # [B, S, M]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev + means

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out  # [B, S, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, S, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out  # [B, S, D]

        raise NotImplementedError

