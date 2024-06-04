"""
This code is largely modified from the codebase of AASIST.
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import random
from typing import Union
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from models.backend import AASIST
try:
    from s3prl.nn import S3PRLUpstream
except NotFoundError:
    S3PRLUpstream = None
import fairseq
import argparse


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(1, 1),
            stride=1,
        )
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(
            in_channels=nb_filts[1],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(0, 1),
            stride=1,
        )

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(
                in_channels=nb_filts[0],
                out_channels=nb_filts[1],
                padding=(0, 1),
                kernel_size=(1, 3),
                stride=1,
            )

        else:
            self.downsample = False
        self.mp = nn.MaxPool2d((1, 3))  # self.mp = nn.MaxPool2d((1,4))

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x
        out = self.conv1(x)

        # print('out',out.shape)
        out = self.bn2(out)
        out = self.selu(out)
        # print('out',out.shape)
        out = self.conv2(out)
        # print('conv2 out',out.shape)
        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out

class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        out_channels,
        kernel_size,
        sample_rate=16000,
        in_channels=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
    ):
        super().__init__()
        filts = [70, [1, 32], [32, 32], [32, 64], [64, 64]]
        
        if in_channels != 1:

            msg = (
                "SincConv only support one input channel (here, in_channels = {%i})"
                % (in_channels)
            )
            raise ValueError(msg)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")
        
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )
        
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(
            -(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1
        )
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(
                2 * fmax * self.hsupp / self.sample_rate
            )
            hLow = (2 * fmin / self.sample_rate) * np.sinc(
                2 * fmin * self.hsupp / self.sample_rate
            )
            hideal = hHigh - hLow

            self.band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(hideal)

    def forward(self, x):
        band_pass_filter = self.band_pass.clone().to(x.device)
        self.filters = (band_pass_filter).view(self.out_channels, 1, self.kernel_size)
        x = x.unsqueeze(1)
        x = F.conv1d(
            x,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)
        # get embeddings using encoder
        # (#bs, #filt, #spec, #seq)
        x = self.encoder(x)
        return x

class Spectrogram(nn.Module):
    def __init__(self, device, sample_rate=16000, n_fft=512, win_length=512, hop_length=160, power=2, normalized=True):
        super(Spectrogram, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.power = power
        self.normalized = normalized
        
        filts = [70, [1, 32], [32, 32], [32, 64], [64, 64]]
        
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=self.power,
            normalized=self.normalized,
        ).to(device)
        
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )
        
        self.linear = nn.Linear((n_fft // 2 + 1) * 4, 23 * 29) # match the output shape of the rawnet encoder

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.spec(x)
        x = self.encoder(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.linear(x)
        x = x.view(x.size(0), x.size(1), 23, 29)
        return x
    
class MelSpectrogram(nn.Module):
    def __init__(self, device, sample_rate=16000, n_mels=80, n_fft=512, win_length=512, hop_length=160):
        super(MelSpectrogram, self).__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        
        filts = [70, [1, 32], [32, 32], [32, 64], [64, 64]]
        
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
        ).to(device)
        
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )
        
        self.linear = nn.Linear(n_mels * 4, 23 * 29) # match the output shape of the rawnet encoder

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.melspec(x)
        x = self.encoder(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.linear(x)
        x = x.view(x.size(0), x.size(1), 23, 29)
        return x
    
class LFCC(nn.Module):
    def __init__(self, device, sample_rate=16000, n_filter=20, f_min=0.0, f_max=None, n_lfcc=60, dct_type=2, norm="ortho", log_lf=False, speckwargs={"n_fft": 512, "win_length": 512, "hop_length": 160, "center": False}):
        super(LFCC, self).__init__()
        self.sample_rate = sample_rate
        self.n_filter = n_filter
        self.f_min = f_min
        self.f_max = f_max
        self.n_lfcc = n_lfcc
        self.dct_type = dct_type
        self.norm = norm
        self.log_lf = log_lf
        self.speckwargs = speckwargs
        
        filts = [70, [1, 32], [32, 32], [32, 64], [64, 64]]
        
        self.lfcc = torchaudio.transforms.LFCC(
            sample_rate=self.sample_rate,
            n_filter=self.n_filter,
            f_min=self.f_min,
            f_max=self.f_max,
            n_lfcc=self.n_lfcc,
            dct_type=self.dct_type,
            norm=self.norm,
            log_lf=self.log_lf,
            speckwargs=self.speckwargs,
        ).to(device)
        
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )
        
        self.linear = nn.Linear(n_lfcc * 4, 23 * 29) # match the output shape of the rawnet encoder

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.lfcc(x)
        x = self.encoder(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.linear(x)
        x = x.view(x.size(0), x.size(1), 23, 29)
        return x
    
class MFCC(nn.Module):
    def __init__(self, device, sample_rate=16000, n_mfcc=40, melkwargs={"n_fft": 512, "win_length": 512, "hop_length": 160, "center": False}):
        super(MFCC, self).__init__()
        self.device = device
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.melkwargs = melkwargs
        
        filts = [70, [1, 32], [32, 32], [32, 64], [64, 64]]
        
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs=self.melkwargs,
        ).to(device)
        
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )
        
        self.linear = nn.Linear(n_mfcc * 4, 23 * 29) # match the output shape of the rawnet encoder

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.mfcc(x)
        x = self.encoder(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.linear(x)
        x = x.view(x.size(0), x.size(1), 23, 29)
        return x
    
class SSLFrontend(nn.Module):
    def __init__(self, device, model_label, model_dim):
        super(SSLFrontend, self).__init__()
        if model_label == "xlsr":
            task_arg = argparse.Namespace(task='audio_pretraining')
            task = fairseq.tasks.setup_task(task_arg)
            # https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt
            model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(['/root/xlsr2_300m.pt'], task=task)
            self.model = model[0]
        self.device = device
        filts = [70, [1, 32], [32, 32], [32, 64], [64, 64]]

        self.sample_rate = 16000 # only 16000 setting is supported
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )
        self.linear = nn.Linear(model_dim * 2, 23 * 29)
        
    def extract_feature(self, x):
        if next(self.model.parameters()).device != x.device \
            or next(self.model.parameters()).dtype != x.dtype:
            self.model.to(x.device, dtype=x.dtype)
            self.model.train()
        emb = self.model(x, mask=False, features_only=True)['x']
        return emb
    
    def forward(self, x):
        x = self.extract_feature(x)
        x = x.transpose(1, 2).unsqueeze(1) # [batch, 1, seq, dim]
        x = self.encoder(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.linear(x)
        x = x.view(x.size(0), x.size(1), 23, 29)
        return x


class S3PRL(nn.Module):
    def __init__(self, device, model_label, model_dim):
        super(S3PRL, self).__init__()
        if S3PRLUpstream is None:
            raise ModuleNotFoundError("s3prl is not found, likely not installed, please install use `pip`")

        filts = [70, [1, 32], [32, 32], [32, 64], [64, 64]]

        self.sample_rate = 16000 # only 16000 setting is supported
        if model_label == "mms":
            self.model = S3PRLUpstream(
                "hf_wav2vec2_custom",
                path_or_url="facebook/mms-300m",
            ).to(device)
            print("Model has been sent to", device)
        else:
            self.model = S3PRLUpstream(model_label).to(device)
            print("Model has been sent to", device)

        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )
        self.linear = nn.Linear(model_dim * 2 * 64, 1) # match the output shape of the rawnet encoder

    def forward(self, x):
        print(x.size()) # expected: torch.Size([batch, 64000])
        x_lens = torch.LongTensor(x.size(0)).to(x.device)
        x, _ = self.model(x, x_lens)
        x = x[-1].transpose(1, 2).unsqueeze(1) # take the last hidden states
        # print(x.size())
        x = self.encoder(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.linear(x)
        x = x.view(x.size(0), 1)
        return x

class SVDDModel(nn.Module):
    def __init__(self, device, frontend=None):
        super(SVDDModel, self).__init__()
        assert frontend in ["rawnet", "spectrogram", "mel-spectrogram", "lfcc", "mfcc", "hubert", "mms", "xlsr", "mrhubert", "wavlablm"], "Invalid frontend"
        if frontend == "rawnet":
            # This follows AASIST's implementation
            self.frontend = SincConv(out_channels=70, kernel_size=128, in_channels=1)
        elif frontend == "spectrogram":
            self.frontend = Spectrogram(
                device=device,
                sample_rate=16000,
                n_fft=512,
                win_length=512,
                hop_length=160,
                power=2,
                normalized=True,
            )
        elif frontend == "mel-spectrogram":
            self.frontend = MelSpectrogram(
                device=device,
                sample_rate=16000,
                n_mels=80,
                n_fft=512,
                win_length=512,
                hop_length=160,
            )
        elif frontend == "lfcc":
            self.frontend = LFCC(
                device=device,
                sample_rate=16000,
                n_filter=20,
                f_min=0.0,
                f_max=None,
                n_lfcc=60,
                dct_type=2,
                norm="ortho",
                log_lf=False,
                speckwargs={
                    "n_fft": 512,
                    "win_length": 512,
                    "hop_length": 160,
                    "center": False,
                },
            )
        elif frontend == "mfcc":
            self.frontend = MFCC(
                device=device,
                sample_rate=16000,
                n_mfcc=40,
                melkwargs={
                    "n_fft": 512,
                    "win_length": 512,
                    "hop_length": 160,
                    "center": False,
                },
            )
        elif frontend == "hubert":
            self.frontend = S3PRL(
                device=device,
                model_label="hubert",
                model_dim=768,
            )
        elif frontend == "xlsr":
            self.frontend = SSLFrontend(
                device=device,
                model_label="xlsr",
                model_dim=1024,
            )
            print("after frontend")
        elif frontend == "mrhubert":
            self.frontend = S3PRL(
                device=device,
                model_label="multires_hubert_multilingual_large600k",
                model_dim=1024,
            )
        elif frontend == "wavlablm":
            self.frontend = S3PRL(
                device=device,
                model_label="wavlablm_ms_40k",
                model_dim=1024,
            )
        elif frontend == "mms":
            self.frontend = S3PRL(
                device=device,
                model_label="mms",
                model_dim=1024,
            )

        self.backend = AASIST()
    
    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x

if __name__ == "__main__":
    x = torch.randn(4, 64000)
    
    print("Testing RawNet Encoder")
    model = SVDDModel(frontend="rawnet")
    _, output = model(x)
    print(output.shape) # expected: torch.Size([4, 2])
    
    print("Testing Spectrogram Encoder")
    model = SVDDModel(frontend="spectrogram")
    _, output = model(x)
    print(output.shape) # expected: torch.Size([4, 1])
    
    print("Testing Mel-Spectrogram Encoder")
    model = SVDDModel(frontend="mel-spectrogram")
    _, output = model(x)
    print(output.shape) # expected: torch.Size([4, 1])
    
    print("Testing LFCC Encoder")
    model = SVDDModel(frontend="lfcc")
    _, output = model(x)
    print(output.shape) # expected: torch.Size([4, 1])
    
    print("Testing MFCC Encoder")
    model = SVDDModel(frontend="mfcc")
    _, output = model(x)
    print(output.shape) # expected: torch.Size([4, 1])