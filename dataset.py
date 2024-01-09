import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import pickle
import os, glob
import librosa
from torch.utils.data.dataloader import default_collate
import torchaudio
from utils import pad_random, get_utt_ids, get_meta_information
import pandas as pd


class SVDD2024(Dataset):
    def __init__(self, root_dir="data", target_sr=16000, partition="train"):
        self.root_dir = root_dir
        self.partition = partition
        self.target_sr = target_sr
        self.cut = 4 * self.target_sr  # take 4 sec audio (64000 samples)

        self.lfcc = torchaudio.transforms.LFCC(
            sample_rate=target_sr,
            n_filter=20,
            f_min=0.0,
            f_max=None,
            n_lfcc=60,  # Assuming you want the same number of coefficients as filters
            dct_type=2,
            norm="ortho",
            log_lf=False,
            speckwargs={"n_fft": 512, "win_length": 320, "hop_length": 160, "center": False},
        )

        self.df = pd.read_csv(os.path.join(root_dir, partition + ".txt"), 
        names=["original dataset", "singer", "filename", "-", "SVS/SVC methods", "label"], sep=" ")
        print(self.df.head())

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # read audio
        filename = self.df.iloc[index]["filename"]
        filepath = os.path.join(self.root_dir, self.partition + "_set", filename + ".flac")
        label = self.df.iloc[index]["label"]
        if label == "bonafide":
            label = 0
        else:
            label = 1
        X, _ = librosa.load(filepath, sr=self.target_sr, mono=False)
        # if not mono, take the first channel
        if len(X.shape) > 1:
            X = X[0]
        # if length < 1, re get
        if X.shape[0] < 1:
            print(f"{filepath} is too short (less than 1 sample.) Re-getting...")
            return self.__getitem__(np.random.randint(self.__len__()))
        # chop or pad to 4 sec
        X_pad = pad_random(X, self.cut)
        X_pad = librosa.util.normalize(X_pad)
        x_inp = Tensor(X_pad)
        x_inp = self.lfcc(x_inp.unsqueeze(0))

        return x_inp, label