import numpy as np
import os
from torch.utils.data import Dataset
import librosa

def pad_random(x: np.ndarray, max_len: int = 64000):
    x_len = x.shape[0]
    if x_len > max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))
    return pad_random(padded_x, max_len)

class SVDD2024(Dataset):
    """
    Dataset class for the SVDD 2024 dataset.
    """
    def __init__(self, base_dir, partition="train", max_len=64000):
        assert partition in ["train", "dev", "test"], "Invalid partition. Must be one of ['train', 'dev', 'test']"
        self.base_dir = base_dir
        self.partition = partition
        self.base_dir = os.path.join(base_dir, partition + "_set")
        self.max_len = max_len
        try:
            with open(os.path.join(base_dir, f"{partition}.txt"), "r") as f:
                self.file_list = f.readlines()
        except FileNotFoundError:
            if partition == "test":
                self.file_list = []
                # get all *.flac files in the test_set directory
                for root, _, files in os.walk(self.base_dir):
                    for file in files:
                        if file.endswith(".flac"):
                            self.file_list.append(file)
            else:
                raise FileNotFoundError(f"File {partition}.txt not found in {base_dir}")
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):            
        if self.partition == "test":
            file_name = self.file_list[index].strip()
            label = 0 # dummy label. Not used for test set.
        else:
            file = self.file_list[index]
            file_name = file.split(" ")[2].strip()
            bonafide_or_spoof = file.split(" ")[-1].strip()
            label = 0 if bonafide_or_spoof == "bonafide" else 1
        try:
            x, _ = librosa.load(os.path.join(self.base_dir, file_name + ".flac"), sr=16000, mono=True)
            x = pad_random(x, self.max_len)
            x = librosa.util.normalize(x)
            # file_name is used for generating the score file for submission
            return x, label, file_name
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            return None