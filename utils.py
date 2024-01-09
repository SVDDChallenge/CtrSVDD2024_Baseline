import os
import torch
import random
import numpy as np
from distutils import util
import re
import unicodedata

def str2bool(v):
    return bool(util.strtobool(v))

def setup_seed(random_seed, cudnn_deterministic=True):
    """ set_random_seed(random_seed, cudnn_deterministic=True)

    Set the random_seed for numpy, python, and cudnn

    input
    -----
      random_seed: integer random seed
      cudnn_deterministic: for torch.backends.cudnn.deterministic

    Note: this default configuration may result in RuntimeError
    see https://pytorch.org/docs/stable/notes/randomness.html
    """

    # # initialization
    # torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = False


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


def get_utt_ids(txt_file_name):
    with open(txt_file_name, "r") as f:
        lines = f.readlines()
    utt_ids = [line.strip().split("|")[0] for line in lines]
    return utt_ids


def convert_unicode_to_chinese(unicode_str):
    formatted_str = unicode_str.replace("#U", "\\U").ljust(10, '0')
    formatted_str = re.sub("[a-zA-Z0-9#]", "", formatted_str)
    return formatted_str


def get_other_meta_info_real_singing(path, meta_info, dataset):
    """
    Get other meta information of the real singing wav file.
            generation_method, svs_or_svc, original_split, song_id, speaker_id, sentence_id
    """
        # normalize unicode
    path = unicodedata.normalize('NFC', path)

    # Extract dataset information
    dir_lst = path.split("/")

    # get the id of "real_singing" or "fake_singing"
    idx = dir_lst.index("real_singing")

    meta_info["generation_method"] = "real"
    meta_info["svs_or_svc"] = "real"

    if dataset == "m4singer":
        # Extract split information
        meta_info["original_split"] = "unknown"
        
        # Extract song ID, speaker ID, and sentence ID
        # Song id:
        singer_and_song = dir_lst[idx + 2]
        meta_info["speaker_id"], meta_info["song_id"] = singer_and_song.split("#")
        meta_info["song_id"] = "#%s" % meta_info["song_id"]
        meta_info["song_id"] = convert_unicode_to_chinese(meta_info["song_id"])
        
        meta_info["sentence_id"] = os.path.basename(path)[:-4]

    elif dataset == "opencpop":
        meta_info["original_split"] = dir_lst[idx + 4]
        meta_info["speaker_id"] = "opencpop_speaker"
        # song id should be continous numbers in the format of 2044001628
        matches = re.findall("[0-9]{10}", path)
        meta_info["song_id"] = matches[0][:4]
        meta_info["sentence_id"] = matches[0][4:]

    elif dataset == "kising":
        meta_info["original_split"] = dir_lst[idx + 2]
        meta_info["speaker_id"] = "kising_speaker"
        # song id 
        _, meta_info["song_id"], meta_info["sentence_id"] = os.path.basename(path)[:-4].split("_")

    elif dataset == "kiritan":
        meta_info["original_split"] = "unknown"
        meta_info["speaker_id"] = "kirintan_speaker"
        basename_splits = os.path.basename(path)[:-4].split("_")
        meta_info["song_id"] = "_".join(basename_splits[:-1])
        meta_info["sentence_id"] = basename_splits[:-1]

    elif dataset == "ofuton":
        meta_info["original_split"] = "unknown"
        meta_info["speaker_id"] = "ofuton"
        basename_splits = os.path.basename(path)[:-4].split("_")
        meta_info["song_id"] = "_".join(basename_splits[:-1])
        meta_info["sentence_id"] = basename_splits[:-1]

    elif dataset == "oniku":
        meta_info["original_split"] = "unknown"
        meta_info["speaker_id"] = "oniku_speaker"
        basename_splits = os.path.basename(path)[:-4].split("_")
        meta_info["song_id"] = "_".join(basename_splits[:-1])
        meta_info["sentence_id"] = basename_splits[:-1]

    elif dataset == "acesinger":
        meta_info["original_split"] = "unknown"
        meta_info["speaker_id"] = dir_lst[idx + 2]
        _, meta_info["song_id"], meta_info["sentence_id"] = os.path.basename(path)[:-4].split("_")

    elif dataset == "jvsmusic":
        meta_info["original_split"] = "unknown"
        meta_info["speaker_id"] = "jvsmusic_speaker"
        meta_info["song_id"], meta_info["sentence_id"] = os.path.basename(path)[:-4].split("-")
        
    return meta_info

def get_other_meta_info_fake_singing(path, meta_info, dataset):    
    """
    Get other meta information of the fake singing wav file.
            generation_method, svs_or_svc, original_split, song_id, speaker_id, sentence_id
    """
    # normalize unicode
    path = unicodedata.normalize('NFC', path)

    # Extract dataset information
    dir_lst = path.split("/")

    # get the id of "real_singing" or "fake_singing"
    idx = dir_lst.index("fake_singing")

    if dataset == "m4singer": 
        meta_info["generation_method"] = dir_lst[idx + 2]
        # Extract split information
        split_match = re.search("(train|dev|test)", path)
        meta_info["original_split"] = split_match.group(0) if split_match else "unknown"
        
        # Extract song ID, speaker ID, and sentence ID
        # Song id:
        # find the first # in basename.
        first_hash_idx = path.find("#")
        meta_info["song_id"] = path[first_hash_idx:]
        # find the first ], _, -, /, or . in song_id.
        first_split_idx = min([meta_info["song_id"].find(i) for i in ["]", "_", "-", "/", "."] if meta_info["song_id"].find(i) != -1])
        meta_info["song_id"] = meta_info["song_id"][:first_split_idx]
        meta_info["song_id"] = convert_unicode_to_chinese(meta_info["song_id"])
        
        # search for all recurring Alto, Bass, Soprano, Tenor and pick the last one
        match_idx = [m.start() for m in re.finditer("Alto|Bass|Soprano|Tenor", path)]
        match_idx = match_idx[1] if len(match_idx) == 2 else match_idx[0]

        if path[match_idx] == "A" or path[match_idx] == "B":
            meta_info["speaker_id"] = path[match_idx:match_idx+6]
        elif path[match_idx] == "S":
            meta_info["speaker_id"] = path[match_idx:match_idx+9]
        elif path[match_idx] == "T":
            meta_info["speaker_id"] = path[match_idx:match_idx+7]
        matches = re.findall("[0-9]{4}", path)
        meta_info["sentence_id"] = matches[-1]
        
    elif dataset == "opencpop":
        meta_info["generation_method"] = dir_lst[idx + 2]
        split_match = re.search("(train|dev|test)", path)
        meta_info["original_split"] = split_match.group(0) if split_match else "unknown"
        meta_info["speaker_id"] = "opencpop_speaker"
        # song id should be continous numbers in the format of 2044001628
        matches = re.findall("[0-9]{10}", path)
        meta_info["song_id"] = matches[0][:4]
        meta_info["sentence_id"] = matches[0][4:]

    elif dataset == "jvsmusic":
        meta_info["generation_method"] = "nusvc" # TODO: not sure
        # Extract split information
        meta_info["original_split"] = dir_lst[idx + 3]
        ## TODO: get the song id and speaker id and sentence id
        ## TODO: the jvsmusic and m4singer need to be considered separately

    elif dataset == "kising":
        meta_info["generation_method"] = dir_lst[idx + 2]
        meta_info["original_split"] = dir_lst[idx + 3]
        
        meta_info["speaker_id"], meta_info["song_id"], meta_info["sentence_id"] = os.path.basename(path)[:-4].split("_")

    elif dataset == "kiritan":
        meta_info["generation_method"] = dir_lst[idx + 2]
        meta_info["original_split"] = dir_lst[idx + 3]
        meta_info["speaker_id"] = "kiritan_speaker"
        meta_info["song_id"] = os.path.basename(path).split("_")[0][-2:]
        meta_info["sentence_id"] = os.path.basename(path)[:-4].split("_")[1]
        if "svc" in meta_info["generation_method"]:
            convert_dataset = meta_info["generation_method"].split("_")[1]
            if convert_dataset == "m4singer":
                # find the first two numbers in the basename as the song id
                meta_info["song_id"] = re.findall("[0-9]{2}", os.path.basename(path))[0]
                # find the first "Alto", "Bass", "Soprano", "Tenor" in the basename as the speaker id
                match_idx = [m.start() for m in re.finditer("Alto|Bass|Soprano|Tenor", path)]
                match_idx = match_idx[1] if len(match_idx) == 2 else match_idx[0]
                if path[match_idx] == "A" or path[match_idx] == "B":
                    meta_info["speaker_id"] = path[match_idx:match_idx+6]
                elif path[match_idx] == "S":
                    meta_info["speaker_id"] = path[match_idx:match_idx+9]
                elif path[match_idx] == "T":
                    meta_info["speaker_id"] = path[match_idx:match_idx+7]
            elif convert_dataset == "jvsmusic":
                meta_info["song_id"] = re.findall("[0-9]{2}", os.path.basename(path))[0]
                meta_info["speaker_id"] = os.path.basename(path).split("_")[13]

            meta_info["generation_method"] = "nusvc"
            
            meta_info["sentence_id"] = os.path.basename(path).split("_")[6]

    elif dataset == "ofuton":  ## TODO: need changes
        meta_info["generation_method"] = dir_lst[idx + 2]
        meta_info["original_split"] = dir_lst[idx + 3]
        meta_info["speaker_id"] = "ofuton_speaker"
        meta_info["song_id"] = os.path.basename(path).split("_")[0][-2:]
        meta_info["sentence_id"] = os.path.basename(path)[:-4].split("_")[1]
        if "svc" in meta_info["generation_method"]:
            convert_dataset = meta_info["generation_method"].split("_")[1]
            if convert_dataset == "m4singer":
                # find the first two numbers in the basename as the song id
                meta_info["song_id"] = re.findall("[0-9]{2}", os.path.basename(path))[0]
                # find the first "Alto", "Bass", "Soprano", "Tenor" in the basename as the speaker id
                match_idx = [m.start() for m in re.finditer("Alto|Bass|Soprano|Tenor", path)]
                match_idx = match_idx[1] if len(match_idx) == 2 else match_idx[0]
                if path[match_idx] == "A" or path[match_idx] == "B":
                    meta_info["speaker_id"] = path[match_idx:match_idx+6]
                elif path[match_idx] == "S":
                    meta_info["speaker_id"] = path[match_idx:match_idx+9]
                elif path[match_idx] == "T":
                    meta_info["speaker_id"] = path[match_idx:match_idx+7]
            elif convert_dataset == "jvsmusic":
                meta_info["song_id"] = re.findall("[0-9]{2}", os.path.basename(path))[0]
                meta_info["speaker_id"] = os.path.basename(path).split("_")[13]

            meta_info["generation_method"] = "nusvc"
            
            meta_info["sentence_id"] = os.path.basename(path).split("_")[6]

    elif dataset == "oniku": ## TODO: check
        meta_info["generation_method"] = dir_lst[idx + 2]
        meta_info["original_split"] = dir_lst[idx + 3]
        meta_info["speaker_id"] = "oniku_speaker"
        meta_info["song_id"] = os.path.basename(path).split("_")[0][-2:]
        meta_info["sentence_id"] = os.path.basename(path)[:-4].split("_")[1]
        if "svc" in meta_info["generation_method"]:
            convert_dataset = meta_info["generation_method"].split("_")[1]
            if convert_dataset == "m4singer":
                # find the first two numbers in the basename as the song id
                meta_info["song_id"] = re.findall("[0-9]{2}", os.path.basename(path))[0]
                # find the first "Alto", "Bass", "Soprano", "Tenor" in the basename as the speaker id
                match_idx = [m.start() for m in re.finditer("Alto|Bass|Soprano|Tenor", path)]
                match_idx = match_idx[1] if len(match_idx) == 2 else match_idx[0]
                if path[match_idx] == "A" or path[match_idx] == "B":
                    meta_info["speaker_id"] = path[match_idx:match_idx+6]
                elif path[match_idx] == "S":
                    meta_info["speaker_id"] = path[match_idx:match_idx+9]
                elif path[match_idx] == "T":
                    meta_info["speaker_id"] = path[match_idx:match_idx+7]
            elif convert_dataset == "jvsmusic":
                meta_info["song_id"] = re.findall("[0-9]{2}", os.path.basename(path))[0]
                meta_info["speaker_id"] = os.path.basename(path).split("_")[13]
            meta_info["generation_method"] = "nusvc"
            meta_info["sentence_id"] = os.path.basename(path).split("_")[6]

    elif dataset == "acesinger":
        meta_info["generation_method"] = "acesinger"
        meta_info["original_split"] = dir_lst[idx + 2]
        basename = os.path.basename(path)
        basename_splits = basename.split("_")
        meta_info["speaker_id"] = basename_splits[0]
        meta_info["song_id"] = "_".join(basename_splits[2:-1]) ## TODO: need double-check
        meta_info["sentence_id"] = basename_splits[-1][:-4]
    
    try:
        meta_info["svs_or_svc"] = "svs" if meta_info["generation_method"] in ["rnn", "visinger", "visinger2", "diffsinger", "xiaoice"] else "svc"
    except:
        print(f"Error loading {path}")
        
    return meta_info


def get_meta_information(path):
    """
    Get meta information of the wav file.

    :param path: path to the wav file
    :return: meta information of the wav file
    """
    ## all the meta information includes: path, basename, original_dataset, real_or_fake, label, 
    ## \ generation_method, svs_or_svc, original_split, song_id, speaker_id, sentence_id
    meta_info = {}
    meta_info['path'] = path

    # normalize unicode
    path = unicodedata.normalize('NFC', path)

    # Extract basename
    meta_info['basename'] = os.path.basename(path)[:-4]

    # Extract dataset information
    dir_lst = path.split("/")
    
    # get the id of "real_singing" or "fake_singing"
    idx = dir_lst.index("real_singing") if "real_singing" in dir_lst else dir_lst.index("fake_singing")
    
    dataset = dir_lst[idx + 1].lower()
    meta_info["original_dataset"] = dataset
    meta_info["real_or_fake"] = dir_lst[idx]
    if meta_info["real_or_fake"] == "real_singing":
        meta_info["label"] = "bonafide"
        meta_info = get_other_meta_info_real_singing(path, meta_info, dataset)
    else:
        meta_info["label"] = "deepfake"
        meta_info = get_other_meta_info_fake_singing(path, meta_info, dataset)

    return meta_info


if __name__ == "__main__":
    # path = "/dataHDD/neil/SVDD_challenge/fake_singing/m4singer/sovits_mrhubert/train/m4singer_Tenor-7#香格里拉#0004.wav"
    # path = "/dataHDD/neil/SVDD_challenge/fake_singing/decode_diffsinger_m4singer/[Tenor-7##U9999#U683c#U91cc#U62c9#0014][P]-m4singer_diff_e2e.wav"
    # path = "/dataHDD/neil/SVDD_challenge/fake_singing/m4singer/diffsinger/[Tenor-7##U9999#U683c#U91cc#U62c9#0014][P]-m4singer_diff_e2e.wav"
    # path = "/dataHDD/neil/SVDD_challenge/fake_singing/m4singer/sovits_mrhubert/train/m4singer_Soprano-1#今宵如此美丽#0023.wav"
    # path = "/dataHDD/neil/SVDD_challenge/fake_singing/m4singer/nusvc_m4singer/train/singing_zh_m4singer_Tenor-7#阿楚姑娘_0014_to_singing_zh_m4singers_Tenor-5#能走多远_0021_gen.wav"
    path = "/dataHDD/neil/SVDD_challenge/fake_singing/jvsmusic/jvsmusic_m4singer/train/singing_ja_jvs_jvs_music_jvs100_song_unique_wav_raw_chunk-04_to_singing_zh_m4singer_Tenor-6#故梦_0025_gen.wav"
    print(get_meta_information(path))
