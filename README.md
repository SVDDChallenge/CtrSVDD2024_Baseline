# SVDD Challenge Baseline Systems (CtrSVDD Track)

Publications: [[SVDD Challenge overview paper @ SLT 2024](https://arxiv.org/pdf/2408.16132)] [[CtrSVDD dataset paper @ Interspeech 2024](https://www.isca-archive.org/interspeech_2024/zang24_interspeech.pdf)]

Datasets: [[CtrSVDD (Train+Dev Set)](https://zenodo.org/records/10467648)] [[CtrSVDD (Eval Set)](https://zenodo.org/records/12703261)] [[WildSVDD](https://zenodo.org/records/10893604)]

This repository contains the baseline system implementations for the SVDD Challenge 2024. To form a comprehensive evaluation, we implemented the front-end features, back-end systems, and evaluation metrics. The baseline systems are implemented in Python and are available as open-source software.

## Updates
[Sep 2024] Our WildSVDD challenge proposal has been accepted by MIREX@ISMIR 2024. Stay tuned on the [challenge website](https://www.music-ir.org/mirex/wiki/2024:Singing_Voice_Deepfake_Detection)!

[Aug 2024] Our SVDD challenge overview paper has been accepted by [SLT 2024](https://arxiv.org/abs/2408.16132).

[Jun 2024] Our CtrSVDD paper has been accepted by [INTERSPEECH 2024](https://www.isca-archive.org/interspeech_2024/zang24_interspeech.html)! We update all five baseline system implementations in `models/model.py`, the corresponding model weights in `weights/`, and our training script.

[March 2024]  We updated training logs in `weights/training_logs`. You can see them using tensorboard (see 'Visualize Training Logs of Provided Baseline Systems'). Also, we realized our training code contains flipped labels (the bonafide are labeled as 0, not 1). The code has been fixed to reflect the correct implementation. EER may change slightly due to this. Since our code contains flipped labels during baseline training, you need to manually flip the sign of the predicted scores if you are only inferencing from our provided baseline systems. To do so, please add a line in `eval.py` after 31: `pred *= -1.0`.

## Getting Started

Setting up the environment:
```bash
conda create -n svdd_baseline python=3.10
conda activate svdd_baseline
pip install -r requirements.txt
```

Getting the dataset:
Please find the dataset on the Zenodo links above. Please note that the training and development sets available on Zenodo are incomplete because of licensing issues of some bonafide datasets. You will need to follow [https://github.com/SVDDChallenge/CtrSVDD_Utils](https://github.com/SVDDChallenge/CtrSVDD_Utils) to retrieve full datasets.

Then you can run the training script with the following command:
```bash
python train.py --base_dir {Where the data is} --gpu {GPU ID} --encoder {Encoder Type} --batch_size {Batch size}
```
You can use `--load_from` flag to resume training.

After training, you can evaluate your model using the following command:
```bash
python eval.py --base_dir {Where the data is} --model_path {The model's weights file} --gpu {GPU ID} --encoder {Encoder Type} --batch_size {Batch size}
```

The main functions in `train` and `eval` specify more options that you can tune. 

Within `base_dir`, the code expects to see `train_set`, `dev_set` and `test_set` directories, along with `train.txt` and `dev.txt` as open-sourced. `train_set`, `dev_set` and `test_set` should directly contain `*.flac` files.

# Visualize Training Logs of Provided Baseline Systems
Run the following command within the CtrSVDD2024_Baseline directory.

```bash
pip install tensorboard
tensorboard --logdir weights/training_logs
```

