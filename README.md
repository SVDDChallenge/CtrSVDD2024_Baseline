# SVDD Challenge Baseline Systems
This repository contains the baseline system implementations for the SVDD Challenge 2024. To form a comprehensive evaluation, we implemented the front-end features, back-end systems and the evaluation metrics. The baseline systems are implemented in Python and are available as open-source software.

# Updates
- March 12, 2024: Since during baseline training, our code contains flipped labels, you need to manually flip the sign of the predicted scores if you are only inferencing from our provided baseline systems. To do so, please add a line in `eval.py` after 31: `pred *= -1.0`.
- March 6, 2024: We update training logs in `weights/training_logs`. You could see them using tensorboard. Also, we realize our training code contains flipped labels (the bonafides are labeled as 0, not 1). The code has been fixed to reflect the correct implementation. EER may change slightly due to this.

# Getting Started

Setting up environment:
```bash
conda create -n svdd_baseline python=3.10
conda activate svdd_baseline
pip install -r requirements.txt
```

Then you can run the training script with the following command:
```bash
python train.py --base_dir {Where the data is} --gpu {GPU ID} --encoder {Encoder Type} --batch_size {Batch size}
```

After training, you can evaluate your model using the following command:
```bash
python eval.py --base_dir {Where the data is} --model_path {The model's weights file} --gpu {GPU ID} --encoder {Encoder Type} --batch_size {Batch size}
```

The main functions in `train` and `eval` specify more options that you can tune. 

Within `base_dir`, the code expects to see `train_set`, `dev_set` and `test_set` directories, along with `train.txt` and `dev.txt` as open-sourced. `train_set`, `dev_set` and `test_set` should directly contain `*.flac` files.
