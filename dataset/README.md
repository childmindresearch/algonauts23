# Algonauts 2023 Huggingface dataset

Here we include some scripts for organizing the [Algonauts 2023 dataset](https://codalab.lisn.upsaclay.fr/competitions/9304#participate) in the [Huggingface format](https://huggingface.co/docs/datasets/create_dataset).

### 1. Download the official dataset

Run `download_data.sh` to download the official challenge data. Make sure you have [gdown](https://github.com/wkentaro/gdown) installed.

```bash
bash download_data.sh
```

You can also copy or link the challenge data to `algonauts_2023_challenge_data/` if you already have it downloaded.

### 2. Generate splits

Generate private splits derived from the official training split. This will generate a list of indices for each subject and split (train, val, testval) saved in npy format in `derived_splits/`.

```bash
python generate_splits.py
```

### 3. Generate the processed dataset

Generate a processed huggingface dataset for each split and fixed image size. Note, this may take a few hours.

```bash
splits="train val testval test"
for split in $splits; do
    python generate_dataset.py --split $split --img_size 256 --workers 16
done
```
