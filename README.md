# Eye2Gene

Public code repository for training the models in Eye2Gene ([eye2gene.com])  

# Setup

## nvidia libs 

(if you have used system packages to install cuda then you probably don't need this, try without it first)

Make sure you have included the cuda toolkit in your `LD_LIBRARY_PATH`.
Mainly you need `cuda/lib64`, `cuda-10.1`, `cuda-10.1/extras`, `cuda-10.1/extras/CUPTI/lib64` (tensorboard)

You also need to add `cuda-10.1/bin` to your path if you want things like `nvidia-smi`

e.g here is my setup:
```
export WORKDIR=/mnt/new_root/rilott
export PATH=$WORKDIR/cuda-10.1/bin:~/Python-3.8.3/:~/.local/bin:$PATH
export LD_LIBRARY_PATH=$WORKDIR/cuda-10.1/lib64:$WORKDIR/cuda/lib64:$WORKDIR/cuda-10.1/extras/CUPTI/lib64:$LD_LIBRARY_PATH
```

## Python packages

Use a virtual environment, or be sure to use `--user` when running `pip` as follows:
```
pip3 install --user -r requirements.txt
```


# Training

You can train a network by running `train.py` in the `bin/` directory
e.g `python3 bin/train.py --model inceptionv3 --epochs 50 --lr 1e-4 --batch-size 8 --lr-schedule poly --lr-power 2 --train-dir ../data/train/AF --val-dir ../data/test/AF --model-save-dir trained_models/ --model-log-dir logs/`

Data must be a directory where each folder corresponds to a different class, or can be supplied as a CSV file with columns `file.path` containing a list of filepaths to the relevant images, and `gene` containing the classname of the label of that particular image. You can either supply `--data-dir`, in which case the code will automatically plit this into train and validation sets, or you can supply separate train and validation datasets as above.

N.B: Ensure the folder for `--model-save-dir` exists, otherwise you will get errors.

```
usage: train.py [-h] [--augmentations AUGMENTATIONS] [--batch-size BATCH_SIZE]
                [--classes CLASSES [CLASSES ...]] [--cfg CFG [CFG ...]]
                [--dataseries-path DATASERIES_PATH]
                [--dataseries-label DATASERIES_LABEL] [--dropout DROPOUT]
                [--epochs EPOCHS] [--lr LR] [--lr-schedule {linear,poly}]
                [--lr-power LR_POWER] [--model-save-dir MODEL_SAVE_DIR]
                [--model-log-dir MODEL_LOG_DIR] [--no-weights] [--preview]
                [--split VALIDATION_SPLIT] [--data-dir DATA_DIR]
                [--train-dir TRAIN_DIR] [--val-dir VAL_DIR]
                [--workers WORKERS] [--verbose]
                {vgg16,inception_resnetv2,inceptionv3,custom,nasnetlarge}

positional arguments:
  {vgg16,inception_resnetv2,inceptionv3,custom,nasnetlarge}
                        Name of model to train

optional arguments:
  -h, --help            show this help message and exit
  --augmentations AUGMENTATIONS
                        Comma separated values containing augmentations e.g
                        horitzontal_flip=True,zoom=0.3
  --batch-size BATCH_SIZE
                        Batch size
  --classes CLASSES [CLASSES ...]
                        List of classes
  --cfg CFG [CFG ...]   Config file to load model config from
  --dataseries-path DATASERIES_PATH
                        Name of dataseries for image paths (if reading from
                        csv)
  --dataseries-label DATASERIES_LABEL
                        Name of dataseries for labels (if reading from csv)
  --dropout DROPOUT     Dropout probability
  --epochs EPOCHS       Number of epochs to train
  --lr LR               Learning rate
  --lr-schedule {linear,poly}
                        Learning rate scheduler
  --lr-power LR_POWER   Power of lr decay, only used when using polynomial
                        learning rate scheduler
  --model-save-dir MODEL_SAVE_DIR
                        Save location for trained models
  --model-log-dir MODEL_LOG_DIR
                        Save location for model logs (used by tensorboard)
  --no-weights          Don't download and use any pretrained model weights,
                        random init
  --preview             Preview a batch of augmented data and exit
  --split VALIDATION_SPLIT
                        Training/Test split (% of data to keep for training,
                        will be halved for validation and testing)
  --data-dir DATA_DIR   Full dataset directory (will be split into
                        train/val/test)
  --train-dir TRAIN_DIR
                        Training data (validation is taken from this)
  --val-dir VAL_DIR     Validation data (can be supplied if you do not want it
                        taken from training data
  --workers WORKERS     Number of workers to use when training
                        (multiprocessing)
  --verbose             Verbose
```

# Prediction

There is another script located at `bin/predict.py` which can be given a directory of images (in a structure keras can read), or a csv file, and a trained model. The script will then output percentages of correct predictions and a numpy file of the predictions and true classes for each image.

The model must be provided as an `.h5` file, then the script will search for a corresponding `.json` file containing the network config.

```
usage: predict.py [-h] image_dir model

positional arguments:
  image_dir
  model

optional arguments:
  -h, --help  show this help message and exit
```



# Saliency

Saliency maps can be generated using `bin/attribution.py`


# Pipeline

1. Start with a csv file with at least two columns, one with paths to the images with heading "file.path" and another with the gene labels with heading "gene". This can be generated using `utils/csv_generator.py` if needed.
1b. For phenotypes, run `utils/gene2phenotype.py` on the csv to get a new dataset csv with phenotypes column.
2. Split the csv file into 5 folds by using `utils/k_fold_generator.py`, this will create 10 additional csv files: one train and one validation file for each fold.
3. Train the model on a single fold using e.g. `python3 bin/train.py inceptionv3 --epochs 250 --train-dir datasets_norm/dataset1_folds_train_0.csv --val-dir datasets_norm/dataset1_folds_val_0.csv --model-save-dir trainedmodels --model-log-dir logs --cfg augmentations_nonorm.json 40class.json hparam_files/hparam_set_2.json`. Replace the cfg files with the apporpriate ones for your setting.
4. Repeat the above for the other 4 folds, updating the `--train-dir` and `--val-dire` files to the next fold.
4b. (Optional) Move `trainedmodels` and `logs` to new locations for safekeeping.

Note:
* Augmentations and image resizing are handled by keras. These can be changed by using different cfg files.

* If you want to update the root of the file-paths in a dataset csv to a new path you can use `utils/change_path.py`


