import contextlib
import os
import random

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import torch
from torch.utils.data.dataloader import DataLoader
# from torchsummary import summary

from hn_cnn.cnn import MODELS
from hn_cnn.constants import *
from  hn_cnn.data_augmentation import get_training_augmentation, TRANSFORM_TO_TENSOR
from hn_cnn.fit import fit
from hn_cnn.parse_data import ImageDataset

def run_network(folders, config, ids):
    """ Run the network according to the configuration provided
        - folders: paths to the folders for training, validation, and testing
        - config: configuration regarding the network to train, data split, ...
        - ids: ids for each subset
    """
    dataloaders = {}
    # Load datasets
    # Include a training dataset that won't be augmented in order to
    # calculate the performance metrics
    folders[TRAIN_METRICS] = folders[TRAIN]
    is_neural_network = config[MODEL] in [CNN, ANN]
    for dataset in [TRAIN, TRAIN_METRICS, VALIDATION, TESTING]:
        paths = DATA[dataset]
        dataset_ids = ids[dataset]
        is_training = (dataset == TRAIN)
        dataset_parsed = ImageDataset(
            paths[CLINICAL_DATA_PATH],
            paths[SCANS_PATH],
            get_training_augmentation(augment=config[DATA_AUGMENTATION]) if is_training else TRANSFORM_TO_TENSOR,
            ids_to_use=dataset_ids,
            timeframe=config[TIME_TO_EVENT],
            event=config[EVENT],
        )
        dataloaders[dataset] = DataLoader(
            dataset_parsed,
            batch_size = config[BATCH_SIZE] if is_training and is_neural_network else len(dataset_parsed.images),
            shuffle = is_training,
            drop_last = is_training,
        )

    model = MODELS[config[MODEL]](
        clinical_data=config.get(CLINICAL_VARIABLES),
    )
    print(model)
    # Print a summary of the model
    # summary(model, (1, 180, 180))
    history = fit(
        model,
        dataloaders,
        parameters=config[HYPERPARAMETERS],
    )
    return history

DATA = {
    TRAIN: {
        CLINICAL_DATA_PATH: os.getenv(f"{TRAIN}_{CLINICAL_DATA_PATH}"),
        SCANS_PATH: os.getenv(f"{TRAIN}_{SCANS_PATH}"),
    },
    VALIDATION: {
        CLINICAL_DATA_PATH: os.getenv(f"{VALIDATION}_{CLINICAL_DATA_PATH}"),
        SCANS_PATH: os.getenv(f"{VALIDATION}_{SCANS_PATH}"),
    },
    TESTING: {
        CLINICAL_DATA_PATH: os.getenv(f"{TESTING}_{CLINICAL_DATA_PATH}"),
        SCANS_PATH: os.getenv(f"{TESTING}_{SCANS_PATH}"),
    }
}

CONFIGURATIONS = {
    MODEL: LR,
    DATA_SPLIT: COHORT_SPLIT,
    FOLDS: 2,
    TIME_TO_EVENT: 2 * 365,
    EVENT: DM,
    HYPERPARAMETERS: {
        # Include any hypeparameters that you want to change
        # from the default ones:
        LEARNING_RATE: 0.05,
        EPOCHS: 5,
    },
    BATCH_SIZE: 64,
    LOGS_PATH: "/mnt/log.txt",
    # Leave empty to train the network without clinical data
    CLINICAL_VARIABLES: [],
    DATA_AUGMENTATION: {},
}

# Set the seeds
# 1) Global python seed
# random_seed = random.randint(..., ...)
# print(f"Random seed: {random_seed}")
random_seed = 0
random.seed(random_seed)
# 2) Random split seed
# random_seed_split = random.randint(..., ...)
random_seed_split = 0
# 3) Random seed torch
# random_seed_torch = random.randint(..., ...)
random_seed_torch = 0
torch.manual_seed(random_seed_torch)

if CONFIGURATIONS[DATA_SPLIT] == COHORT_SPLIT:
    # Store the model training logs in a file
    file_path = CONFIGURATIONS[LOGS_PATH]
    dataset_train = ImageDataset(
        DATA[TRAIN][CLINICAL_DATA_PATH],
        DATA[TRAIN][SCANS_PATH],
        TRANSFORM_TO_TENSOR,
        timeframe=CONFIGURATIONS[TIME_TO_EVENT],
        event=CONFIGURATIONS[EVENT],
    )
    # with open(file_path, "w") as o:
    #     with contextlib.redirect_stdout(o):
    dataloaders = {}
    train_ids = [id for id in dataset_train.keys if "HGJ" in id or "CHUS" in id]
    validation_ids =  [id for id in dataset_train.keys if "HMR" in id or "CHUM" in id]

    DATA[TRAIN_METRICS] = DATA[TRAIN]
    dataset_ids = {}
    for dataset in [TRAIN, TRAIN_METRICS, VALIDATION, TESTING]:
        is_training = (dataset == TRAIN)
        dataset_ids[dataset] = []
        if dataset in [TRAIN, TRAIN_METRICS, VALIDATION]:
            dataset_ids[dataset] = train_ids if is_training or dataset == TRAIN_METRICS else validation_ids
    run_network(DATA, CONFIGURATIONS, dataset_ids)

elif CONFIGURATIONS[DATA_SPLIT] == CROSS_VALIDATION:
    kfold = StratifiedKFold(n_splits=CONFIGURATIONS[FOLDS], shuffle=True, random_state=random_seed_split)
    dataset_train = ImageDataset(
        DATA[TRAIN][CLINICAL_DATA_PATH],
        DATA[TRAIN][SCANS_PATH],
        TRANSFORM_TO_TENSOR,
        timeframe=CONFIGURATIONS[TIME_TO_EVENT],
        event=CONFIGURATIONS[EVENT],
    )
    for fold, (train_ids, validation_ids) in enumerate(kfold.split(dataset_train, np.array(dataset_train.y))):
        print(f"Fold: {fold}")
        # Store the model training logs in a file
        file_path = CONFIGURATIONS[LOGS_PATH]
        #with open(file_path, "w") as o:
        #    with contextlib.redirect_stdout(o):
        DATA[TRAIN_METRICS] = DATA[TRAIN]
        dataset_ids = {}
        for dataset in [TRAIN, TRAIN_METRICS, VALIDATION, TESTING]:                
            is_training = (dataset == TRAIN)
            dataset_ids[dataset] = []
            if dataset in [TRAIN, TRAIN_METRICS, VALIDATION]:
                dataset_ids[dataset] = [dataset_train.keys[i] for i in train_ids] if \
                    is_training or dataset == TRAIN_METRICS else [dataset_train.keys[i] for i in validation_ids]
        run_network(DATA, CONFIGURATIONS, dataset_ids)
