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

DATA = {
    TRAIN: {
        CLINICAL_DATA_PATH: "./canada.csv",
        SCANS_PATH: "./pre-processed/canada_c",
    },
    VALIDATION: {
        CLINICAL_DATA_PATH: "./canada.csv",
        SCANS_PATH: "./pre-processed/canada_c",
    },
    TESTING: {
        CLINICAL_DATA_PATH: "./maastro.csv",
        SCANS_PATH: "./pre-processed/maastro_c",
    }
}

CONFIGURATIONS = {
    MODEL: CNN,
    DATA_SPLIT: COHORT_SPLIT,
    # FOLDS: 2,
    TIME_TO_EVENT: 2 * 365,
    EVENT: LRF,
    HYPERPARAMETERS: {
        # Include any hypeparameters that you want to change
        # from the default ones:
        #LEARNING_RATE: 0.05,
        #EPOCHS: 5,
    },
    BATCH_SIZE: 64,
    LOGS_PATH: "/mnt/logs/log_lrf_cd.txt",
    # Leave empty to train the network without clinical data
    CLINICAL_VARIABLES: True,
    DATA_AUGMENTATION: {},
    STORE_MODEL: {
        MODEL_ID: "lrf_cd",
        MODEL_PATH: "/mnt/backup",
        THRESHOLD: 0.65,
        MAX_DIFFERENCE: 0.15,
    }
}

# Set the seeds
# 1) Global python seed
# random_seed = random.randint(..., ...)
# print(f"Random seed: {random_seed}")
random.seed(1892101)
# 2) Random split seed
# random_seed_split = random.randint(..., ...)
random_seed_split = random.randint(0, 9174937)
# 3) Random seed torch
torch.manual_seed(4951952)

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
    with open(file_path, "w") as o:
        with contextlib.redirect_stdout(o):
            dataloaders = {}
            train_ids = [id for id in dataset_train.keys if "HGJ" in id or "CHUS" in id]
            validation_ids =  [id for id in dataset_train.keys if "HMR" in id or "CHUM" in id]
            # Load datasets
            # Include a training dataset that won't be augmented in order to
            # calculate the performance metrics
            DATA[TRAIN_METRICS] = DATA[TRAIN]
            is_neural_network = CONFIGURATIONS[MODEL] in [CNN, ANN]
            for dataset in [TRAIN, TRAIN_METRICS, VALIDATION, TESTING]:
                paths = DATA[dataset]
                dataset_ids = []
                is_training = (dataset == TRAIN)
                if dataset in [TRAIN, TRAIN_METRICS, VALIDATION]:
                    dataset_ids = train_ids if is_training or dataset == TRAIN_METRICS else validation_ids
                dataset_parsed = ImageDataset(
                    paths[CLINICAL_DATA_PATH],
                    paths[SCANS_PATH],
                    get_training_augmentation(augment=CONFIGURATIONS[DATA_AUGMENTATION]) if is_training else TRANSFORM_TO_TENSOR,
                    ids_to_use=dataset_ids,
                    timeframe=CONFIGURATIONS[TIME_TO_EVENT],
                    event=CONFIGURATIONS[EVENT],
                )
                dataloaders[dataset] = DataLoader(
                    dataset_parsed,
                    batch_size = CONFIGURATIONS[BATCH_SIZE] if is_training and is_neural_network else len(dataset_parsed.images),
                    shuffle = is_training,
                    drop_last = is_training,
                )
            model = MODELS[CONFIGURATIONS[MODEL]](
                clinical_data=CONFIGURATIONS.get(CLINICAL_VARIABLES),
            )
            print(model)
            # Print a summary of the model
            # summary(NaturalSceneClassification(), (1, 180, 180))
            history = fit(
                model,
                dataloaders,
                parameters=CONFIGURATIONS[HYPERPARAMETERS],
                store_model=CONFIGURATIONS[STORE_MODEL],
            )
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
            with open(file_path, "w") as o:
                with contextlib.redirect_stdout(o):
                    dataloaders = {}
                    # Load datasets
                    # Include a training dataset that won't be augmented in order to
                    # calculate the performance metrics
                    DATA[TRAIN_METRICS] = DATA[TRAIN]
                    is_neural_network = CONFIGURATIONS[MODEL] in [CNN, ANN]
                    for dataset in [TRAIN, TRAIN_METRICS, VALIDATION, TESTING]:
                        paths = DATA[dataset]
                        dataset_ids = []
                        is_training = (dataset == TRAIN)
                        if dataset in [TRAIN, TRAIN_METRICS, VALIDATION]:
                            dataset_ids = [dataset_train.keys[i] for i in train_ids] if \
                                is_training or dataset == TRAIN_METRICS else [dataset_train.keys[i] for i in validation_ids]
                        dataset_parsed = ImageDataset(
                            paths[CLINICAL_DATA_PATH],
                            paths[SCANS_PATH],
                            get_training_augmentation(augment=CONFIGURATIONS[DATA_AUGMENTATION]) if is_training else TRANSFORM_TO_TENSOR,
                            ids_to_use=dataset_ids,
                            timeframe=CONFIGURATIONS[TIME_TO_EVENT],
                            event=CONFIGURATIONS[EVENT],
                        )
                        dataloaders[dataset] = DataLoader(
                            dataset_parsed,
                            batch_size = CONFIGURATIONS[BATCH_SIZE] if is_training and is_neural_network else len(dataset_parsed.images),
                            shuffle = is_training,
                            drop_last = is_training,
                        )

                    model = MODELS[CONFIGURATIONS[MODEL]](
                        clinical_data=CONFIGURATIONS.get(CLINICAL_VARIABLES),
                    )
                    print(model)
                    # Print a summary of the model
                    # summary(NaturalSceneClassification(), (1, 180, 180))
                    history = fit(
                        model,
                        dataloaders,
                        parameters=CONFIGURATIONS[HYPERPARAMETERS],
                        store_model=CONFIGURATIONS[STORE_MODEL],
                    )
