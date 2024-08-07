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
from hn_cnn.data_augmentation import get_training_augmentation, TRANSFORM_TO_TENSOR
from hn_cnn.fit import fit
from hn_cnn.parse_data import ImageDataset

# Specify the path to the training, validation, and testing dataset file and imaging folder.
DATA = {
    TRAIN: {
        CLINICAL_DATA_PATH: "",
        SCANS_PATH: "",
    },
    VALIDATION: {
        CLINICAL_DATA_PATH: "",
        SCANS_PATH: "",
    },
    TESTING: {
        CLINICAL_DATA_PATH: "",
        SCANS_PATH: "",   
    }
}

CONFIGURATIONS = {
    # The Model can be: CNN, ANN, LR
    MODEL: CNN,
    # How to split the data (COHORT_SPLIT or CROSS_VALIDATION)
    DATA_SPLIT: COHORT_SPLIT,
    # The number of folds in case of using CROSS_VALIDATION
    FOLDS: 5,
    # The period considered for the event to occur in days
    TIME_TO_EVENT: 2 * 365,
    # The Event can be: DM, LRF, OS
    EVENT: DM,
    HYPERPARAMETERS: {
        # Include any hypeparameters that you want to change
        # from the default ones:
        LEARNING_RATE: 0.05,
    },
    BATCH_SIZE: 64,
    LOGS_PATH: "/path/to/save/logs",
    # Leave empty to train the network without clinical data
    CLINICAL_VARIABLES: [],
    DATA_AUGMENTATION: {},
    # Include the following section if you want to store the model
    # STORE_MODEL: {
    #     # The filename will consist of model_id + epoch
    #     MODEL_ID: "model1",
    #     # Path for the folder to store the models
    #     MODEL_PATH: '/mnt/models',
    #     # Minimum AUC for the validation set
    #     THRESHOLD: 0.8,
    #     # Maximum difference between training and validation AUC
    #     MAX_DIFFERENCE: 0.1,
    # }
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
    file_path = CONFIGURATIONS[LOGS_PATH] + f"log_{random_seed}"
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
                    batch_size = CONFIGURATIONS[BATCH_SIZE] if is_training and is_neural_network else len(dataset_parsed.keys),
                    shuffle = is_training,
                    drop_last = is_training,
                )
            # Print a summary of the model
            # summary(NaturalSceneClassification(), (1, 180, 180))
            model = MODELS[CONFIGURATIONS[MODEL]]()
            print(model)
            history = fit(
                model,
                dataloaders,
                parameters=CONFIGURATIONS[HYPERPARAMETERS],
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
            file_path = CONFIGURATIONS[LOGS_PATH] + f"log_{random_seed}_{random_seed_split}_fold_{fold}"
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
                    # Print a summary of the model
                    # summary(NaturalSceneClassification(), (1, 180, 180))
                    model = MODELS[CONFIGURATIONS[MODEL]]()
                    print(model)
                    history = fit(
                        model,
                        dataloaders,
                        parameters=CONFIGURATIONS[HYPERPARAMETERS],
                    )
