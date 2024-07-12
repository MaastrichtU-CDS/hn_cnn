import os
import random

import torch
from torch.utils.data.dataloader import DataLoader
# from torchsummary import summary

from hn_cnn.cnn import MODELS
from hn_cnn.constants import *
from hn_cnn.data_augmentation import get_training_augmentation, TRANSFORM_TO_TENSOR
from hn_cnn.fit import predict
from hn_cnn.parse_data import ImageDataset

# Specify the path to the training, validation, and testing dataset file and imaging folder.
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

# Configure the network parameters
CONFIGURATIONS = {
    # MODEL:
    # - LR (logistic regression)
    # - CNN (Convolutional Neural Network imaging data and (optional) clinical data)
    # - ANN (Neural Network for clinical data)
    MODEL: CNN,
    DATA_SPLIT: COHORT_SPLIT,
    FOLDS: 2,
    TIME_TO_EVENT: 2 * 365,
    EVENT: DM,
    BATCH_SIZE: 64,
    # Leave empty to train the network without clinical data
    CLINICAL_VARIABLES: [],
    DATA_AUGMENTATION: {},
    MODEL_PATH: '/mnt/models/model_model1_0.pth.tar',
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
    dataset_train = ImageDataset(
        DATA[TRAIN][CLINICAL_DATA_PATH],
        DATA[TRAIN][SCANS_PATH],
        TRANSFORM_TO_TENSOR,
        timeframe=CONFIGURATIONS[TIME_TO_EVENT],
        event=CONFIGURATIONS[EVENT],
    )
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
    model = MODELS[CONFIGURATIONS[MODEL]]()
    # Load the model
    model.load_state_dict(torch.load(CONFIGURATIONS[MODEL_PATH])['model_state_dict'])
    output = predict(
        model,
        dataloaders,
    )
    # The output includes the metrics and predictions for each dataset
    for set, result in output.items():
        print(f"Predictions for set {set}")
        print(result[PREDICTIONS])
