#  Convolutional Neural Network (CNN) for head and neck outcome classification

Image based prognosis in head and neck cancer using convolutional neural networks: a case study in reproducibility and optimization.

- [Convolutional Neural Network (CNN) for head and neck outcome classification](#convolutional-neural-network-cnn-for-head-and-neck-outcome-classification)
  - [Description](#description)
  - [Requirements](#requirements)
  - [Data pre-processing](#data-pre-processing)
  - [Running the model](#running-the-model)
    - [Data split](#data-split)
    - [Training](#training)
  - [Citation](#citation)

## Description

In this work, we evaluated the reproducility of proposed Convolutional Neural Networks for head and neck cancer prognosis.
As a result, we developed a less complex network based on previous work, trained and evaluated the performance for outcome prediction (distant metastasis, loco-regional failure and overall survival), and evaluated the impact of pre-processing the CT scans and the model selection method.
In this repository, you'll find the necessary tools to train/validate/test the proposed model. Additionally, it also includes the necessary tools to reproduce the work by using the same seeds (./seed.xlsx).

## Requirements

To train/validate/test the model you can use docker (the necessary docker images are available) and docker-compose. This avoids the need to configure a local environment and guarantees an equal environment to the one used while developing the network.

Docker images:
- FSL official image: `vistalab/fsl-v5.0`
- Custom docker image for pre-processing and training the network: `pmateus/hn-cnn:1.0.0`

It's also possible to configure a local environment without using docker, directly install the necessary dependencies using the `requirements.txt` file.

## Data pre-processing

The pre-processing pipeline developed consisted of:
1. Converting the imaging data (DICOM CT scans and DICOM RT STRUCT) to NIFTIs (using `dcmrtstruct2nii` library)
2. Apply the GTV mask to the NIFTI (FSL)
3. Cropping around the tumor region
4. Smoothing and windowing of the Hounsfield units (HU)
5. Transformation to a png format with 256 values (the input for the network)

## Running the model

The script './training.py' provides the base to train and evaluate the model under different configurations.
In './example' you can find a small subset of images to test the network training:
- In the terminal: `docker-compose run hn-cnn`
- Once in the container's terminal: `cd /mnt`
- Run the training script: `python3 training.py`

### Data split

In this work, we followed the same data split as previous studies. In this strategy, two cohorts are used for training, two for validation, and one exclusively for external validation. To train the model following this method, configure the `DATA_SPLIT` parameter with `COHORT_SPLIT`.

![Data split based on the cohorts available](/censored_data.png)

Addittionally, we evaluated the uncertainity of the model with a 5-fold cross validation strategy. To train the model following this method, configure the `DATA_SPLIT` parameter with `CROSS_VALIDATION`.

### Training

The current implementation allows to train and evaluate 3 different models:
- A convolutional neural network (set the `Model` parameter in the configuration to `CNN`)
- An artificial neural network (set the `Model` parameter in the configuration to `ANN`)
- A logistic regression (set the `Model` parameter in the configuration to `LR`)

These models can be evaluated by splitting the data following the `COHORT_SPLIT` or cross validation (check previous section).

The additional parameters available are described below:
- `FOLDS`: Number of folds to use when performing cross validation
- `TIME_TO_EVENT`: The minimum observation period for a non-event to be included in the training
- `EVENT`: Event that the network will predict (`DM` - Distant Metastasis, `LRF` - Local-Regional Failure, `OS` - Survival)
- `HYPERPARAMETERS`: Set of hyperparameters to change from the default ones (check below)
- `BATCH_SIZE`: the batch size
- `LOGS_PATH`: Path to store the logs and metrics
- `CLINICAL_VARIABLES`: State the clinical variables that will be included when training the model
- `DATA_AUGMENTATION`: Data augmentation techniques to apply

Regarding the hyperparameters employed:
- `LEARNING_RATE`: the learning rate (default: 0.05)
- `EPOCHS`: number of epochs
- `MOMENTUM`: momentum
- `DAMPENING`: dampening (default: 0.00)
- `RELU_SLOPE`: slope for the RELU function (default: 0.10)
- `WEIGHTS_DECAY`: weight decay (L2 penalty) (default: 0.0001)
- `OPTIMIZER`: (optimizer used, default: SGD - https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
- `CLASS_WEIGHTS`: weights for each class in the loss function (default: [0.7, 3.7])

Regarding the data augmentation techniques:
- `HORIZONTAL_FLIP`: by default a probability of 0.5
- `VERTICAL_FLIP`: by default a probability of 0.5
- `ROTATE_90`: randomly rotate the image 90 degrees 1-3 times, by default a probability of 0.75
- `ROTATION`: randomly rotate the image a certain number of degrees, by default maximum 10 degrees

The training scripts allow to set up the necessary seeds in order to make the results fully reproducible:
- the random seed for python (`random.seed()`)
- the random seed for the data split, only necessary when performing cross-validation (`StratifiedKFold`)
- the random seed for the pytorch library (`torch.manual_seed()`)

## Citation

If you find this code useful for your research, please cite:
[10.1038/s41598-023-45486-5](https://doi.org/10.1038/s41598-023-45486-5)
