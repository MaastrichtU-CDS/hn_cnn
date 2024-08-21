#  Convolutional Neural Network (CNN) for head and neck outcome classification

Image based prognosis in head and neck cancer using convolutional neural networks: a case study in reproducibility and optimization.

- [Convolutional Neural Network (CNN) for head and neck outcome classification](#convolutional-neural-network-cnn-for-head-and-neck-outcome-classification)
  - [Description](#description)
  - [Requirements](#requirements)
  - [Notes](#notes)
  - [Data pre-processing](#data-pre-processing)
  - [Running the model](#running-the-model)
    - [Example](#example)
    - [Data split](#data-split)
    - [Training](#training)
    - [Clinical Data](#clinical-data)
  - [Reproducibility](#reproducibility)
  - [Results](#results)
  - [Citation](#citation)

## Description

In this work, we evaluated the reproducility of proposed Convolutional Neural Networks for head and neck cancer prognosis.
As a result, we developed a less complex network based on previous work, trained and evaluated the performance for outcome prediction (distant metastasis, loco-regional failure and overall survival), and evaluated the impact of pre-processing the CT scans and the model selection method.
In this repository, you'll find the necessary tools to train/validate/test the proposed model. Additionally, it also includes the necessary tools to reproduce the work by using the same seeds (./seed.xlsx).

## Requirements

To train/validate/test the model you can use docker (the necessary docker images are available) and docker-compose. This avoids the need to configure a local environment and guarantees an equal environment to the one used while developing the network.

Docker images:
- FSL official image: `vistalab/fsl-v5.0`
- Custom docker image for pre-processing and training the network: `pmateus/hn-cnn:1.5.0`

It's also possible to configure a local environment without using docker, directly install the necessary dependencies using the `requirements.txt` file.

## Notes

Due to vulnerabilities in the dependencies, we updated the versions starting with release `pmateus/hn-cnn:1.5.0`.
The information on the original versions and code can be found in the release for version `1.4.0` (and on the branch `reproduce`).

## Data pre-processing

The pre-processing pipeline developed consisted of:
1. Converting the imaging data (DICOM CT scans and DICOM RT STRUCT) to NIFTIs (using `dcmrtstruct2nii` library)
2. Reorient the scans and apply the GTV mask to the NIFTI (FSL)
3. Slice selection* and cropping around the tumor region
4. Smoothing and windowing of the Hounsfield units (HU)
5. Transformation to a png format with 256 values (the input for the network)

For step 1 and 2 we provide the script `image_preprocesing/convert_dicom.py`: it takes as input a csv file identifying the CT scans for each patient, transforms the scans to NIFTI, and prints the FSL commands that allow to subtact the mask to scan. After running this script, configure the `image_preprocesing/docker-compose.yaml` file with the correct data folders and run the FSL commands stored in the `fsl_script.sh` script.

For step 3, 4, and 5, we provide the script `image_preprocesing/windowing_cropping.py`: the input should be the results from the FSL pre-processing (the masked CT scans `{scan_id}_im.nii.gz` and the masks `{scan_id}_mask.nii.gz`). This script will output the png images that can be used to train or test the CNN.

<sub>*we used the masked CT scan (obtained in step 2), applied the HU window (-50 to 300), and selected the slice with largest area (based on the number of non-zero values)<sub>

## Running the model

The script './training.py' provides the base to train and evaluate the model under different configurations.
To run the model, make sure to:
- (Using Docker) Set the path to the data folder in the docker-compose file:
```yaml
    volumes:
      - ./path/to/data:/mnt/data
```
- Set the path for the training, validation, and testing sets in the './training.py' script (check the example script under './example'):
```python
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
```

### Example

In './example' you can find a small subset of images to test the network training:
- In the terminal: `docker-compose run hn-cnn`
- Once in the container's terminal: `cd /mnt`
- Run the training script: `python3 training.py`

### Data split

In this work, we followed the same data split as previous studies. In this strategy, two cohorts are used for training, two for validation, and one exclusively for external validation. To train the model following this method, configure the `DATA_SPLIT` parameter with `COHORT_SPLIT`.
The cohort for training and validation are part of the [Head-Neck-PET-CT dataset](https://doi.org/10.7937/K9/TCIA.2017.8oje5q00).
The dataset [HEAD-NECK-RADIOMICS-HN1](https://doi.org/10.7937/tcia.2019.8kap372n) from Maastro was used exclusively for external validation.

![Data split based on the cohorts available](/censored_data.png)

Additionally, we evaluated the uncertainity of the model with a 5-fold cross validation strategy. To train the model following this method, configure the `DATA_SPLIT` parameter with `CROSS_VALIDATION`.

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

To store the model, include the following parameters:
- `MODEL_ID`: will be used for the file name in combination with the epoch number
- `MODEL_PATH`: the path for the folder to store the models
- `THRESHOLD`: (optional) Stores the model if the AUC for the validation set is above this value
- `MAX_DIFFERENCE`: (optional) Stores the model if the difference between the training and validation AUC is below this value

### Clinical Data

The clinical variables currently considered can be checked in the file `parse_data.py`.
The dictionary `CLINICAL_DATA` identifies the necessary variables and the values used across the different datasets.
This information is used to harmonize the datasets before training the network.

The clinical information considered: primary site, T-stage, N-stage, TNM-stage, HPV (human papillomavirus) status, 
volume, and area of the tumour.

Currently, the clinical variables included in the model (when setting `CLINICAL_VARIABLES` to `True`) are pre-set in the
file `parse_data.py`:
```python
  features = tabular[["n0", "n1", "n2", "t1", "t2", "t3", "t4", "vol0", "vol1", "vol2", "vol3"]]
``` 
If you change this set, make sure to also modify the number of neurons in the network accordingly (file `cnn.py`).

## Reproducibility

The training scripts allow to set up the necessary seeds in order to make the results fully reproducible:
- the random seed for python (`random.seed()`)
- the random seed for the data split, only necessary when performing cross-validation (`StratifiedKFold`)
- the random seed for the pytorch library (`torch.manual_seed()`)

When reproducing the results from the manuscript (using the seeds provided in this repository), 
please include the following in the training script (as exemplified in `/data/models/training_example_dm.py`):

```python
random.seed()
# Include the next line:
random_seed_split = random.randint(0, 9174937)
torch.manual_seed()
```

Finally, the scripts provided in the folder `/data/models` already include all the necessary configurations 
to reproduce our results for the prediction of each outcome.

We've trained the network in [DSRI](dsri.maastrichtuniversity.nl), an openshift cluster of servers. Although 
we provide the seeds and scripts to reproduce the results, inconsistencies may occur in certain machines. 
We observed that some systems differ when executing the `torch.nn.Dropout` function (using the same seeds).
From experiments with different machines, we think this is caused by different CPU architecture. To obtain 
the same results that we provide, you should use a CPU with an ARM architecture ( tested in AWS with 
an Ubuntu 24.04 LTS image and a t4g.micro 64-bit ARM CPU).
```python
import random
import torch
random.seed(7651962)
random_seed_split = random.randint(0, 9174937)
torch.manual_seed(775135)
# Gives the same result:
data = torch.randn(4, 4)
# Gives different results:
dp1 = torch.nn.Dropout(p=0.3)
dp1(data)
# Original experiments in DSRI (also works in AWS with an 
# ARM architecture)
# tensor([[-1.0361, -3.5810, -1.1379,  1.8477],
#         [-0.0000, -1.2687, -0.0000,  1.7217],
#         [-1.4952, -2.3690, -0.0955, -0.0000],
#         [-0.4844,  1.5011,  1.8367,  2.3154]])
# vs
# x86 architecture
# tensor([[-1.0361, -0.0000, -0.0000,  1.8477],
#         [-2.0117, -1.2687, -0.0785,  0.0000],
#         [-0.0000, -2.3690, -0.0955, -0.0000],
#         [-0.4844,  1.5011,  1.8367,  2.3154]])
```

Setting up the following configurations did not change the behavior. The `Dropout` function still provided different 
results:
```python
device = torch.device("cpu")
torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)
```

Using our own implementation of `Dropout` can be an option to avoid this problem in the future:
```python
x = torch.ones(10, 20)
p = 0.5
mask = torch.distributions.Bernoulli(probs=(1-p)).sample(x.size())
x[~mask.bool()] = x.mean()
out = x * mask * 1/(1-p)
```
## Results

Performance results using only imaging data:

| Outcome | Validation AUC | Test AUC | Epoch |  Model |
|----------|:--------:|-------:|-----:|---------------------:|
| DM |  0.89 | 0.89 | 689 | ./data/models/model_dm_689.pth.tar |
| LRF |    0.77   | 0.77 | 2340 |   ./data/models/model_lrf_2340.pth.tar |
| OS | 0.80 | 0.67 | 1341 |    ./data/models/model_os_1341.pth.tar |

Performance results using imaging and clinical data:

| Outcome  | Validation AUC | Test AUC | Epoch | Model |
|----------|:--------:|--------:|-----:|---------------------:|
| DM |  0.89 | 0.93 | 646 | ./data/models/with_clinical_data/model_dm_cd_646.pth.tar |
| LRF |    0.70   | 0.59 | 964 |   ./data/models/with_clinical_data/model_lrf_cd_964.pth.tar |
| OS | 0.74 | 0.69 | 2445 |  ./data/models/with_clinical_data/model_os_cd_2445.pth.tar |

Training set: [HGJ and CHUS](https://doi.org/10.7937/K9/TCIA.2017.8oje5q00)
Validation set: [HMR and CHUS](https://doi.org/10.7937/K9/TCIA.2017.8oje5q00)
Testing set: [Maastro](https://doi.org/10.7937/tcia.2019.8kap372n)

For more information check the [publication](https://doi.org/10.1038/s41598-023-45486-5).

## Citation

If you find this code useful for your research, please cite:
[10.1038/s41598-023-45486-5](https://doi.org/10.1038/s41598-023-45486-5)
