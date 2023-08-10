#  Convolutional Neural Network (CNN) for head and neck outcome classification

Image based prognosis in head and neck cancer using convolutional neural networks: a case study in reproducibility and optimization.

## Description

In this work, we evaluated the reproducility of proposed Convolutional Neural Networks for head and neck cancer prognosis.
As a result, we developed a less complex network based on previous work, trained and evaluated the performance for outcome prediction (distant metastasis, loco-regional failure and overall survival), and evaluated the impact of pre-processing the CT scans and the model selection method.
In this repository, you'll find the necessary tools to train/validate/test the proposed model. Additionally, it also includes the necessary tools to reproduce the work by using the same seeds (./seed.xlsx).

## Requirements

To train/validate/test the model you can use docker (the necessary docker images are available) and docker-compose. This avoids the need to configure a local environment and guarantees an equal environment to the one used while developing the network.

Docker images:
- FSL official image:
- Custom docker image for pre-processing and training the network:

It's also possible to configure a local environment without using docker, directly install the necessary dependencies using the `requirements.txt` file.

## Data pre-processing

The pre-processing pipeline developed consisted of:
1. Converting the imaging data (DICOM CT scans and DICOM RT STRUCT) to NIFTIs (dcmrtstruct2nii)
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

## Citation

If you find this code useful for your research, please cite:

