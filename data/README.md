## Data

In this section, you can find the pre-processed data used to train the proposed CNN model.
In addition, the scripts that allow to reproduce our work are also provided.
We conducted the experiments in a distributed computing platform using the docker image developed.

### Content

- `canada.csv` and `maastro.csv`: contain the clinical data necessary for training the network
- `training_example_{}.py`: contains the code to train the network (includes the seed used for the study) - dm (Distant metastasis); lrf (Loco-regional failure); os - Overall survival
- `pre-processed`: the imaging input to the network (already pre-processed according to the pipeline described in the Methods)
- `seeds.xlsx`: the seeds used in each experiment

### Running the model

1. Create a container using the docker image `pmateus/hn-cnn:x.y.z` (check the most recent version in the releases)
2. If not present, create a folder `logs` (store the log files)  and `backup` (store the models)
3. Run the script to train the network
