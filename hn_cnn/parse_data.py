import os

import pandas as pd
from PIL import Image
import torch

from hn_cnn.constants import *
from hn_cnn.image_preprocessing import process_scan

# Mapping for the clinical data
# Include each variable: Primary Site, T-stage, N-stage, ...
#    Key: the name of the column containing the variable in the dataset
#    Values: mapping between each output value and the codes used in the dataset
#    Thresholds: specify the thresholds when using a numeric value that needs to be discretized
CLINICAL_DATA = {
    PRIMARY_SITE: {
        KEY: "site",
        VALUES: {
            "h": ["HYPOPHARYNX"],
            "l": ["LARYNX"],
            "n": ["NASOPHARYNX"],
            "o": ["OROPHARYNX"],
            "u": ["UNKNOWN"],
        }
    },
    T_STAGE: {
        KEY: "tstage",
        VALUES: {
           "t1": ['T1', '1'],
           "t2": ['T2', '2'],
           "t3": ['T3', '3', 'T3 (2)'],
           "t4": ['T4', 'T4A', 'T4B', '4'], 
           "tx": ["TX"],
        }
    },
    N_STAGE: {
        KEY: "nstage",
        VALUES: {
            "n0": ['N0', '0'],
            "n1": ['N1', '1'],
            # Only 1 patient with N3 stage
            "n2": ['N2', 'N2A', 'N2B', 'N2C', '2', 'N3', 'N3B', '3'],
            "n3": ['N3', 'N3B', '3'],
        }
    },
    TNM_STAGE: {
        KEY: "groupstage",
        VALUES: {
            "s1": ['STADE I', 'STAGE I', 'I'],
            "s2": ['STADE II', 'STAGE II', 'STAGEII', 'STAGE IIB', 'II', 'IIB'],
            "s2b": ['STAGE IIB', 'IIB'],
            "s3": ['STADE III', 'STAGE III', 'III'],
            "s4": ['STAGE IV', 'STADE IV', 'STAGE IVA', 'STAGE IVB', 'STADE IVA', 
                'STADE IVB','IV', 'IVA', 'IVB', 'IVC'],
            "s4o": ['STAGE IV', 'STADE IV', 'IV'],
            "s4a": ['STAGE IVA', 'STADE IVA', 'IVA'],
            "s4b": ['STAGE IVB', 'STADE IVB', 'IVB'],
            "s4c": ['IVC'],
            "nan": ["N/A", "NAN"],
        }
    },
    HPC: {
        KEY: "overall_hpv_status",
        VALUES: {
            "hpc+": ['+', 'POSITIVE', 'POSITIVE -STRONG', 'POSITIVE -FOCAL'],
            "hpc-": ['-', 'NEGATIVE'],
            "nan": ['N/A', 'NAN', 'NOT TESTED'],
        }
    },
    VOLUME: {
        # To calculate the volume we used FSL:
        # /usr/share/fsl/5.0/bin/fslstats /path/to/masks/{mask_id}_mask.nii.gz -V
        KEY: "vol",
        THRESHOLDS: {
            "vol0": [0, 11000],
            "vol1": [11000, 24000],
            "vol2": [24000, 43000],
            "vol3": [43000, -1]
        }
    },
    AREA: {
        # To calculate the area:
        # cord = list(scan.header.get_zooms())[0:1]
        # area.append(pixels_by_slice[np.argmax(pixels_by_slice)] * cord[0] * cord[1])
        KEY: "area",
        THRESHOLDS: {
            "area0": [0, 470],
            "area1": [470, 770],
            "area2": [770, 1100],
            "area3": [1100, -1]
        }
    }
}

EVENTS = {
    DM: {
        EVENT: ["Distant", "event_distant_metastases"],
        TIME_TO_EVENT: ["Time - diagnosis to DM (days)", "distant_metastases_in_days"],
        FU: ["Time - diagnosis to last follow-up(days)", "distant_metastases_in_days"],
    },
    LRF: {
        EVENT: ["Locoregional", "event_locoregional_recurrence"],
        TIME_TO_EVENT: ["Time - diagnosis to LR (days)", "locoregional_recurrence_in_days"],
        FU: ["Time - diagnosis to last follow-up(days)", "locoregional_recurrence_in_days"],
    },
    OS: {
        EVENT: ["Death", "event_overall_survival"],
        TIME_TO_EVENT: ["Time - diagnosis to Death (days)", "overall_survival_in_days"],
        FU: ["Time - diagnosis to last follow-up(days)", "overall_survival_in_days"],
    },
}

# Primary site
# ['HYPOPHARYNX', 'LARYNX', 'NASOPHARYNX', 'OROPHARYNX', 'UNKNOWN']
# T-stage
# ['T1', 'T2', 'T3', 'T4', 'T4A', 'T4B', 'TX']
# N-stage
# ['N0', 'N1', 'N2', 'N2B', 'N2C', 'N3', 'N3B']
# TNM-stage
# ['STAGE I', 'STAGE II', 'STAGE IIB', 'STAGE III', 'STAGE IV', 'STAGE IVA', 'STAGE IVB']
# HPV Status
# ['HPV+', 'HPV-', 'UNKNOWN']
def parse_clinical(tabular):
    """ Parse the clinical information to the same coding.
    """
    for variable, variable_info in CLINICAL_DATA.items():
        column_name = variable_info[KEY]
        if VALUES in variable_info:
            column_value = str(tabular[column_name]).upper().strip()
            variable_values = []
            for site_key, site_values in variable_info[VALUES].items():
                variable_values.extend(site_values)
                tabular[site_key] = int(column_value in site_values)
            if column_value not in variable_values:
                print(f'Error in parsing the {variable}, unknow value: {tabular[column_name]}')
        elif THRESHOLDS in variable_info:
            # Variables that need to be discretized
            column_value = float(tabular[column_name])
            for site_key, site_values in variable_info[THRESHOLDS].items():
                tabular[site_key] = int(
                    column_value >= site_values[0] and (site_values[1] < 0 or column_value < site_values[1])
                )
    features = tabular[["n0", "n1", "n2", "t1", "t2", "t3", "t4", "vol0", "vol1", "vol2", "vol3"]]
    return torch.FloatTensor(features.tolist())

def parse_event(tabular, event):
    """ Parse the event/outcome information and related data
    """
    parsed_event = {}
    for variable, variable_info in EVENTS[event].items():
        for column in variable_info:
            if column in tabular:
                parsed_event[variable] = tabular[column]
                break
    # TODO: Create error when one of the parameters doesn't have a result
    return parsed_event

class ImageDataset(torch.utils.data.Dataset):
    """Class to read, transform, and provide the data to the CNN"""

    def __init__(
            self,
            dataset_path,
            scans_path,
            transforms,
            ids_to_use=[],
            timeframe=2*365,
            event="dm",
            preprocess=False,
            mask_path=None,
            mask_suffix=None,
        ):
        # Read the dataset
        self.tabular = pd.read_csv(dataset_path, delimiter=';')
        # Extract the information
        self.transforms = transforms
        self.images = {}
        self.tab_data = {}
        self.keys = []
        self.y = []
        for index, tb in self.tabular.iterrows():
            event_info = parse_event(tb, event)
            id = tb[ID]
            if (len(ids_to_use) == 0 or id in ids_to_use) and \
                (event_info[FU] >= timeframe or event_info[EVENT] == 1):
                    if preprocess and mask_path:
                        self.images[id] = process_scan(
                            scan_path=f"{scans_path}/{id}.nii.gz",
                            mask_path=f"{mask_path}/{id}{mask_suffix}.nii.gz",
                        )
                    else:
                        self.images[id] = Image.open(f"{scans_path}/{id}.png")
                    self.tab_data[id] = parse_clinical(tb)
                    self.keys.append(id)
                    self.y.append(
                        1 if event_info[EVENT] == 1 and event_info[TIME_TO_EVENT] <= timeframe else 0
                    )
        print(f"Imported {len(self.images)} scans from {scans_path} with {sum(self.y)} events")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        id = self.keys[idx]
        image = self.transforms(self.images[id])
        return image, self.tab_data[id], int(self.y[idx])
