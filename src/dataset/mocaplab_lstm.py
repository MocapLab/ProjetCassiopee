import os
import random
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from .mcl_io import read_csv as mcl_read_csv
from .mcl_io import load_data as mcl_load_data

class MocaplabDatasetLSTM(Dataset):
    """
    PyTorch dataset for the Mocaplab dataset.
    """

    def __init__(self, path, padding=True, train_test_ratio=8, validation_percentage=0.01, nb_samples=None, bones_to_keep=None, center=None, col_num=6):
        super().__init__()
        self.path = path
        self.padding = padding
        self.train_test_ratio = train_test_ratio
        self.validation_percentage = validation_percentage
        self.class_dict = None
        self.max_length = 0
        self.bones_to_keep = bones_to_keep
        self.header = None
        self.x = []
        self.y = []
        self.data = None
        self.labels = None
        self.removed = []
        self.center = center
        self.col_num = col_num

        self._create_labels_dict()
        self._load_data()
        print(f"removed {self.removed}")
        print(f"number of 3D data : {len(self.header)}")
        if nb_samples is not None:
            # Shuffle data in order to have multiple classes
            x_and_y = list(zip(self.x, self.y))
            random.shuffle(x_and_y)
            x_and_y = x_and_y[:nb_samples]
            self.x = [x for x,y in x_and_y]
            self.y = [y for x,y in x_and_y]
    
    def read_csv(self, csv_file) :
        data, self.header, self.bones_to_keep = mcl_read_csv(csv_file, self.bones_to_keep, self.center)
        if data.shape[1]!=len(self.bones_to_keep)*6:
            ValueError(f"missing {set(self.bones_to_keep) - set(self.header)}, for {csv_file}")
        return data

    def __len__(self):
        return len(self.y)
    
    def _create_labels_dict(self):
        labels = pd.read_csv(os.path.join(self.path,
                                          "Annotation_gloses.csv"), sep="\t")
        unique_val = labels.iloc[:,self.col_num].dropna(inplace=False).unique()
        self.class_dict = {}
        for i, val in enumerate(unique_val[::-1]):
            self.class_dict[val] = i
        return self.class_dict
    
    def _load_data(self):
        self.labels, self.max_length, self.x, self.y, self.removed, self.header, self.bones_to_keep = mcl_load_data(self.path, self.class_dict, self.max_length, self.x, self.y, self.removed, bones_to_keep=self.bones_to_keep, col_num=self.col_num)

    def __getitem__(self, idx):
        label = self.y[idx]
        if self.data and self.data[idx] is not None:
            return self.data[idx], label, self.x[idx]
        data_path = os.path.join(self.path, self.x[idx])

        data = self.read_csv(data_path)

        if self.padding:
            data = data.tolist()
            for _ in range(self.max_length-len(data)) :
                data.append([0.0 for _ in range(len(data[0]))])
            data = np.stack(data)
        if not self.data:
            self.data = [None]*len(self.y)
        self.data[idx] = data

        return data, label, self.x[idx]
    
    def get_labels_weights(self):
        from collections import Counter
        return {i: j/len(self.y) for i,j in zip(Counter(self.y).keys(),Counter(self.y).values())}