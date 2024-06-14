import os
import random
import pandas as pd
import csv
from torch.utils.data import Dataset
import numpy as np

class MocaplabDatasetFC(Dataset):
    """
    PyTorch dataset for the Mocaplab dataset.
    """

    def __init__(self, path, padding=True, train_test_ratio=8, validation_percentage=0.01, nb_samples=None, bones_to_keep=None):
        super().__init__()
        self.path = path
        print(f"TEST {self.path}")
        self.padding = padding
        self.train_test_ratio = train_test_ratio
        self.validation_percentage = validation_percentage
        self.bones_to_keep = bones_to_keep
        self.class_dict = None
        self.max_length = 0
        self.header = None
        self.x = []
        self.y = []
        self.labels = None
        self.removed = []

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
    
    def _read_csv(self, csv_file) :
        data = []
        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file, delimiter=';')
            n=0
            for line in csv_reader:
                if n==0:
                    header = line
                    self.header = list(set(header))
                    id = {}
                    for i, bone in enumerate(header):
                        if bone not in id.keys() and bone in self.bones_to_keep:
                            id[bone] = [i]
                        elif bone in id.keys() and bone in self.bones_to_keep:
                            id[bone].extend([i])
                if n>=2:
                    values = []
                    if id:
                        for bone in id.keys():
                            for i in id[bone]:
                                values.append(float(line[i]))
                        data.append(values)
                n+=1
        data = np.stack(data)
        if data.shape[1]!=len(self.bones_to_keep)*3:
            raise ValueError
        return data

    def __len__(self):
        return len(self.y)
    
    def _create_labels_dict(self):
        self.class_dict = {"Mono": 0, "Bi": 1}
        return self.class_dict
    
    def _load_data(self):
        # Retrieve labels
        labels = pd.read_csv(os.path.join(self.path, "Annotation_gloses.csv"),
                             sep="\t")
        labels.dropna(inplace=True)
        self.labels = {n: c for n, c in zip(labels["Nom.csv"], labels["Mono/Bi"]) if os.path.exists(os.path.join(self.path,f"{n}.csv"))}
        
        # Retrieve files
        files = os.listdir(self.path)
        for name, label in self.labels.items():
            filename = name + ".csv"
            if filename not in files:
                self.removed.append(filename)
            else:
                self.x.append(filename)
                self.y.append(self.class_dict[label])

                # Retrieve max length
                data = self._read_csv(os.path.join(self.path, filename))
                length = len(data)
                if length > self.max_length:
                    self.max_length = length

    def __getitem__(self, idx):
        data_path = os.path.join(self.path, self.x[idx])

        data = self._read_csv(data_path)
        label = self.y[idx]

        if self.padding:
            data = data.tolist()
            for _ in range(self.max_length-len(data)) :
                data.append([0.0 for _ in range(len(data[0]))])
            data = np.stack(data)
        
        return data, label, self.x[idx]