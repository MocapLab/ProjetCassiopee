import csv
import numpy as np
import os
import pandas as pd

def read_csv(csv_file, bones_to_keep=None):
    data = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file, delimiter=';')
        n = 0
        for line in csv_reader:
            if n == 0:
                header_1 = line
                out_header = list(set(header_1))
            if n == 1:
                header_2 = line
                if bones_to_keep is None:
                    bones_to_keep = out_header
                id = {}
                for i, bone in enumerate(header_1):
                    bone_type = header_2[i]
                    if bone not in id.keys() and bone.removesuffix('_glob') in bones_to_keep and ((bone_type.startswith('T') and bone_type.endswith('_glob')) or (bone_type.startswith('R') and not bone_type.endswith('_glob'))):
                        id[bone] = [i]
                    elif bone in id.keys() and bone in bones_to_keep and ((bone_type.startswith('T') and bone_type.endswith('_glob')) or (bone_type.startswith('R') and not bone_type.endswith('_glob'))):
                        id[bone].extend([i])
            if n > 2:
                values = []
                if id:
                    for bone in id.keys():
                        for i in id[bone]:
                            values.append(float(line[i]))
                    data.append(values)
            n+=1
        data = np.stack(data)
    return data, out_header, bones_to_keep

def load_data(path, class_dict, max_length=0, x=None, y=None, removed=None, col_num=3):
    labels = pd.read_csv(os.path.join(path,
                                          "Annotation_gloses.csv"), sep="\t")
    labels.dropna(inplace=True)
    out_labels = {n: c for n, c in zip(labels.iloc[:,0], labels.iloc[:,col_num]) if os.path.exists(os.path.join(path,f"{n}.csv"))}
    if x is None:
        x = []
    if y is None:
        y = []
    if removed is None:
        removed = []
    # Retrieve files
    files = [i.lower() for i in os.listdir(path)]
    for name, label in out_labels.items():
        filename = name + ".csv"
        if filename.lower() not in files:
            removed.append(filename)
        else:
            x.append(filename)
            y.append(class_dict[label])

            # Retrieve max length
            data, out_header, bones_to_keep = read_csv(os.path.join(path, filename))
            length = len(data)
            if length > max_length:
                max_length = length
    return out_labels, max_length, x, y, removed, out_header, bones_to_keep