import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import sys
import os
src_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..\..\..'))

sys.path.append(src_folder)
from src.dataset import MocaplabDatasetFC
from torch.utils.data import DataLoader

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.setup import setup_python, setup_pytorch
from src.dataset import MocaplabDatasetFC
from src.models.mocaplab import MocaplabFC
from fc.plot_results import plot_results
from fc.train import *

# Create figure
#fig, axs = plt.subplots(1, 1, figsize=(16, 9))
dataset = MocaplabDatasetFC(path="%s/data/mocaplab/Cassiopée_Allbones"%src_folder,
                              padding = True, 
                              train_test_ratio = 8,
                              validation_percentage = 0.01)
data_loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=True)

# Get a sample
for i, (x, y) in enumerate(data_loader):
    x = x.squeeze().numpy()
    fig = plt.figure()
    plt.matshow(x)
    plt.savefig("%s/src/visualisation/array/array.png"%src_folder)
    plt.show()
    break

print('done')
