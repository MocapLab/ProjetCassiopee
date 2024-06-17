import sys
import os
from torch.utils.data import DataLoader
import torch
import numpy as np
src_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..\..\..'))
sys.path.append(src_folder)
from src.setup import setup_python, setup_pytorch
from src.dataset import MocaplabDatatestsetCNN
from cnn.cnn import TestCNN

if __name__ == "__main__":

    print("#### Set-Up ####")

    # Set-up Python
    setup_python()

    # Set-up PyTorch
    DEVICE = setup_pytorch(gpu=False)

    print("#### Dataset ####")
    
    bones_to_keep = list(set("UPHD;UPHD;UPHD;LFHD;LFHD;LFHD;RFHD;RFHD;RFHD;LBHD;LBHD;LBHD;RBHD;RBHD;RBHD;C7;C7;C7;T10;T10;T10;LBAC;LBAC;LBAC;RBAC;RBAC;RBAC;CLAV;CLAV;CLAV;STRN;STRN;STRN;LCLAV;LCLAV;LCLAV;RCLAV;RCLAV;RCLAV;LFSHO;LFSHO;LFSHO;LSHOULD;LSHOULD;LSHOULD;LBSHO;LBSHO;LBSHO;LUPA;LUPA;LUPA;LELB;LELB;LELB;LELBEXT;LELBEXT;LELBEXT;LFRM;LFRM;LFRM;LWRA;LWRA;LWRA;LWRB;LWRB;LWRB;RFSHO;RFSHO;RFSHO;RSHOULD;RSHOULD;RSHOULD;RBSHO;RBSHO;RBSHO;RUPA;RUPA;RUPA;RELB;RELB;RELB;RELBEXT;RELBEXT;RELBEXT;RFRM;RFRM;RFRM;RWRA;RWRA;RWRA;RWRB;RWRB;RWRB;LFWT;LFWT;LFWT;RFWT;RFWT;RFWT;LBWT;LBWT;LBWT;RBWT;RBWT;RBWT;LHIP;LHIP;LHIP;LUPLEG;LUPLEG;LUPLEG;LKNE;LKNE;LKNE;LPER;LPER;LPER;LTIB;LTIB;LTIB;LANK;LANK;LANK;LHEE;LHEE;LHEE;LMT5;LMT5;LMT5;LTOE;LTOE;LTOE;LMT1;LMT1;LMT1;RHIP;RHIP;RHIP;RUPLEG;RUPLEG;RUPLEG;RKNE;RKNE;RKNE;RPER;RPER;RPER;RTIB;RTIB;RTIB;RANK;RANK;RANK;RHEE;RHEE;RHEE;RMT5;RMT5;RMT5;RTOE;RTOE;RTOE;RMT1;RMT1;RMT1".split(';')))
    
    dataset_cnn = MocaplabDatatestsetCNN(path=(f"{src_folder}/data/mocaplab/LSDICOS"),
                                padding=True, 
                                train_test_ratio=8,
                                validation_percentage=0.01, bones_to_keep=bones_to_keep)
    print("#### Data Loader ####")
    # data_loader_fc = DataLoader(dataset_fc,
    #                          batch_size=1,
    #                          shuffle=False)

    
    print(dataset_cnn.max_length)
    data_loader_cnn = DataLoader(dataset_cnn,
                             batch_size=1,
                             shuffle=False)
    cnn = TestCNN()

    # Load the trained weights cnn old model
    cnn.load_state_dict(torch.load((f"{src_folder}/src/models/mocaplab/all/saved_models/CNN/CNN_20240614_193233.ckpt"),
                                    map_location=torch.device("cpu")))

    # Load the trained weights cnn new model
    #cnn.load_state_dict(torch.load(("/home/self_supervised_learning_gr/self_supervised_learning/dev/ProjetCassiopee/src/models/mocaplab/cnn/saved_models/CNN_20240514_211739.ckpt"),
                                    # map_location=torch.device("cpu")))

    # set the evaluation mode
    cnn.eval()

    heatmap_data = []   # a table that contains 112 lists of 100 lists of size 10 (10 max joints for each frame for each data)

    for k, img in enumerate(data_loader_cnn):
        ###TO GET THE HEATMAP LIST OF 10 MOST IMPORTANT JOINTS###
        img, label, name = img
        #print(f"img {os.path.splitext(name[0])[0]}: {k:4} / {len(data_loader_cnn)} ")
        # get the most likely prediction of the model
        pred = cnn(img)
        if pred[0,1].detach().numpy()>0.98:
            label_pred = "Bi"
        elif pred[0,0].detach().numpy()>0.98:
            label_pred = "Mono"
        else:
            label_pred = "Unknown"
        val = f"{name[0].split('.csv')[0]}\t{label.detach().numpy()[0]}\t{label_pred}\t{np.array2string(pred[0,:].detach().numpy(), precision=3, floatmode='fixed', separator=',', suppress_small=True)[1:-1]}"
        with open(f"{src_folder}/test_results/mocaplab/results.csv", "a") as f:
            f.write("%s\n"%val)
        print(val)
