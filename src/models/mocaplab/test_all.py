import sys
import os
from torch.utils.data import DataLoader
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
src_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..\..\..'))
sys.path.append(src_folder)
from src.setup import setup_python, setup_pytorch
from src.dataset import MocaplabDatatestsetCNN,MocaplabDatasetCNN
from cnn.cnn import TestCNN
from train import test_cnn
if __name__ == "__main__":

    print("#### Set-Up ####")

    # Set-up Python
    setup_python()

    # Set-up PyTorch
    DEVICE = setup_pytorch(gpu=False)

    print("#### Dataset ####")
    
    bones_to_keep = list(set("CC_Base_Waist;CC_Base_Waist;CC_Base_Waist;CC_Base_Waist;CC_Base_Waist;CC_Base_Waist;CC_Base_Spine01;CC_Base_Spine01;CC_Base_Spine01;CC_Base_Spine01;CC_Base_Spine01;CC_Base_Spine01;CC_Base_Spine02;CC_Base_Spine02;CC_Base_Spine02;CC_Base_Spine02;CC_Base_Spine02;CC_Base_Spine02;CC_Base_R_Clavicle;CC_Base_R_Clavicle;CC_Base_R_Clavicle;CC_Base_R_Clavicle;CC_Base_R_Clavicle;CC_Base_R_Clavicle;CC_Base_R_Upperarm;CC_Base_R_Upperarm;CC_Base_R_Upperarm;CC_Base_R_Upperarm;CC_Base_R_Upperarm;CC_Base_R_Upperarm;CC_Base_R_UpperarmTwist02;CC_Base_R_UpperarmTwist02;CC_Base_R_UpperarmTwist02;CC_Base_R_UpperarmTwist02;CC_Base_R_UpperarmTwist02;CC_Base_R_UpperarmTwist02;CC_Base_L_Clavicle;CC_Base_L_Clavicle;CC_Base_L_Clavicle;CC_Base_L_Clavicle;CC_Base_L_Clavicle;CC_Base_L_Clavicle;CC_Base_L_Upperarm;CC_Base_L_Upperarm;CC_Base_L_Upperarm;CC_Base_L_Upperarm;CC_Base_L_Upperarm;CC_Base_L_Upperarm;CC_Base_L_UpperarmTwist02;CC_Base_L_UpperarmTwist02;CC_Base_L_UpperarmTwist02;CC_Base_L_UpperarmTwist02;CC_Base_L_UpperarmTwist02;CC_Base_L_UpperarmTwist02;C_Base_NeckTwist01;CC_Base_NeckTwist01;CC_Base_NeckTwist01;CC_Base_NeckTwist01;CC_Base_NeckTwist01;CC_Base_NeckTwist01;CC_Base_NeckTwist02;CC_Base_NeckTwist02;CC_Base_NeckTwist02;CC_Base_NeckTwist02;CC_Base_NeckTwist02;CC_Base_NeckTwist02;CC_Base_Head;CC_Base_Head;CC_Base_Head;CC_Base_Head;CC_Base_Head;CC_Base_Head;lowerarm_l;lowerarm_l;lowerarm_l;lowerarm_l;lowerarm_l;lowerarm_l;lowerarm_twist_01_l;lowerarm_twist_01_l;lowerarm_twist_01_l;lowerarm_twist_01_l;lowerarm_twist_01_l;lowerarm_twist_01_l;hand_l;hand_l;hand_l;hand_l;hand_l;hand_l;lowerarm_r;lowerarm_r;lowerarm_r;lowerarm_r;lowerarm_r;lowerarm_r;lowerarm_twist_01_r;lowerarm_twist_01_r;lowerarm_twist_01_r;lowerarm_twist_01_r;lowerarm_twist_01_r;lowerarm_twist_01_r;hand_r;hand_r;hand_r;hand_r;hand_r;hand_r".split(';')))
    dataset_cnn_labelled = MocaplabDatasetCNN(path=(f"{src_folder}/data/mocaplab/Autoannotation"),
                                padding=True, 
                                train_test_ratio=8,
                                validation_percentage=0.01, bones_to_keep=bones_to_keep)
    dataset_cnn = MocaplabDatatestsetCNN(path=(f"{src_folder}/data/mocaplab/Autoannotation"),
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
    test_data_loader = DataLoader(dataset_cnn_labelled,
                             batch_size=1,
                             shuffle=False)
    cnn = TestCNN()

    # Load the trained weights cnn old model
    cnn.load_state_dict(torch.load((f"{src_folder}/src/models/mocaplab/all/saved_models/CNN/CNN_Mono_Bi_c3dbody.ckpt"),
                                    map_location=torch.device("cpu")))
    # set the evaluation mode
    test_acc, test_confusion_matrix, misclassified = test_cnn(cnn, test_data_loader, DEVICE)
    sns.heatmap(test_confusion_matrix, annot=True, cmap="flare",  fmt="d", cbar=True)
    plt.savefig(f"{src_folder}/train_results/mocaplab/CNN_Mono_Bi_c3dbody.png")

    # Load the trained weights cnn new model
    #cnn.load_state_dict(torch.load(("/home/self_supervised_learning_gr/self_supervised_learning/dev/ProjetCassiopee/src/models/mocaplab/cnn/saved_models/CNN_20240514_211739.ckpt"),
                                    # map_location=torch.device("cpu")))

    

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
