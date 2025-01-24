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
from src.dataset import MocaplabDatatestsetCNN,MocaplabDatasetCNN,MocaplabDatasetFC,MocaplabDatatestsetFC
from cnn.cnn import TestCNN
from fc.fc import MocaplabFC
from train import test
from src.dataset.mcl_io import read_csv as mcl_read_csv
import glob

model_sel = {"mono_bi": ["FC", "FC_bigger_à X mains_20250103_173828",6], "Main_proches":["FC","FC_bigger_Mains_20250103_174622", 3]}

if __name__ == "__main__":
    LOSS_FUNCTION = torch.nn.CrossEntropyLoss()
    print("#### Set-Up ####")
    
    # Set-up Python
    setup_python()

    # Set-up PyTorch
    DEVICE = setup_pytorch()
    print("#### Dataset ####")
    data_path = '%s/data/mocaplab/Autoannotation'%src_folder
    # bones_to_keep = list(set("CC_Base_Waist;CC_Base_Waist;CC_Base_Waist;CC_Base_Waist;CC_Base_Waist;CC_Base_Waist;CC_Base_Spine01;CC_Base_Spine01;CC_Base_Spine01;CC_Base_Spine01;CC_Base_Spine01;CC_Base_Spine01;CC_Base_Spine02;CC_Base_Spine02;CC_Base_Spine02;CC_Base_Spine02;CC_Base_Spine02;CC_Base_Spine02;CC_Base_R_Clavicle;CC_Base_R_Clavicle;CC_Base_R_Clavicle;CC_Base_R_Clavicle;CC_Base_R_Clavicle;CC_Base_R_Clavicle;CC_Base_R_Upperarm;CC_Base_R_Upperarm;CC_Base_R_Upperarm;CC_Base_R_Upperarm;CC_Base_R_Upperarm;CC_Base_R_Upperarm;CC_Base_R_UpperarmTwist02;CC_Base_R_UpperarmTwist02;CC_Base_R_UpperarmTwist02;CC_Base_R_UpperarmTwist02;CC_Base_R_UpperarmTwist02;CC_Base_R_UpperarmTwist02;CC_Base_L_Clavicle;CC_Base_L_Clavicle;CC_Base_L_Clavicle;CC_Base_L_Clavicle;CC_Base_L_Clavicle;CC_Base_L_Clavicle;CC_Base_L_Upperarm;CC_Base_L_Upperarm;CC_Base_L_Upperarm;CC_Base_L_Upperarm;CC_Base_L_Upperarm;CC_Base_L_Upperarm;CC_Base_L_UpperarmTwist02;CC_Base_L_UpperarmTwist02;CC_Base_L_UpperarmTwist02;CC_Base_L_UpperarmTwist02;CC_Base_L_UpperarmTwist02;CC_Base_L_UpperarmTwist02;C_Base_NeckTwist01;CC_Base_NeckTwist01;CC_Base_NeckTwist01;CC_Base_NeckTwist01;CC_Base_NeckTwist01;CC_Base_NeckTwist01;CC_Base_NeckTwist02;CC_Base_NeckTwist02;CC_Base_NeckTwist02;CC_Base_NeckTwist02;CC_Base_NeckTwist02;CC_Base_NeckTwist02;CC_Base_Head;CC_Base_Head;CC_Base_Head;CC_Base_Head;CC_Base_Head;CC_Base_Head;lowerarm_l;lowerarm_l;lowerarm_l;lowerarm_l;lowerarm_l;lowerarm_l;lowerarm_twist_01_l;lowerarm_twist_01_l;lowerarm_twist_01_l;lowerarm_twist_01_l;lowerarm_twist_01_l;lowerarm_twist_01_l;hand_l;hand_l;hand_l;hand_l;hand_l;hand_l;lowerarm_r;lowerarm_r;lowerarm_r;lowerarm_r;lowerarm_r;lowerarm_r;lowerarm_twist_01_r;lowerarm_twist_01_r;lowerarm_twist_01_r;lowerarm_twist_01_r;lowerarm_twist_01_r;lowerarm_twist_01_r;hand_r;hand_r;hand_r;hand_r;hand_r;hand_r".split(';')))
    bones_to_keep = ['CC_Base_Head', 'CC_Base_L_Clavicle', 'CC_Base_L_Upperarm', 'CC_Base_L_UpperarmTwist02', 'CC_Base_NeckTwist01', 'CC_Base_NeckTwist02', 'CC_Base_R_Clavicle', 'CC_Base_R_Upperarm', 'CC_Base_R_UpperarmTwist02', 'CC_Base_Spine01', 'CC_Base_Spine02', 'CC_Base_Waist', 'C_Base_NeckTwist01', 'hand_l', 'hand_r', 'index_01_l', 'index_01_r', 'index_02_l', 'index_02_r', 'index_03_l', 'index_03_r', 'index_metacarpal_l', 'index_metacarpal_r', 'lowerarm_l', 'lowerarm_r', 'lowerarm_twist_01_l', 'lowerarm_twist_01_r', 'middle_01_l', 'middle_01_r', 'middle_02_l', 'middle_02_r', 'middle_03_l', 'middle_03_r', 'middle_metacarpal_l', 'middle_metacarpal_r', 'pinky_01_l', 'pinky_01_r', 'pinky_02_l', 'pinky_02_r', 'pinky_03_l', 'pinky_03_r', 'pinky_metacarpal_l', 'pinky_metacarpal_r', 'ring_01_l', 'ring_01_r', 'ring_02_l', 'ring_02_r', 'ring_03_l', 'ring_03_r', 'ring_metacarpal_l', 'ring_metacarpal_r', 'thumb_01_l', 'thumb_01_r', 'thumb_02_l', 'thumb_02_r', 'thumb_03_l', 'thumb_03_r']
    data, _,_ = mcl_read_csv(data_path + "/MLD_X0006_00003-00398-00686-1_CAM_V3.csv", bones_to_keep=bones_to_keep)
    
    data_neutal = data[0:1,:]
    list_models = glob.glob(r'D:\Github\ProjetCassiopeeMCL\src\models\mocaplab\all\saved_models\*\*.ckpt')
    data_ref = os.path.join(data_path,"Annotation_gloses.csv")
    with open(data_ref,"r", encoding="utf-8") as f:
        header = f.readline().strip().split("	")
    for i in list_models:
        model_name = os.path.basename(i).split('.')[0]
        out = model_name.split('_')
        model_data = out[0]
        if len(out) == 4:
            predicted_name = out[1]
        if len(out) >= 5:
            if out[1] == "bigger":
                predicted_name = '_'.join(out[2:-2])
            else:
                predicted_name = '_'.join(out[1:-2])
        col_num = header.index(predicted_name)

        if model_data == "CNN":
            dataset_cnn_labelled = MocaplabDatasetCNN(path=data_path,
                                    padding=True, 
                                    bones_to_keep=bones_to_keep, center=data_neutal, col_num=col_num, max_length=1736)
            dataset_cnn = MocaplabDatatestsetCNN(path=data_path,
                                    padding=True, 
                                    bones_to_keep=bones_to_keep, center=data_neutal, col_num=col_num, max_length=1736)
        elif model_data == "FC":
            dataset_cnn_labelled = MocaplabDatasetFC(path=data_path,
                                        padding=True, 
                                        bones_to_keep=bones_to_keep, center=data_neutal, col_num=col_num, max_length=1736)
            dataset_cnn = MocaplabDatatestsetFC(path=data_path,
                                        padding=True, 
                                        bones_to_keep=bones_to_keep, center=data_neutal, col_num=col_num, max_length=1736)
        print("#### Data Loader ####")
        
        # data_loader_fc = DataLoader(dataset_fc,
        #                          batch_size=1,
        #                          shuffle=False)

        
        print(dataset_cnn.max_length)
        print(len(dataset_cnn))
        # print(len(dataset_cnn_labelled))
        data_loader_cnn = DataLoader(dataset_cnn,
                                    batch_size=1,
                                    shuffle=False)
        test_data_loader = DataLoader(dataset_cnn_labelled,
                                    batch_size=1,
                                    shuffle=False)
        if model_data == "CNN":
            cnn = TestCNN(nb_classes=2).to(DEVICE)
        elif model_data == "FC":
            cnn = MocaplabFC(dataset_cnn_labelled.max_length*dataset_cnn_labelled[0][0].shape[1], loss=LOSS_FUNCTION, numclass=2).to(DEVICE)

        # Load the trained weights cnn old model
        dict_model = torch.load((f"{src_folder}/src/models/mocaplab/all/saved_models/{model_data}/{model_name}.ckpt"), map_location=torch.device("cpu"))
        if "_lossfunc.weight" in dict_model.keys():
            dict_model.pop("_lossfunc.weight")
        cnn.load_state_dict(dict_model)
        # set the evaluation mode
        class_weights_dict = dataset_cnn_labelled.get_labels_weights()
        test_acc, test_confusion_matrix, misclassified = test(cnn, model_data, test_data_loader, DEVICE, weight=[class_weights_dict[label] for label in class_weights_dict.keys()])
        print(f"Test accuracy: {test_acc}")
        fig, (ax1) = plt.subplots(1,1)
        sns.heatmap(test_confusion_matrix, annot=True, cmap="flare",  fmt="d", cbar=True, ax=ax1)
        plt.savefig(f"{src_folder}/train_results/mocaplab/{model_data}_{dataset_cnn.col_name}_skl.png")

        # Load the trained weights cnn new model
        #cnn.load_state_dict(torch.load(("/home/self_supervised_learning_gr/self_supervised_learning/dev/ProjetCassiopee/src/models/mocaplab/cnn/saved_models/CNN_20240514_211739.ckpt"),
                                        # map_location=torch.device("cpu")))

        
        print("#### Test ####")
        heatmap_data = []   # a table that contains 112 lists of 100 lists of size 10 (10 max joints for each frame for each data)
        print(len(data_loader_cnn))
        for k, img in enumerate(data_loader_cnn):
            ###TO GET THE HEATMAP LIST OF 10 MOST IMPORTANT JOINTS###
            img, name = img
            img = img.to(torch.float32).to(DEVICE)
            
            #print(f"img {os.path.splitext(name[0])[0]}: {k:4} / {len(data_loader_cnn)} ")
            # get the most likely prediction of the model
            if model_data == "FC":
                img = img.view(img.size(0), -1)
            pred = cnn(img)
            _, label_pred = torch.max(pred.data, dim=1)
            label_pred = label_pred.detach().item()
            # if pred[0,1].detach().numpy()>0.98:
            #     label_pred = "Oui"
            # elif pred[0,0].detach().numpy()>0.98:
            #     label_pred = "Non"
            # else:
            #     label_pred = "Unknown"
            val = f"{name[0].split('.csv')[0]}\t\t{label_pred}\t{np.array2string(pred[0,:].detach().cpu().numpy(), precision=3, floatmode='fixed', separator=',', suppress_small=True)[1:-1]}"
            with open(f"{src_folder}/test_results/mocaplab/results_{dataset_cnn.col_name}{model_data}.csv", "a") as f:
                f.write("%s\n"%val)
            print(val)
