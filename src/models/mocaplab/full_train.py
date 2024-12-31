from datetime import datetime
import sys
import os

src_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..\..\..'))
sys.path.append(src_folder)
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

from train import train, test
from plot_results import plot_results

from src.dataset import MocaplabDatasetCNN
from cnn.cnn import TestCNN
from src.dataset import MocaplabDatasetFC
from fc.fc import MocaplabFC
from src.dataset import MocaplabDatasetLSTM
from lstm.lstm import LSTM
from src.setup import setup_python, setup_pytorch


if __name__ == "__main__" :

    # Begin set-up
    print("#### Set-Up ####")

    # Set-up Python
    setup_python()

    # Set-up PyTorch
    DEVICE = setup_pytorch()

    # Dataset parameters

    
    # '''
    # Fully connected Training
    # '''
    sample_weight = [1., 1.]
    # # Training parameters
    BATCH_SIZE = 5 # Batch size
    # LOSS_FUNCTION = torch.nn.CrossEntropyLoss(weight=torch.tensor(sample_weight, dtype=torch.float).to(DEVICE)) # Loss function
    LOSS_FUNCTION = torch.nn.CrossEntropyLoss() # Loss function
    OPTIMIZER_TYPE = "Adam"                      # Type of optimizer "Adam" or "SGD"
    EPOCHS = [999999]                      # Number of epochs
    LEARNING_RATES = [0.01]     # Learning rates
    EARLY_STOPPING = True # Early stopping flag
    PATIENCE = 20        # Early stopping patience
    MIN_DELTA = 0.001     # Early stopping minimum delta

    DEBUG = False # Debug flag

    generator = torch.Generator()
    generator.manual_seed(0)
    
    # # Datasets
    print("#### FC Datasets ####")
    # # bones_to_keep = "abdomenUpper_T_glob;abdomenUpper_T_glob;abdomenUpper_T_glob;chestLower_T_glob;chestLower_T_glob;chestLower_T_glob;chestUpper_T_glob;chestUpper_T_glob;chestUpper_T_glob;neckLower_T_glob;neckLower_T_glob;neckLower_T_glob;rCollar_T_glob;rCollar_T_glob;rCollar_T_glob;rShldrBend_T_glob;rShldrBend_T_glob;rShldrBend_T_glob;rShldrTwist_T_glob;rShldrTwist_T_glob;rShldrTwist_T_glob;rForearmBend_T_glob;rForearmBend_T_glob;rForearmBend_T_glob;rForearmTwist_T_glob;rForearmTwist_T_glob;rForearmTwist_T_glob;rHand_T_glob;rHand_T_glob;rHand_T_glob;rCarpal4_T_glob;rCarpal4_T_glob;rCarpal4_T_glob;rPinky1_T_glob;rPinky1_T_glob;rPinky1_T_glob;rPinky2_T_glob;rPinky2_T_glob;rPinky2_T_glob;rPinky3_T_glob;rPinky3_T_glob;rPinky3_T_glob;rPinky3_end_T_glob;rPinky3_end_T_glob;rPinky3_end_T_glob;rCarpal3_T_glob;rCarpal3_T_glob;rCarpal3_T_glob;rRing1_T_glob;rRing1_T_glob;rRing1_T_glob;rRing2_T_glob;rRing2_T_glob;rRing2_T_glob;rRing3_T_glob;rRing3_T_glob;rRing3_T_glob;rRing3_end_T_glob;rRing3_end_T_glob;rRing3_end_T_glob;rCarpal2_T_glob;rCarpal2_T_glob;rCarpal2_T_glob;rMid1_T_glob;rMid1_T_glob;rMid1_T_glob;rMid2_T_glob;rMid2_T_glob;rMid2_T_glob;rMid3_T_glob;rMid3_T_glob;rMid3_T_glob;rMid3_end_T_glob;rMid3_end_T_glob;rMid3_end_T_glob;rCarpal1_T_glob;rCarpal1_T_glob;rCarpal1_T_glob;rIndex1_T_glob;rIndex1_T_glob;rIndex1_T_glob;rIndex2_T_glob;rIndex2_T_glob;rIndex2_T_glob;rIndex3_T_glob;rIndex3_T_glob;rIndex3_T_glob;rIndex3_end_T_glob;rIndex3_end_T_glob;rIndex3_end_T_glob;rThumb1_T_glob;rThumb1_T_glob;rThumb1_T_glob;rThumb2_T_glob;rThumb2_T_glob;rThumb2_T_glob;rThumb3_T_glob;rThumb3_T_glob;rThumb3_T_glob;rThumb3_end_T_glob;rThumb3_end_T_glob;rThumb3_end_T_glob;lCollar_T_glob;lCollar_T_glob;lCollar_T_glob;lShldrBend_T_glob;lShldrBend_T_glob;lShldrBend_T_glob;lShldrTwist_T_glob;lShldrTwist_T_glob;lShldrTwist_T_glob;lForearmBend_T_glob;lForearmBend_T_glob;lForearmBend_T_glob;lForearmTwist_T_glob;lForearmTwist_T_glob;lForearmTwist_T_glob;lHand_T_glob;lHand_T_glob;lHand_T_glob;lCarpal4_T_glob;lCarpal4_T_glob;lCarpal4_T_glob;lPinky1_T_glob;lPinky1_T_glob;lPinky1_T_glob;lPinky2_T_glob;lPinky2_T_glob;lPinky2_T_glob;lPinky3_T_glob;lPinky3_T_glob;lPinky3_T_glob;lPinky3_end_T_glob;lPinky3_end_T_glob;lPinky3_end_T_glob;lCarpal3_T_glob;lCarpal3_T_glob;lCarpal3_T_glob;lRing1_T_glob;lRing1_T_glob;lRing1_T_glob;lRing2_T_glob;lRing2_T_glob;lRing2_T_glob;lRing3_T_glob;lRing3_T_glob;lRing3_T_glob;lRing3_end_T_glob;lRing3_end_T_glob;lRing3_end_T_glob;lCarpal2_T_glob;lCarpal2_T_glob;lCarpal2_T_glob;lMid1_T_glob;lMid1_T_glob;lMid1_T_glob;lMid2_T_glob;lMid2_T_glob;lMid2_T_glob;lMid3_T_glob;lMid3_T_glob;lMid3_T_glob;lMid3_end_T_glob;lMid3_end_T_glob;lMid3_end_T_glob;lCarpal1_T_glob;lCarpal1_T_glob;lCarpal1_T_glob;lIndex1_T_glob;lIndex1_T_glob;lIndex1_T_glob;lIndex2_T_glob;lIndex2_T_glob;lIndex2_T_glob;lIndex3_T_glob;lIndex3_T_glob;lIndex3_T_glob;lIndex3_end_T_glob;lIndex3_end_T_glob;lIndex3_end_T_glob;lThumb1_T_glob;lThumb1_T_glob;lThumb1_T_glob;lThumb2_T_glob;lThumb2_T_glob;lThumb2_T_glob;lThumb3_T_glob;lThumb3_T_glob;lThumb3_T_glob;lThumb3_end_T_glob;lThumb3_end_T_glob;lThumb3_end_T_glob".split(';')

    # bones_to_keep = list(set("C7;C7;C7;T10;T10;T10;LBAC;LBAC;LBAC;RBAC;RBAC;RBAC;CLAV;CLAV;CLAV;STRN;STRN;STRN;LCLAV;LCLAV;LCLAV;RCLAV;RCLAV;RCLAV;LFSHO;LFSHO;LFSHO;LSHOULD;LSHOULD;LSHOULD;LBSHO;LBSHO;LBSHO;LUPA;LUPA;LUPA;LELB;LELB;LELB;LELBEXT;LELBEXT;LELBEXT;LFRM;LFRM;LFRM;LWRA;LWRA;LWRA;LWRB;LWRB;LWRB;RFSHO;RFSHO;RFSHO;RSHOULD;RSHOULD;RSHOULD;RBSHO;RBSHO;RBSHO;RUPA;RUPA;RUPA;RELB;RELB;RELB;RELBEXT;RELBEXT;RELBEXT;RFRM;RFRM;RFRM;RWRA;RWRA;RWRA;RWRB;RWRB;RWRB;LFWT;LFWT;LFWT;RFWT;RFWT;RFWT;LBWT;LBWT;LBWT;RBWT;RBWT;RBWT".split(';')))
    bones_to_keep = list(set("CC_Base_Waist;CC_Base_Waist;CC_Base_Waist;CC_Base_Waist;CC_Base_Waist;CC_Base_Waist;CC_Base_Spine01;CC_Base_Spine01;CC_Base_Spine01;CC_Base_Spine01;CC_Base_Spine01;CC_Base_Spine01;CC_Base_Spine02;CC_Base_Spine02;CC_Base_Spine02;CC_Base_Spine02;CC_Base_Spine02;CC_Base_Spine02;CC_Base_R_Clavicle;CC_Base_R_Clavicle;CC_Base_R_Clavicle;CC_Base_R_Clavicle;CC_Base_R_Clavicle;CC_Base_R_Clavicle;CC_Base_R_Upperarm;CC_Base_R_Upperarm;CC_Base_R_Upperarm;CC_Base_R_Upperarm;CC_Base_R_Upperarm;CC_Base_R_Upperarm;CC_Base_R_UpperarmTwist02;CC_Base_R_UpperarmTwist02;CC_Base_R_UpperarmTwist02;CC_Base_R_UpperarmTwist02;CC_Base_R_UpperarmTwist02;CC_Base_R_UpperarmTwist02;CC_Base_L_Clavicle;CC_Base_L_Clavicle;CC_Base_L_Clavicle;CC_Base_L_Clavicle;CC_Base_L_Clavicle;CC_Base_L_Clavicle;CC_Base_L_Upperarm;CC_Base_L_Upperarm;CC_Base_L_Upperarm;CC_Base_L_Upperarm;CC_Base_L_Upperarm;CC_Base_L_Upperarm;CC_Base_L_UpperarmTwist02;CC_Base_L_UpperarmTwist02;CC_Base_L_UpperarmTwist02;CC_Base_L_UpperarmTwist02;CC_Base_L_UpperarmTwist02;CC_Base_L_UpperarmTwist02;C_Base_NeckTwist01;CC_Base_NeckTwist01;CC_Base_NeckTwist01;CC_Base_NeckTwist01;CC_Base_NeckTwist01;CC_Base_NeckTwist01;CC_Base_NeckTwist02;CC_Base_NeckTwist02;CC_Base_NeckTwist02;CC_Base_NeckTwist02;CC_Base_NeckTwist02;CC_Base_NeckTwist02;CC_Base_Head;CC_Base_Head;CC_Base_Head;CC_Base_Head;CC_Base_Head;CC_Base_Head;lowerarm_l;lowerarm_l;lowerarm_l;lowerarm_l;lowerarm_l;lowerarm_l;lowerarm_twist_01_l;lowerarm_twist_01_l;lowerarm_twist_01_l;lowerarm_twist_01_l;lowerarm_twist_01_l;lowerarm_twist_01_l;hand_l;hand_l;hand_l;hand_l;hand_l;hand_l;lowerarm_r;lowerarm_r;lowerarm_r;lowerarm_r;lowerarm_r;lowerarm_r;lowerarm_twist_01_r;lowerarm_twist_01_r;lowerarm_twist_01_r;lowerarm_twist_01_r;lowerarm_twist_01_r;lowerarm_twist_01_r;hand_r;hand_r;hand_r;hand_r;hand_r;hand_r;index_metacarpal_r;index_metacarpal_r;index_metacarpal_r;index_metacarpal_r;index_metacarpal_r;index_metacarpal_r;index_01_r;index_01_r;index_01_r;index_01_r;index_01_r;index_01_r;index_02_r;index_02_r;index_02_r;index_02_r;index_02_r;index_02_r;index_03_r;index_03_r;index_03_r;index_03_r;index_03_r;index_03_r;middle_metacarpal_r;middle_metacarpal_r;middle_metacarpal_r;middle_metacarpal_r;middle_metacarpal_r;middle_metacarpal_r;middle_01_r;middle_01_r;middle_01_r;middle_01_r;middle_01_r;middle_01_r;middle_02_r;middle_02_r;middle_02_r;middle_02_r;middle_02_r;middle_02_r;middle_03_r;middle_03_r;middle_03_r;middle_03_r;middle_03_r;middle_03_r;thumb_01_r;thumb_01_r;thumb_01_r;thumb_01_r;thumb_01_r;thumb_01_r;thumb_02_r;thumb_02_r;thumb_02_r;thumb_02_r;thumb_02_r;thumb_02_r;thumb_03_r;thumb_03_r;thumb_03_r;thumb_03_r;thumb_03_r;thumb_03_r;ring_metacarpal_r;ring_metacarpal_r;ring_metacarpal_r;ring_metacarpal_r;ring_metacarpal_r;ring_metacarpal_r;ring_01_r;ring_01_r;ring_01_r;ring_01_r;ring_01_r;ring_01_r;ring_02_r;ring_02_r;ring_02_r;ring_02_r;ring_02_r;ring_02_r;ring_03_r;ring_03_r;ring_03_r;ring_03_r;ring_03_r;ring_03_r;pinky_metacarpal_r;pinky_metacarpal_r;pinky_metacarpal_r;pinky_metacarpal_r;pinky_metacarpal_r;pinky_metacarpal_r;pinky_01_r;pinky_01_r;pinky_01_r;pinky_01_r;pinky_01_r;pinky_01_r;pinky_02_r;pinky_02_r;pinky_02_r;pinky_02_r;pinky_02_r;pinky_02_r;pinky_03_r;pinky_03_r;pinky_03_r;pinky_03_r;pinky_03_r;pinky_03_r;index_metacarpal_l;index_metacarpal_l;index_metacarpal_l;index_metacarpal_l;index_metacarpal_l;index_metacarpal_l;index_01_l;index_01_l;index_01_l;index_01_l;index_01_l;index_01_l;index_02_l;index_02_l;index_02_l;index_02_l;index_02_l;index_02_l;index_03_l;index_03_l;index_03_l;index_03_l;index_03_l;index_03_l;middle_metacarpal_l;middle_metacarpal_l;middle_metacarpal_l;middle_metacarpal_l;middle_metacarpal_l;middle_metacarpal_l;middle_01_l;middle_01_l;middle_01_l;middle_01_l;middle_01_l;middle_01_l;middle_02_l;middle_02_l;middle_02_l;middle_02_l;middle_02_l;middle_02_l;middle_03_l;middle_03_l;middle_03_l;middle_03_l;middle_03_l;middle_03_l;thumb_01_l;thumb_01_l;thumb_01_l;thumb_01_l;thumb_01_l;thumb_01_l;thumb_02_l;thumb_02_l;thumb_02_l;thumb_02_l;thumb_02_l;thumb_02_l;thumb_03_l;thumb_03_l;thumb_03_l;thumb_03_l;thumb_03_l;thumb_03_l;ring_metacarpal_l;ring_metacarpal_l;ring_metacarpal_l;ring_metacarpal_l;ring_metacarpal_l;ring_metacarpal_l;ring_01_l;ring_01_l;ring_01_l;ring_01_l;ring_01_l;ring_01_l;ring_02_l;ring_02_l;ring_02_l;ring_02_l;ring_02_l;ring_02_l;ring_03_l;ring_03_l;ring_03_l;ring_03_l;ring_03_l;ring_03_l;pinky_metacarpal_l;pinky_metacarpal_l;pinky_metacarpal_l;pinky_metacarpal_l;pinky_metacarpal_l;pinky_metacarpal_l;pinky_01_l;pinky_01_l;pinky_01_l;pinky_01_l;pinky_01_l;pinky_01_l;pinky_02_l;pinky_02_l;pinky_02_l;pinky_02_l;pinky_02_l;pinky_02_l;pinky_03_l;pinky_03_l;pinky_03_l;pinky_03_l;pinky_03_l;pinky_03_l".split(';')))
    data_path = '%s/data/mocaplab/Autoannotation'%src_folder
    from src.dataset.mcl_io import read_csv as mcl_read_csv
    data, _,_ = mcl_read_csv(data_path + "/MLD_X0006_00003-00398-00686-1_CAM_V3.csv", bones_to_keep=bones_to_keep)
    dataset = MocaplabDatasetFC(data_path, padding=True, bones_to_keep=bones_to_keep, center=data[0,:])
    print(dataset)
    
    # print(dataset.get_labels_weights())
    # Split dataset
    n = len(dataset)
    class_weights_dict = dataset.get_labels_weights() # inverse relative amount of samples per class
    sample_weights = [0] * n
    class_weights = [class_weights_dict[label] for label in class_weights_dict.keys()]
    for idx in range(n):
        (data, label, name) = dataset[idx]
        sample_weights[idx] = class_weights_dict[label]

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    split = [int(n*0.6), int(n*0.2), int(n*0.2)]
    diff = n - split[0] - split[1] - split[2]
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[split[0], split[1], split[2]+diff], generator=generator)
    #50% data
    #diff = n - int(n*0.3) - int(n*0.2) - int(n*0.5)
    #train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(n*0.15), int(n*0.1), int(n*0.75)+diff], generator=generator)
    
    #25% data
    #diff = n - int(n*0.15) - int(n*0.1) - int(n*0.75)
    #train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(n*0.15), int(n*0.1), int(n*0.75)+diff], generator=generator)
    
    #10% data
    #diff = n - int(n*0.05) - int(n*0.05) - int(n*0.9)
    #train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(n*0.05), int(n*0.05), int(n*0.90)+diff], generator=generator)
    
    print(f"Total length -> {len(dataset)} samples")
    print(f"Train dataset -> {len(train_dataset)} samples")
    print(f"Test dataset -> {len(test_dataset)} samples")
    print(f"Validation dataset -> {len(validation_dataset)} samples")
    
    # Data loaders
    print("#### FC Data Loaders ####")

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=False)
    
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False)
    
    validation_data_loader = DataLoader(validation_dataset,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)
    
    # Create neural network
    print("#### FC Model ####")
    model = MocaplabFC(dataset.max_length*dataset[0][0].shape[1], loss=LOSS_FUNCTION, numclass=2).to(DEVICE)

    """state_dict = torch.load("self_supervised_learning/dev/ProjetCassiopee/data/mocaplab/Cassiopée_Allbones")
    
    flattened_state_dict = {}
    for key, val in state_dict.items():
        for sub_key, sub_val in val.items():
            new_key = key + '.' + sub_key
            flattened_state_dict[new_key] = sub_val
    
    model.load_state_dict(state_dict=flattened_state_dict)

    # Désactiver le calcul du gradient pour tous les paramètres du modèle
    for param in model.parameters():
        param.requires_grad = False

    # Activer le calcul du gradient pour les paramètres de fc1, fc2, fc3
    for param in model.fc1.parameters():
        param.requires_grad = True
    #for param in model.fc2.parameters():
    #    param.requires_grad = True
    #for param in model.fc3.parameters():
    #    param.requires_grad = True"""

    #Save training time start
    start_timestamp = datetime.now()

    # Create path for saving the model and results
    model_path = f"FC_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
    #model_path = f"FC_50%_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
    #model_path = f"FC_25%_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
    #model_path = f"FC_10%_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
    #model_path = f"SSL_FC_10%_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"

    # Begin training
    print("#### FC Training ####")

    #Train model
    try:
        train_acc, train_loss, val_acc, val_loss, run_epochs = train(model,
                                                                    train_data_loader,
                                                                    validation_data_loader,
                                                                    LOSS_FUNCTION,
                                                                    OPTIMIZER_TYPE,
                                                                    EPOCHS,
                                                                    LEARNING_RATES,
                                                                    EARLY_STOPPING,
                                                                    PATIENCE,
                                                                    MIN_DELTA,
                                                                    DEVICE,
                                                                    DEBUG,
                                                                    class_weights=class_weights,
                                                                    model_type="FC")
        
        # Save training time stop
        stop_timestamp = datetime.now()
        
        # Test model
        
        test_acc, test_confusion_matrix, misclassified = test(model, "FC",test_data_loader, DEVICE)

        # Plot results
        if test_acc > 0.8:
            plot_results(train_acc, train_loss,
                        val_acc, val_loss,
                        run_epochs, type(model).__name__, start_timestamp, DEVICE,
                        LOSS_FUNCTION, OPTIMIZER_TYPE,
                        EPOCHS, LEARNING_RATES, EARLY_STOPPING, PATIENCE, MIN_DELTA,
                        test_acc, test_confusion_matrix, stop_timestamp, model_path,
                        [])
                        #misclassified)
        
        # Save model
        if test_acc > 0.8:
            torch.save(model.state_dict(), "%s/src/models/mocaplab/all/saved_models/FC/%s.ckpt"%(src_folder, model_path))
        
        #End training
        print("#### FC End ####")
    except Exception as e:
        print(f"Error: {e}")
    
    










    try:
        '''
        LSTM Training
        '''
        # Training parameters
        BATCH_SIZE = 5 # Batch size
        EPOCHS = [999999]                      # Number of epochs
        LEARNING_RATES = [0.001]     # Learning rates
        EARLY_STOPPING = True # Early stopping flag
        PATIENCE = 20        # Early stopping patience
        MIN_DELTA = 0.001     # Early stopping minimum delta

        DEBUG = False # Debug flag
        
        # Datasets
        print("#### LSTM Datasets ####")
        dataset = MocaplabDatasetLSTM(path=data_path,
                                      padding = True,
                                      bones_to_keep=bones_to_keep, 
                                      center=data[0,:])

        # Split dataset
        n = len(dataset)

        train_dataset = Subset(dataset, train_dataset.indices)
        validation_dataset = Subset(dataset, validation_dataset.indices)
        test_dataset = Subset(dataset, test_dataset.indices)
        
        print(f"Total length -> {len(dataset)} samples")
        print(f"Train dataset -> {len(train_dataset)} samples")
        print(f"Test dataset -> {len(test_dataset)} samples")
        print(f"Validation dataset -> {len(validation_dataset)} samples")
        
        # Data loaders
        print("#### LSTM Data Loaders ####")

        train_data_loader = DataLoader(train_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False)
        
        test_data_loader = DataLoader(test_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False)
        
        validation_data_loader = DataLoader(validation_dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False)
        
        # Create neural network
        print("#### LSTM Model ####")
        model = LSTM(input_size=dataset[0][0].shape[1], hidden_size=48, num_layers=4, output_size=2).to(DEVICE)

        # Save training time start
        start_timestamp = datetime.now()

        # Create path for saving things...
        model_path = f"LSTM_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
        #model_path = f"LSTM_50%_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
        #model_path = f"LSTM_25%_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
        #model_path = f"LSTM_10%_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"

        # Begin training
        print("#### LSTM Training ####")

        # Train model
        train_acc, train_loss, val_acc, val_loss, run_epochs = train(model,
                                                                    train_data_loader,
                                                                    validation_data_loader,
                                                                    LOSS_FUNCTION,
                                                                    OPTIMIZER_TYPE,
                                                                    EPOCHS,
                                                                    LEARNING_RATES,
                                                                    EARLY_STOPPING,
                                                                    PATIENCE,
                                                                    MIN_DELTA,
                                                                    DEVICE,
                                                                    DEBUG,
                                                                    class_weights=class_weights,
                                                                    model_type="LSTM")
        
        # Save training time stop
        stop_timestamp = datetime.now()
        
        # Test model
        test_acc, test_confusion_matrix, misclassified = test(model, "LSTM",test_data_loader, DEVICE)

        # Plot results
        plot_results(train_acc, train_loss,
                    val_acc, val_loss,
                    run_epochs, type(model).__name__, start_timestamp, DEVICE,
                    LOSS_FUNCTION, OPTIMIZER_TYPE,
                    EPOCHS, LEARNING_RATES, EARLY_STOPPING, PATIENCE, MIN_DELTA,
                    test_acc, test_confusion_matrix, stop_timestamp, model_path,
                    [])
        
        # Save model
        if test_acc > 0.8:
            torch.save(model.state_dict(), "%s/src/models/mocaplab/all/saved_models/LSTM/%s.ckpt"%(src_folder, model_path))
        
        # End training
        print("#### LSTM End ####")
    except Exception as e:
        print(f"Error: {e}")

    










    '''
    CNN Training
    '''
    print("#### CNN Datasets ####")
    #bones_to_keep = None
    dataset = MocaplabDatasetCNN(data_path, padding=True, bones_to_keep=bones_to_keep)
    # Split dataset
    n = len(dataset)
    
    train_dataset = Subset(dataset, train_dataset.indices)
    validation_dataset = Subset(dataset, validation_dataset.indices)
    test_dataset = Subset(dataset, test_dataset.indices)
    from collections import defaultdict

    label_counter = defaultdict(int)

    for data, label, name in train_dataset:
        label_counter[label] += 1
    print(label_counter)
    # weight = [0]*n
    # weight[dataset.labels == 0] = 0.27
    # weight[dataset.labels == 1] = 0.73

    #WeightedRandomSampler(weight,)
    for i in range(5,6,1):
    # Training parameters
        BATCH_SIZE = i                                  # Batch sizes
        EPOCHS = [999999]                        # Number of epochs
        LEARNING_RATES = [0.001]    # Learning rates
        EARLY_STOPPING = True                           # Early stopping flag
        PATIENCE = 10                                   # Early stopping patience
        MIN_DELTA = 0.01                                # Early stopping minimum delta

        DEBUG = False                                   # Debug flag

        print(f"Total length -> {len(dataset)} samples")
        print(f"Train dataset -> {len(train_dataset)} samples")
        print(f"Test dataset -> {len(test_dataset)} samples")
        print(f"Validation dataset -> {len(validation_dataset)} samples")

        
        # Data loaders
        print("#### CNN Data Loaders ####")

        train_data_loader = DataLoader(train_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False)
        
        test_data_loader = DataLoader(test_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False)
        
        validation_data_loader = DataLoader(validation_dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False)
        
        # Create neural network
        print("#### CNN Model ####")
        model = TestCNN(nb_classes=2).to(DEVICE)
        """state_dict = torch.load("self_supervised_learning/dev/ProjetCassiopee/src/models/mocaplab/all/saved_models/SSL_CNN/encoder_SSL_CNN_90%_20240529_162101_epoch_280.ckpt")
        
        flattened_state_dict = {}
        for key, val in state_dict.items():
            for sub_key, sub_val in val.items():
                new_key = key + '.' + sub_key
                flattened_state_dict[new_key] = sub_val
        
        model.load_state_dict(state_dict=flattened_state_dict)

        # Désactiver le calcul du gradient pour tous les paramètres du modèle
        for param in model.parameters():
            param.requires_grad = False

        # Activer le calcul du gradient pour les paramètres de fc1 et fc2
        for param in model.fc1.parameters():
            param.requires_grad = True
        for param in model.fc2.parameters():
            param.requires_grad = True
        for param in model.fc3.parameters():
            param.requires_grad = True
        for param in model.conv3_3.parameters():
            param.requires_grad = True
        for param in model.conv3_2.parameters():
            param.requires_grad = True
        for param in model.conv3_1.parameters():
            param.requires_grad = True"""

        # Save training time start
        start_timestamp = datetime.now()

        # Create path for saving things...
        model_path = f"CNN_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
        #model_path = f"CNN_50%_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
        #model_path = f"CNN_25%_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
        #model_path = f"CNN_10%_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
        #model_path = f"SSL_CNN_fc-only_10%_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"

        # Begin training
        print("#### CNN Training ####")

        # Train model
        train_acc, train_loss, val_acc, val_loss, run_epochs = train(model,
                                                                    train_data_loader,
                                                                    validation_data_loader,
                                                                    LOSS_FUNCTION,
                                                                    OPTIMIZER_TYPE,
                                                                    EPOCHS,
                                                                    LEARNING_RATES,
                                                                    EARLY_STOPPING,
                                                                    PATIENCE,
                                                                    MIN_DELTA,
                                                                    DEVICE,
                                                                    DEBUG,
                                                                    class_weights=class_weights,
                                                                    model_type="CNN")
        
        # Save training time stop
        stop_timestamp = datetime.now()
        
        # Test model
        test_acc, test_confusion_matrix, misclassified = test(model,"CNN", test_data_loader, DEVICE)

        # Plot results
        if test_acc > 0.8:
            plot_results(train_acc, train_loss,
                        val_acc, val_loss,
                        run_epochs, type(model).__name__, start_timestamp, DEVICE,
                        LOSS_FUNCTION, OPTIMIZER_TYPE,
                        EPOCHS, LEARNING_RATES, EARLY_STOPPING, PATIENCE, MIN_DELTA,
                        test_acc, test_confusion_matrix, stop_timestamp, model_path,
                        [])
        if test_acc > 0.8:
            # Save model
            torch.save(model.state_dict(), "%s/src/models/mocaplab/all/saved_models/CNN/%s.ckpt"%(src_folder, model_path))
            
        # End training

        print("#### CNN End ####")
    
