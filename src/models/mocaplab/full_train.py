from datetime import datetime
import sys
import os

src_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..\..\..'))
sys.path.append(src_folder)
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

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
    sample_weight = [1, 0.2]
    # # Training parameters
    BATCH_SIZE = 30 # Batch size
    LOSS_FUNCTION = torch.nn.CrossEntropyLoss(weight=torch.tensor(sample_weight).to(DEVICE)) # Loss function
    OPTIMIZER_TYPE = "SGD"                      # Type of optimizer "Adam" or "SGD"
    EPOCHS = [2,999999]                      # Number of epochs
    LEARNING_RATES = [0.1,0.01]     # Learning rates
    EARLY_STOPPING = True # Early stopping flag
    PATIENCE = 10        # Early stopping patience
    MIN_DELTA = 0.001     # Early stopping minimum delta

    DEBUG = False # Debug flag

    generator = torch.Generator()
    generator.manual_seed(0)
    
    # # Datasets
    print("#### FC Datasets ####")
    # # bones_to_keep = "abdomenUpper_T_glob;abdomenUpper_T_glob;abdomenUpper_T_glob;chestLower_T_glob;chestLower_T_glob;chestLower_T_glob;chestUpper_T_glob;chestUpper_T_glob;chestUpper_T_glob;neckLower_T_glob;neckLower_T_glob;neckLower_T_glob;rCollar_T_glob;rCollar_T_glob;rCollar_T_glob;rShldrBend_T_glob;rShldrBend_T_glob;rShldrBend_T_glob;rShldrTwist_T_glob;rShldrTwist_T_glob;rShldrTwist_T_glob;rForearmBend_T_glob;rForearmBend_T_glob;rForearmBend_T_glob;rForearmTwist_T_glob;rForearmTwist_T_glob;rForearmTwist_T_glob;rHand_T_glob;rHand_T_glob;rHand_T_glob;rCarpal4_T_glob;rCarpal4_T_glob;rCarpal4_T_glob;rPinky1_T_glob;rPinky1_T_glob;rPinky1_T_glob;rPinky2_T_glob;rPinky2_T_glob;rPinky2_T_glob;rPinky3_T_glob;rPinky3_T_glob;rPinky3_T_glob;rPinky3_end_T_glob;rPinky3_end_T_glob;rPinky3_end_T_glob;rCarpal3_T_glob;rCarpal3_T_glob;rCarpal3_T_glob;rRing1_T_glob;rRing1_T_glob;rRing1_T_glob;rRing2_T_glob;rRing2_T_glob;rRing2_T_glob;rRing3_T_glob;rRing3_T_glob;rRing3_T_glob;rRing3_end_T_glob;rRing3_end_T_glob;rRing3_end_T_glob;rCarpal2_T_glob;rCarpal2_T_glob;rCarpal2_T_glob;rMid1_T_glob;rMid1_T_glob;rMid1_T_glob;rMid2_T_glob;rMid2_T_glob;rMid2_T_glob;rMid3_T_glob;rMid3_T_glob;rMid3_T_glob;rMid3_end_T_glob;rMid3_end_T_glob;rMid3_end_T_glob;rCarpal1_T_glob;rCarpal1_T_glob;rCarpal1_T_glob;rIndex1_T_glob;rIndex1_T_glob;rIndex1_T_glob;rIndex2_T_glob;rIndex2_T_glob;rIndex2_T_glob;rIndex3_T_glob;rIndex3_T_glob;rIndex3_T_glob;rIndex3_end_T_glob;rIndex3_end_T_glob;rIndex3_end_T_glob;rThumb1_T_glob;rThumb1_T_glob;rThumb1_T_glob;rThumb2_T_glob;rThumb2_T_glob;rThumb2_T_glob;rThumb3_T_glob;rThumb3_T_glob;rThumb3_T_glob;rThumb3_end_T_glob;rThumb3_end_T_glob;rThumb3_end_T_glob;lCollar_T_glob;lCollar_T_glob;lCollar_T_glob;lShldrBend_T_glob;lShldrBend_T_glob;lShldrBend_T_glob;lShldrTwist_T_glob;lShldrTwist_T_glob;lShldrTwist_T_glob;lForearmBend_T_glob;lForearmBend_T_glob;lForearmBend_T_glob;lForearmTwist_T_glob;lForearmTwist_T_glob;lForearmTwist_T_glob;lHand_T_glob;lHand_T_glob;lHand_T_glob;lCarpal4_T_glob;lCarpal4_T_glob;lCarpal4_T_glob;lPinky1_T_glob;lPinky1_T_glob;lPinky1_T_glob;lPinky2_T_glob;lPinky2_T_glob;lPinky2_T_glob;lPinky3_T_glob;lPinky3_T_glob;lPinky3_T_glob;lPinky3_end_T_glob;lPinky3_end_T_glob;lPinky3_end_T_glob;lCarpal3_T_glob;lCarpal3_T_glob;lCarpal3_T_glob;lRing1_T_glob;lRing1_T_glob;lRing1_T_glob;lRing2_T_glob;lRing2_T_glob;lRing2_T_glob;lRing3_T_glob;lRing3_T_glob;lRing3_T_glob;lRing3_end_T_glob;lRing3_end_T_glob;lRing3_end_T_glob;lCarpal2_T_glob;lCarpal2_T_glob;lCarpal2_T_glob;lMid1_T_glob;lMid1_T_glob;lMid1_T_glob;lMid2_T_glob;lMid2_T_glob;lMid2_T_glob;lMid3_T_glob;lMid3_T_glob;lMid3_T_glob;lMid3_end_T_glob;lMid3_end_T_glob;lMid3_end_T_glob;lCarpal1_T_glob;lCarpal1_T_glob;lCarpal1_T_glob;lIndex1_T_glob;lIndex1_T_glob;lIndex1_T_glob;lIndex2_T_glob;lIndex2_T_glob;lIndex2_T_glob;lIndex3_T_glob;lIndex3_T_glob;lIndex3_T_glob;lIndex3_end_T_glob;lIndex3_end_T_glob;lIndex3_end_T_glob;lThumb1_T_glob;lThumb1_T_glob;lThumb1_T_glob;lThumb2_T_glob;lThumb2_T_glob;lThumb2_T_glob;lThumb3_T_glob;lThumb3_T_glob;lThumb3_T_glob;lThumb3_end_T_glob;lThumb3_end_T_glob;lThumb3_end_T_glob".split(';')

    # bones_to_keep = list(set("C7;C7;C7;T10;T10;T10;LBAC;LBAC;LBAC;RBAC;RBAC;RBAC;CLAV;CLAV;CLAV;STRN;STRN;STRN;LCLAV;LCLAV;LCLAV;RCLAV;RCLAV;RCLAV;LFSHO;LFSHO;LFSHO;LSHOULD;LSHOULD;LSHOULD;LBSHO;LBSHO;LBSHO;LUPA;LUPA;LUPA;LELB;LELB;LELB;LELBEXT;LELBEXT;LELBEXT;LFRM;LFRM;LFRM;LWRA;LWRA;LWRA;LWRB;LWRB;LWRB;RFSHO;RFSHO;RFSHO;RSHOULD;RSHOULD;RSHOULD;RBSHO;RBSHO;RBSHO;RUPA;RUPA;RUPA;RELB;RELB;RELB;RELBEXT;RELBEXT;RELBEXT;RFRM;RFRM;RFRM;RWRA;RWRA;RWRA;RWRB;RWRB;RWRB;LFWT;LFWT;LFWT;RFWT;RFWT;RFWT;LBWT;LBWT;LBWT;RBWT;RBWT;RBWT".split(';')))
    bones_to_keep = list(set("UPHD;LFHD;RFHD;LBHD;RBHD;C7;C7;C7;T10;T10;T10;LBAC;LBAC;LBAC;RBAC;RBAC;RBAC;CLAV;CLAV;CLAV;STRN;STRN;STRN;LCLAV;LCLAV;LCLAV;RCLAV;RCLAV;RCLAV;LFSHO;LFSHO;LFSHO;LSHOULD;LSHOULD;LSHOULD;LBSHO;LBSHO;LBSHO;LUPA;LUPA;LUPA;LELB;LELB;LELB;LELBEXT;LELBEXT;LELBEXT;LFRM;LFRM;LFRM;LWRA;LWRA;LWRA;LWRB;LWRB;LWRB;RFSHO;RFSHO;RFSHO;RSHOULD;RSHOULD;RSHOULD;RBSHO;RBSHO;RBSHO;RUPA;RUPA;RUPA;RELB;RELB;RELB;RELBEXT;RELBEXT;RELBEXT;RFRM;RFRM;RFRM;RWRA;RWRA;RWRA;RWRB;RWRB;RWRB;LFWT;LFWT;LFWT;RFWT;RFWT;RFWT;LBWT;LBWT;LBWT;RBWT;RBWT;RBWT;LHIP;LHIP;LHIP;LUPLEG;LUPLEG;LUPLEG;LKNE;LKNE;LKNE;LPER;LPER;LPER;LTIB;LTIB;LTIB;LANK;LANK;LANK;LHEE;LHEE;LHEE;LMT5;LMT5;LMT5;LTOE;LTOE;LTOE;LMT1;LMT1;LMT1;RHIP;RHIP;RHIP;RUPLEG;RUPLEG;RUPLEG;RKNE;RKNE;RKNE;RPER;RPER;RPER;RTIB;RTIB;RTIB;RANK;RANK;RANK;RHEE;RHEE;RHEE;RMT5;RMT5;RMT5;RTOE;RTOE;RTOE;RMT1;RMT1;RMT1".split(';')))
    data_path = '%s/data/mocaplab/LSDICOS'%src_folder
    dataset = MocaplabDatasetFC(data_path, padding=True, bones_to_keep=bones_to_keep)
    print(dataset)
    
    print(dataset.get_labels_weights())
    # Split dataset
    n = len(dataset)
    class_weights = dataset.get_labels_weights() # inverse relative amount of samples per class
    sample_weights = [0] * n

    for idx in range(n):
        (data, label, name) = dataset[idx]
        sample_weights[idx] = class_weights[label]

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    split = [int(n*0.8), int(n*0.1), int(n*0.1)]
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
    model = MocaplabFC(dataset.max_length*dataset[0][0].shape[1]).to(DEVICE)

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
                                                                    model_type="FC")
        
        # Save training time stop
        stop_timestamp = datetime.now()
        
        # Test model
        test_acc, test_confusion_matrix, misclassified = test(model, "FC",test_data_loader, DEVICE)

        # Plot results
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
        BATCH_SIZE = 30 # Batch size
        LOSS_FUNCTION = torch.nn.CrossEntropyLoss(weight=torch.tensor(sample_weight).to(DEVICE)) # Loss function
        OPTIMIZER_TYPE = "Adam"                      # Type of optimizer
        EPOCHS = [5,999999]                      # Number of epochs
        LEARNING_RATES = [0.001,0.0001]     # Learning rates
        EARLY_STOPPING = True # Early stopping flag
        PATIENCE = 10        # Early stopping patience
        MIN_DELTA = 0.01     # Early stopping minimum delta

        DEBUG = False # Debug flag

        generator = torch.Generator()
        generator.manual_seed(0)
        
        # Datasets
        print("#### LSTM Datasets ####")
        dataset = MocaplabDatasetLSTM(path=data_path,
                                      padding = True,
                                      bones_to_keep=bones_to_keep)

        # Split dataset
        n = len(dataset)

        split = [int(n*0.8), int(n*0.1), int(n*0.1)]
        diff = n - split[0] - split[1] - split[2]
        train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[split[0], split[1], split[2]+diff], generator=generator)

        #50% data
        #diff = n - int(n*0.3) - int(n*0.2) - int(n*0.5)
        #train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(n*0.15), int(n*0.1), int(n*0.75)+diff], generator=generator)
        
        #25% data
        #diff = n - int(n*0.15) - int(n*0.1) - int(n*0.75)
        #train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(n*0.15), int(n*0.1), int(n*0.75)+diff], generator=generator)
        
        #10% data
        # diff = n - int(n*0.05) - int(n*0.05) - int(n*0.9)
        # train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[int(n*0.05), int(n*0.05), int(n*0.90)+diff], generator=generator)
        
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
    generator = torch.Generator()
    generator.manual_seed(0)
    # Split dataset
    n = len(dataset)
    split = [int(n*0.8), int(n*0.1), int(n*0.1)]
    diff = n - split[0] - split[1] - split[2]
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[split[0], split[1], split[2]+diff], generator=generator)
    from collections import defaultdict

    label_counter = defaultdict(int)

    for data, label, name in train_dataset:
        label_counter[label] += 1
    print(label_counter)
    # weight = [0]*n
    # weight[dataset.labels == 0] = 0.27
    # weight[dataset.labels == 1] = 0.73

    #WeightedRandomSampler(weight,)
    for i in range(25,40,5):
    # Training parameters
        BATCH_SIZE = i                                  # Batch size
        LOSS_FUNCTION = torch.nn.CrossEntropyLoss(weight=torch.tensor(sample_weight).to(DEVICE))     # Loss function
        OPTIMIZER_TYPE = "SGD"                          # Type of optimizer
        EPOCHS = [2,2,15,999999]                               # Number of epochs
        LEARNING_RATES = [0.05, 0.01, 0.001,0.0001]                  # Learning rates
        EARLY_STOPPING = True                           # Early stopping flag
        PATIENCE = 10                               # Early stopping patience
        MIN_DELTA = 0.01                              # Early stopping minimum delta

        DEBUG = False # Debug flag


        
        # Datasets
        
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
    
