import torch
from sklearn.metrics import confusion_matrix

def create_optimizer(optimizer_type, model, learning_rate, momentum=0):
    """
    Create optimizer based on the PyTorch name of the optimizer.

    Parameters
    ----------
    optimizer_type : str
        Optimizer type, i.e. PyTorch name of the optimizer
    model : torch.nn.Module
        Neural network model
    learning_rate : float
        Learning rate

    Returns
    -------
    torch.optim.Optimizer
        Instance of optimizer_type optimizer
    """
    optimizer = None
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate)
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=learning_rate,
                                    momentum=momentum)
    else:
        print(f"Unknow optimizer type: {optimizer_type}")
        
    return optimizer


###############################################################################
#### DO NOT USE FOLLOWING FUNCTIONS ###########################################
###############################################################################


def train_one_epoch(model, data_loader, loss_function, optimizer, loss_mmodality=5, accuracy=True):

    # Enable training
    model.train(True)

    # Initialise accuracy variables
    total = 0
    correct = 0

    # Initialise loss
    train_loss = 0.0

    # Pass over all batches
    for i, batch in enumerate(data_loader):
        rgb, depth, mask, loc_x, loc_y, label = batch

        # Zero gradient
        optimizer.zero_grad()

        # Prepare batch
        # Data preprocessing?

        # Make predictions for batch
        output = model(rgb)

        # Update accuracy variables
        if accuracy:
            _, predicted = torch.max(output.data, 1)
            total += len(label)
            batch_correct = (predicted == label).sum().item()
            correct += batch_correct

        # Compute loss
        loss = loss_function(output, batch[loss_mmodality])

        # Compute gradient loss
        loss.backward()

        # Update weights
        optimizer.step()

        # Update losses
        train_loss += loss.item()

        # Log
        if i % 10 == 0:
            if accuracy:
                print(f"    Batch {i}: accuracy={batch_correct / label.size(0)} | loss={loss}")
            else:
                print(f"    Batch {i}: loss={loss}")

    # Compute validation accuracy and loss
    train_accuracy = None
    if accuracy:
        train_accuracy = correct / total
    train_loss /= (i + 1) # Average loss over all batches of the epoch
    
    return train_accuracy, train_loss


def evaluate(model, data_loader, loss_function, loss_modality=5, accuracy=True):

    # Initialise accuracy variables
    total = 0
    correct = 0

    # Initialise losses
    validation_loss = 0.0

    # Freeze the model
    model.eval()
    with torch.no_grad():

        # Iterate over batches
        for i, batch in enumerate(data_loader):
            rgb, depth, mask, loc_x, loc_y, label = batch
            
            # Prepare batch
            # Data preprocessing?

            # Make predictions for batch
            output = model(rgb)

            # Update accuracy variables
            if accuracy:
                _, predicted = torch.max(output.data, 1)
                total += len(label)
                correct = (predicted == label).sum().item()

            # Compute loss
            loss = loss_function(output, batch[loss_modality])

            # Update batch loss
            validation_loss += loss.item()

    # Compute validation accuracy and loss
    validation_accuracy = None
    if accuracy:
        validation_accuracy = correct / total
    validation_loss /= (i + 1) # Average loss over all batches of the epoch

    return validation_accuracy, validation_loss


def train(
        model,
        train_data_loader,
        validation_data_loader,
        loss_function,
        optimizer_type,
        epochs_list,
        learning_rates_list,
        early_stopping=False,
        patience=5,
        min_delta=1e-3,
        loss_modality=5,
        accuracy=True,
        debug=False):

    # Accuracies
    train_accuracies = []
    validation_accuracies = []

    # Losses
    train_losses = []
    validation_losses = []

    # Early stopping counter
    counter = 0

    # Run epochs counter
    run_epochs = []

    for epochs, learning_rate in list(zip(epochs_list, learning_rates_list)):

        # Create optimizer
        optimizer = create_optimizer(optimizer_type, model, learning_rate)

        for epoch in range(epochs):
            print(f"#### EPOCH {epoch} ####")
            
            # Train for one epoch
            if debug:
                with torch.autograd.detect_anomaly():
                    train_accuracy, train_loss = train_one_epoch(model, train_data_loader, loss_function, optimizer, loss_modality, accuracy)
                    train_accuracies.append(train_accuracy)
                    train_losses.append(train_loss)
            else:
                train_accuracy, train_loss = train_one_epoch(model, train_data_loader, loss_function, optimizer, loss_modality, accuracy)
                train_accuracies.append(train_accuracy)
                train_losses.append(train_loss)

            # Evaluate model
            validation_accuracy, validation_loss = evaluate(model, validation_data_loader, loss_function, loss_modality, accuracy)
            validation_accuracies.append(validation_accuracy)
            validation_losses.append(validation_loss)

            if accuracy:
                print(f"Train: accuracy={train_accuracy} | loss={train_loss}")
                print(f"Validation: accuracy={validation_accuracy} | loss={validation_loss}")
            else:
                print(f"Train: loss={train_loss}")
                print(f"Validation: loss={validation_loss}")

            # Early stopping
            if early_stopping:
                if epoch > 0 and abs(validation_losses[epoch - 1] - validation_losses[epoch]) < min_delta:
                    counter += 1
                    if counter >= patience:
                        print("==== Early Stopping ====")
                        break
                else:
                    counter = 0

        run_epochs.append(epoch + 1)

    return train_accuracies, train_losses, validation_accuracies, validation_losses, run_epochs


def test(model, test_data_loader):

    # Accuracy variables
    correct = 0
    total = 0

    # Confusion matrix variables
    all_label = None
    all_predicted = None

    with torch.no_grad():
        for i, batch in enumerate(test_data_loader):
            rgb, depth, mask, loc_x, loc_y, label = batch

            # Prepare batch
            # Data preprocessing?
            
            # Make predictions for batch
            output = model(rgb)

            # Update accuracy variables
            _, predicted = torch.max(output.data, 1)
            total += len(label)
            correct += (predicted == label).sum().item()

            # Update confusion matrix variables
            if all_label is None and all_predicted is None:
                all_label = label.detach().clone()
                all_predicted = predicted.detach().clone()
            else:
                all_label = torch.cat((all_label, label))
                all_predicted = torch.cat((all_predicted, predicted))
            
    # Compute test accuracy
    test_accuracy = correct / total

    # Create "confusion matrix"
    test_confusion_matrix = confusion_matrix(all_label.cpu(), all_predicted.cpu())

    return test_accuracy, test_confusion_matrix