# PyTorch imports:
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import torchvision
from torch.utils.data import DataLoader

# Other imports:
import math
from tqdm import tqdm


def evaluate(model : torch.nn.Module, data_loader : DataLoader) -> float:
    """
    Evaluates a given model on the accuracy metric

    Parameters:
    -----------
    model (torch.nn.Module):
        The model to be Evaluates.

    data_loader (torch.utils.data.DataLoader):
        The data loader to test the model with

    Returns:
    --------
    accuracy (float):
        The accuracy of the predictions the model generates

    """
    # Selecting the device depending on the machine running this code:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Setting the model to the evaluation mode:
    model.eval()
    # Moving the model to the selected device:
    model = model.to(device)

    num_correct = 0
    count = 0

    for x, y in data_loader:
        # Moving the images and their labels to the selected device:
        x = x.to(device)
        y = y.type(torch.LongTensor)
        y = y.to(device)

        # Forward pass:
        y_pred = model(x)

        _, y_pred = torch.max(y_pred, axis=1)

        # Calculating the number of correct prediction in this batch and
        # accumulating it in a variable:
        num_correct += (y_pred == y).sum().item()

        count += y.shape[0]

    # Calculating the accuracy:
    accuracy = num_correct / count
    return accuracy


def train(model : torch.nn.Module, criterion : torch.nn.Module, optimizer : torch.optim,
          train_loader : DataLoader, val_loader : DataLoader, epochs : int, lr_scheduler=None,
          verbose=False) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Trains a given model with given routine: loss function, optimizer, learning rate scheduler,
    and a training set.

    Parameters:
    -----------
    model (torch.nn.Module):
        The model to be trained.

    criterion (torch.nn.Module):
        The loss function.

    optimizer (torch.optim):
        The optimizer.

    train_loader (torch.utils.data.DataLoader):
        A data loader that contains the training data.

    val_loader (torch.utils.data.DataLoader):
        A data loader the contains the validation data.

    lr_scheduler (torch.optim.lr_sechuler) Defaults to None:
        The learning rate scheduler.

    verbose (bool) Defaults to False:
        If true the function prints relevant information about the training process after the
        evaluation of each epoch.

    Returns:
    --------
        total_loss (list[flaot]):
            A list containing the loss values for each epoch.

        train_accuracies (list[float]):
            A list containing the accuracy of the model on the training set.

        val_accuracies (list[float]):
            A list containing the accuracy of the model on the validation set.

        lrs (list[float]):
            A list containing the learning rate at each epoch.

    """
    # Selecting the device depending on the machine running this code:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setting the model to the train mode:
    model.train()
    # Moving the model to the selected device:
    model = model.to(device)

    total_loss = []
    lrs = []

    train_accuracies = []
    val_accuracies = []

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0

        for x, y in train_loader:

            # Enabling autocast
            with torch.autocast(device_type=device.__str__(),
                                enabled=(False if device=='cpu' else True)):

                # Moving the images and their labels to the selected device:
                x = x.to(device)
                y = y.type(torch.LongTensor)
                y = y.to(device)

                # Resetting the gradients so they are calculated for each batch
                # independently:
                optimizer.zero_grad()

                # Generating a predication:
                y_hat = model(x).type(torch.float64)

                # Calculating the loss:
                loss = criterion(y_hat, y)

                # Accumulating the loss of the current epoch:
                epoch_loss += loss.item()

                # Backpropagation:
                loss.backward()

                # Advancing the optimzer:
                optimizer.step()


        # Advancing the scheduler if it is given:
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Accumulating the loss values for all the epochs in a list:
        total_loss.append(epoch_loss)

        if verbose:
            print('Evaluating epoch...')

        # --- Evaluating the epoch -----------------------------------

        # Evaluating the model at the current epoch on the train and validation sets:
        train_acc = evaluate(model, train_loader)
        val_acc = evaluate(model, val_loader)

        # Adding the accuracy of the train and validation sets to their corresponding lists:
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)

        if verbose:
            status = f'Epoch: {epoch} | Loss: {epoch_loss:.4f} | Train_acc: {100 * train_acc :.2f}% | ' \
f'Val_acc: {100 * val_acc :.2f}% | LR: {current_lr}'
            print(status)

    return total_loss, train_accuracies, val_accuracies, lrs


def predict(model, data_loader):
    """
    Generates predictions using a given model on a given dataset.

    Parameters:
    -----------
        model (torch.nn.Module):
            The model will generate the predictions.

        data_loader (torch.utils.data.DataLoader):
            The dataset that model will generate the predictions according to.

    Returns:
    --------
        predictions (numpy.ndarray):
            A numpy array containing the predictions of size 1 x n where n is the number
            of data points in the data_loader.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    precitions = torch.tensor([]).to(device)

    model.eval()
    model = model.to(device)

    for x, _ in data_loader:

        x = x.to(device)

        y_pred = model(x)
        _, y_pred = torch.max(y_pred, axis=1)

        precitions = torch.cat([precitions, y_pred])

    precitions = precitions.cpu().detach().numpy()
    return precitions


def load_data(batch_size : int, num_workers : int, train_transform : torchvision.transforms,
              val_transform : torchvision.transforms, val_proportion=0.6,
              test_proportion=0.4) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:

    train_dataset = dsets.FashionMNIST(root='/.data', train=True, download=True, transform=train_transform)
    validation_dataset = dsets.FashionMNIST(root='/.data', train=False, download=True, transform=val_transform)

    val_size = math.floor(len(validation_dataset) * val_proportion)
    test_size = math.ceil(len(validation_dataset) * test_proportion)

    validation_dataset, test_dataset = torch.utils.data.random_split(validation_dataset, [val_size, test_size])

    return train_dataset, validation_dataset, test_dataset