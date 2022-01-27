import os
import torch

# The path for the saved parameters.
default_path = os.getcwd()
default_path = os.path.join(default_path, 'parameters')

def train(model, train_loader, optimizer, criterion, epochs):

    total_loss = []

    for epoch in range(epochs):
        batch_loss = 0
        for x, y in train_loader:

            optimizer.zero_grads()
            yhat = model(x)

            loss = criterion(yhat, y)
            batch_loss += loss.item()   

            loss.backward()
            optimizer.step()

        total_loss.append(batch_loss)
        export_parameters(model, f'param_epoch_{epoch}')

    return total_loss


def export_parameters(model, param_name, path=default_path):
    path = os.join(path, param_name)
    with open(path, 'wb') as file:
        torch.save({'model_state_dict': model.state_dict()}, file)
