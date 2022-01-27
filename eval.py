import torch
import os

# The path for the saved parameters.
default_path = os.getcwd()
default_path = os.path.join(default_path, 'parameters')

def evaluate(model, validation_loader, epochs):
    model.eval()
    total_accuracy = []
    for epoch in range(epochs):
        model.load_state_dict(get_model_state_dict(f'param_epoch_{epoch}'))
        correct = 0

        for x, y in validation_loader:
            yhat = model(x)

            _, label = torch.max(yhat, 1)
            correct += (y == label).sum().item()

        accuracy = 100 * correct / (len(validation_loader) * validation_loader.batch_size)
        total_accuracy.append(accuracy)
        print(f'Accuracy: {accuracy:.2f}%\t |\tEpoch: {epoch}')
    return total_accuracy


def get_model_state_dict(param_name, path=default_path):
    path = os.path.join(path, param_name)
    with open(path, 'rb') as file:
        model_state_dict = torch.load(file)['model_state_dict']
    return model_state_dict