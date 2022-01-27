import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader


def load_data(train_batch_size, validation_batch_size, image_size = 28):

    composed = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])

    train_dataset = dsets.FashionMNIST(root='/.data', train=True, download=True, transform=composed)
    validation_dataset = dsets.FashionMNIST(root='/.data', train=False, download=True, transform=composed)

    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=validation_batch_size)

    return train_loader, validation_loader