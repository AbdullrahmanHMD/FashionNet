import torch
import torch.nn as nn


class FashionNet(nn.Module):
    def __init__(self, number_of_classes, in_channel=1, image_size=28):
        super(FashionNet, self).__init__()
        
        self.relu = nn.LeakyReLU(negative_slope=0.1)

        # --- First Layer --------------------------------------------------------------    
        kernel_size, stride, padding, = 3, 1, 1
        out_channels_1 = 24
        self.conv_1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channels_1, 
                                kernel_size=kernel_size, stride=stride, padding=padding)

        self.bn1 = nn.BatchNorm2d(num_features=out_channels_1)
        
        p = 0.5
        self.dropout_1 = nn.Dropout(p=p)
        new_image_size = get_image_size(image_size, kernel_size, padding, stride)

        # --- Second Layer -------------------------------------------------------------
        kernel_size, stride, padding, = 3, 2, 1
        out_channels_2 = 64
        self.conv_2 = nn.Conv2d(in_channels=out_channels_1, out_channels=out_channels_2, 
                                kernel_size=kernel_size, stride=stride, padding=padding)

        self.bn2 = nn.BatchNorm2d(num_features=out_channels_2)

        p = 0.5
        self.dropout_2 = nn.Dropout(p=p)

        # --- Fully Connected Layer ----------------------------------------------------
        in_features = get_image_size(new_image_size, kernel_size, padding, stride)**2 * out_channels_2
        self.fc = nn.Linear(in_features, number_of_classes)

        # --- Weight Initialization ----------------------------------------------------
        nn.init.kaiming_normal_(self.conv_1.weight)
        nn.init.kaiming_normal_(self.conv_2.weight)

        nn.init.constant_(self.conv_1.bias, 0.0)
        nn.init.constant_(self.conv_2.bias, 0.0)
        # ------------------------------------------------------------------------------

    def forward(self, x):

        # --- First Layer Forward ------------------
        x = self.relu(self.conv_1(x))
        x = self.dropout_1(self.bn1(x))

        # --- Second Layer Forward -----------------
        x = self.relu(self.conv_2(x))
        x = self.dropout_2(self.bn2(x))

        # --- Fully Connected Layer Forward --------
        # Flatten the image:
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def get_image_size(image_size, kernel_size, padding, stride):
    in_features = (image_size - kernel_size + 2 * padding) / stride + 1

    return int(in_features)