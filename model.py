import torch
import torch.nn as nn


class FashionNet(nn.Module):
    def __init__(self, number_of_classes, in_channel=1, image_size=27):
        super(FashionNet, self).__init__()

        in_features_list = []

        # --- First Layer --------------------------------------------------------------    
        kernel_size, stride, padding, = 3, 1, 1
        out_channels_1 = 12
        self.conv_1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channels_1, 
                                kernel_size=kernel_size, stride=stride, padding=padding)

        self.bn1 = nn.BatchNorm2d(num_features=out_channels_1)
        self.relu_1 = nn.ReLU()

        p = 0.5
        self.dropout_1 = nn.Dropout(p=p)

        in_features_list.append({"imsize" : image_size, "ksize": kernel_size, "pad": padding, "stride": stride})
        # --- Second Layer -------------------------------------------------------------
        kernel_size, stride, padding, = 3, 2, 1
        out_channels_2 = 24
        self.conv_2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channels_2, 
                                kernel_size=kernel_size, stride=stride, padding=padding)

        p = 0.25
        self.dropout_2 = nn.Dropout(p=p)

        new_image_size = image_size(in_features_list)

        in_features_list.append({"imsize" : new_image_size, "ksize": kernel_size, "pad": padding, "stride": stride})
        # --- Fully Connected Layer ----------------------------------------------------

        in_features = image_size(in_features_list) * out_channels_1 * out_channels_2
        self.fc = nn.Linear(in_features, number_of_classes)

    def forward(self, x):

        # --- First Layer Forward ------------------
        x = self.relu(self.conv_1(x))
        x = self.dropout_1(self.bn1(x))

        # --- Second Layer Forward -----------------
        x = self.relu(self.conv_2(x))
        x = self.dropout_2(x)

        # --- Fully Connected Layer Forward --------
        x = self.fc(x)

        return x


def image_size(convs : list):
    in_features = 0

    for conv in convs:
        image_size = conv['imsize']
        kernel_size = conv['ksize']
        padding = conv['pad']
        stride = conv['stride']

        in_features += (image_size - kernel_size + 2 * padding) / stride + 1
    
    return in_features