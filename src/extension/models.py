import torch.nn as nn 
import torch.nn.functional as F 

class SurfNet(nn.Module):
    def __init__(self, intermediate_layer_size):
        super(SurfNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=intermediate_layer_size, kernel_size=3, stride=1, padding=1)
        # self.dropout1 = nn.Dropout2d(0.25)
        self.conv3 = nn.Conv2d(in_channels=intermediate_layer_size, out_channels=1, kernel_size=1, stride=1)        



    # x represents our data
    def forward(self, x):

        y = self.conv1(x)
        y = F.relu(y)

        y = self.conv2(y)
        y = F.relu(y)

        # x = self.dropout1(x)

        y =  self.conv3(y)

        return y

