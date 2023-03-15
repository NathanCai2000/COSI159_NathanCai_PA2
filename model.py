import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, 2)
        self.conv3 = nn.Conv2d(128, 256, 3, 2)
        self.conv4 = nn.Conv2d(256, 512, 3, 2)
        self.conv4_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(14, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        #x = x.view(-1, 320)
        #x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)

        return F.log_softmax(x)
