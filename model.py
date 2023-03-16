import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, 2)
        self.conv3 = nn.Conv2d(128, 256, 3, 2)
        self.conv4 = nn.Conv2d(256, 512, 3, 2)
        self.fc1 = nn.Linear(512*5*5, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.fc1(x)

        return self.A_Softmax(x)
    
    def A_Softmax(input: Tensor) -> Tensor:
        """
        ---Implimentation not Complete---

        Parameters
        ----------
        input : Tensor
            The input Tensor

        Returns
        -------
        Tensor
            A Tensor of same size to the input Tensor

        """
        means = torch.mean(input, 1, keepdim=True)[0]
        a = torch.exp(input-means)
        a_sum = torch.sum(a, 1, keepdim=True)
    
        retu = -torch.log(a/a_sum)/Tensor.size(input, dim=1)
        
        return retu