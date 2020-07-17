import torch
from torch import nn

'''
Basic neural network using pytorch
'''

image = torch.randn(3,10,20)
d0 = image.nelement()

class mynet(nn.Module):
    def __init__(self, d0,d1,d2,d3):
        super().__init__()
        self.m0 = nn.Linear(d0,d1)
        self.m1 = nn.Linear(d1,d2)
        self.m2 = nn.Linear(d2,d3)
    def forward(self,x):
        z0 = x.view(-1) ## flattten input tensor
        s1 = self.m0(z0)
        z1 = torch.relu(s1)
        s2 = self.m1(z1)
        z2 = torch.relu(s2)
        s3 = self.m2(z2)
        return s3

model = mynet(d0,60,40,10)
out = model(image)
