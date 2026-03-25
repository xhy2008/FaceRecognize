import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
class Facenet(nn.Module):
    def __init__(self):
        super(Facenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1,bias=False)#8,112,112
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0,bias=False)#16,110,110
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.AdaptiveMaxPool2d((55,55))
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1,bias=False)#32,55,55
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0,bias=False)#64,53,53
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.AdaptiveMaxPool2d((27,27))
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1,groups=64, padding=1,bias=False)#128,27,27
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=1,groups=128, padding=0,bias=False)#256,25,25
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.AdaptiveMaxPool2d((13,13))
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1,groups=256, padding=1,bias=False)#512,13,13
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 1024, kernel_size=3, stride=1,groups=512, padding=0,bias=False)#1024,11,11
        self.bn8 = nn.BatchNorm2d(1024)
        self.maxpool5 = nn.AdaptiveMaxPool2d((4,4))
        self.conv9 = nn.Conv2d(1024, 1024, kernel_size=4, stride=1,groups=1024, padding=0,bias=False)#2048,1,1
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.maxpool3(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.maxpool4(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.maxpool5(x)
        x = self.conv9(x)
        return x.view(-1, 1024)

class Logits(nn.Module):
    def __init__(self, in_features, out_features):
        super(Logits, self).__init__()
        self.linear=nn.Linear(in_features, out_features)
    def forward(self, features):
        return self.linear(features)

if __name__ == '__main__':
    writer = SummaryWriter('logs')
    model = Facenet()
    input = torch.randn(1, 3, 112, 112)
    output = model(input)
    print(output.shape)
    writer.add_graph(model, input)
    writer.close()
