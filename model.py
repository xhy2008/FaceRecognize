import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.utils.tensorboard import SummaryWriter
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        #input:3*112*112
        self.conv1=nn.Conv2d(3,8,kernel_size=3,stride=1,padding=0,bias=False)#8*110*110
        self.bn1=nn.BatchNorm2d(8)
        self.relu1=nn.ReLU(inplace=True)
        self.convlayer1=nn.Sequential(
            nn.Conv2d(8,16,kernel_size=3,stride=1,padding=0,bias=False),#16*108*108
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,32,kernel_size=3,stride=1,padding=0,bias=False),#32*106*106 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=0,bias=False),#64*104*104
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1=nn.AdaptiveMaxPool2d((50,50))#64*50*50
        #深度可分离卷积
        self.convlayer2=nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=0,groups=64,bias=False),#128*48*48
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=0,groups=128,bias=False),#256*46*46
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,512,kernel_size=3,stride=1,padding=0,groups=256,bias=False),#512*44*44
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=0,groups=512,bias=False),#1024*42*42
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.pool2=nn.AdaptiveMaxPool2d((20,20))#1024*20*20
        #线性卷积
        self.convlayer3=nn.Sequential(
            nn.Conv2d(1024,512,kernel_size=3,stride=1,padding=0,bias=False),#512*18*18
            nn.BatchNorm2d(512),
            nn.Conv2d(512,256,kernel_size=3,stride=1,padding=0,bias=False),#256*16*16
            nn.BatchNorm2d(256),
            nn.Conv2d(256,128,kernel_size=3,stride=1,padding=0,bias=False),#128*14*14
            nn.BatchNorm2d(128),
            nn.Conv2d(128,64,kernel_size=3,stride=1,padding=0,bias=False),#64*14*14
            nn.BatchNorm2d(64),
        )
        self.pool3=nn.AdaptiveAvgPool2d((6,6))#64*6*6
        self.linear=nn.Conv2d(64,32,kernel_size=3,stride=1,padding=0,bias=False)#32*4*4
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.convlayer1(x)
        x=self.pool1(x)
        x=self.convlayer2(x)
        x=self.pool2(x)
        x=self.convlayer3(x)
        x=self.pool3(x)
        x=self.linear(x)
        x=x.view(x.size(0),-1)#batch*1024
        return x

class ArcFaceLoss(nn.Module):
    def __init__(self, in_features=1024, out_features=500, s=32.0, m=0.50, easy_margin=False):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        # init.kaiming_uniform_()
        # self.weight.data.normal_(std=0.001)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

if __name__=="__main__":
    writer=SummaryWriter("logs")
    net=ConvNet()
    criterion=ArcFaceLoss(512,500)
    input=torch.randn(8,3,112,112)
    label=torch.randint(0,500,(8,)) 
    output=net(input)
    print(output.shape)
    logits=criterion(output,label)
    loss=F.cross_entropy(logits,label)
    print(loss.item())
    writer.add_graph(net,input)
    writer.close()