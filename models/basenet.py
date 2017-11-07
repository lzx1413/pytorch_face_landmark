'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei
'''
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



class BaseNet(nn.Module):

    def __init__(self, num_classes=152):
        super(BaseNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(x,inplace=True)
        x = self.classifier(x)
        return x

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

class MobileNet(nn.Module):

    def __init__(self,num_classes = 152):
        super(MobileNet,self).__init__()
        self.features = nn.Sequential(
            conv_bn(3,16,2),
            conv_dw(16,32,2),
            conv_dw(32,64,2),
            conv_dw(64,128,2),
        )
        self.classifier = nn.Linear(256*128,num_classes)

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(x,inplace=False)
        x = self.classifier(x)
        return x


class SqueezeNet(nn.Module):
    def __init__(self,num_classes):
        super(SqueezeNet,self).__init__()
        self.pretrain_net = models.squeezenet1_1(pretrained=True)
        self.base_net = self.pretrain_net.features
        self.pooling  = nn.AvgPool2d(3)
        self.fc = nn.Linear(512,num_classes)
    def forward(self,x):
        x = self.base_net(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x








