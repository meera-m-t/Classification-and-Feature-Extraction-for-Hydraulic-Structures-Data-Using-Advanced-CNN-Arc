import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),nn.BatchNorm2d(32),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5),nn.PReLU(), nn.BatchNorm2d(64),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(64, 64, 5), nn.PReLU(),nn.BatchNorm2d(64),
                                     nn.MaxPool2d(2, stride=2)
                                    
                                    )

#         self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
#                                      nn.MaxPool2d(2, stride=2),
#                                      nn.Conv2d(32, 64, 5),nn.PReLU(), 
#                                      nn.MaxPool2d(2, stride=2),
#                                     )
        self.fc = nn.Sequential(nn.Linear(5184  , 256),  #   12800  10368 6272 1152 # 30976 28224 23104 14400 #5184 4096 2304
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),                                
                                nn.Linear(256,2)
                                )

    def forward(self, x):
        #print(x.shape)
        output = self.convnet(x)
        #print(output.shape)
        output = output.view(output.size()[0], -1)
        #print(output.shape)
        output = self.fc(output)
        #print(output.shape)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        #print(output1.shape,"&&&&&&&&&&&&&&&&&&&&&&&&&&")
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
