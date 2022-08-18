from lib2to3.pgen2.token import GREATER
from random import shuffle
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from utils import Logger
from six.moves import urllib

proxy = urllib.request.ProxyHandler({'http': 'http://proxyseso.scania.com:8080'})
opener = urllib.request.build_opener(proxy)
urllib.request.install_opener(opener)



class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        n_features = 784
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features,1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.out = nn.Sequential(
            torch.nn.Linear(256,n_out),
            torch.nn.Sigmoid()
        )

    def forward(self,x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        n_features = 100
        n_out = 784

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features,256),
            nn.LeakyReLU(0.2)
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(256,512),
            nn.LeakyReLU(0.2)
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(512,1024),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Seqeuntial(
            nn.Linear(1024,n_out),
            nn.Tanh()
        )

    def forward(self,x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((.5,.5,.5),(.5,.5,.5))
        ])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir,train=True, transform=compose, download=True)

def images_to_vectors(images):
    return images.view(images.size(0),784)
def vectors_to_images(vectors):
    return vectors.view(vectors.size(0),1,28,28)

data = mnist_data()

data_loader = torch.utils.data.DataLoader(data,batch_size = 100, shuffle= True)
num_batches = len(data_loader)

disc = Discriminator()
gen = Generator()