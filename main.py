from lib2to3.pgen2.token import GREATER
from random import shuffle
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from utils import Logger

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

        self.out = nn.Sequential(
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
        transforms.Normalize((.5,),(.5,))
        ])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir,train=True, transform=compose, download=True)

def images_to_vectors(images):
    return images.view(images.size(0),784)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0),1,28,28)

def ones_target(size):
    return Variable(torch.ones(size,1))

def zeros_target(size):
    return Variable(torch.zeros(size,1))

def noise(size):
    return Variable(torch.randn(size,100))

def train_disc(optimizer,real_data,gen_data):
    N = real_data.size(0)
    optimizer.zero_grad()

    # train on real data
    prediction_real = disc(real_data)
    #calculate error and backpropagate
    error_real = loss_func(prediction_real,ones_target(N))
    error_real.backward()

    #train on generated data
    prediction_gen = disc(gen_data)
    #calculate error and backpropagate
    error_gen = loss_func(prediction_gen,zeros_target(N))
    error_gen.backward()

    #update weithgs with gradients
    optimizer.step()

    return error_real+ error_gen , prediction_real, prediction_gen

def train_gen(optimizer,gen_data):
    N = gen_data.size(0)
    optimizer.zero_grad()

    # generate fake data from sampling noise and get prediction
    prediction = disc(gen_data)

    #calculate error and backpropagate
    error = loss_func(prediction,ones_target(N))
    error.backward()
    #update weights with gradients
    optimizer.step()

    return error

learning_rate = 0.0002
test_samples_num = 16
epochs_num = 200

data = mnist_data()
data_loader = torch.utils.data.DataLoader(data,batch_size = 100, shuffle= True)
num_batches = len(data_loader)

# initializing discrimiator and generator neural networks
disc = Discriminator()
gen = Generator()

# initializing optimizers [Adam]
disc_optimizer = optim.Adam(disc.parameters(),lr=learning_rate)
gen_optimizer = optim.Adam(gen.parameters(),lr=learning_rate)

# initialize loss function [binary cross entropy]
loss_func = nn.BCELoss()

test_noise = noise(test_samples_num)
logger = Logger(model_name='VGAN',data_name='MNIST')

for epoch in range(epochs_num):
    for n_batch,(real_batch,_) in enumerate(data_loader):
        N = real_batch.size(0)

        # get real data and transform to vectors
        real_data = Variable(images_to_vectors(real_batch))

        # generate fake data and detach <- so gardients are not calculated for generator
        gen_data = gen(noise(N)).detach()

        # train discriminator
        disc_error, disc_pred_real, disc_pred_gen = train_disc(disc_optimizer,real_data,gen_data)

        # generate fake data
        gen_data = gen(noise(N))

        # train generator
        gen_error = train_gen(gen_optimizer,gen_data)

        # log bach error
        logger.log(disc_error,gen_error,epoch,n_batch,num_batches)

        if (n_batch) % 100 == 0:
            test_images = vectors_to_images(gen(test_noise))
            test_images = test_images.data

            logger.log_images(
                test_images, test_samples_num,
                epoch, n_batch, num_batches
            )

            logger.display_status(
                epoch, epochs_num, n_batch, num_batches,
                disc_error, gen_error, disc_pred_real, disc_pred_gen
            )