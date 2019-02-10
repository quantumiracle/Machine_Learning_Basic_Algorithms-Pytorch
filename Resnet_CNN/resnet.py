import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2D(nn.Module):
    
    def __init__(self, inchannel, outchannel, kernel_size, stride, padding, bias = True):
        
        super(Conv2D, self).__init__()
        
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weights = nn.Parameter(torch.Tensor(outchannel, inchannel, 
                                                 kernel_size, kernel_size))
        self.weights.data.normal_(-0.1, 0.1)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(outchannel, ))
            self.bias.data.normal_(-0.1, 0.1)
        else:
            self.bias = None
        # for test
        self.output = nn.Conv2d(in_channels=self.inchannel, out_channels=self.outchannel, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

            
        
    def forward(self, x):
        
        ##############################################################
        #                       YOUR CODE HERE                       #       
        ##############################################################
        ''' 1. code from scratch without unfold/vectorization '''
        # # padding
        # if self.padding > 0:
        #     x_p = torch.zeros(x.shape[0],x.shape[1], x.shape[-2]+2*self.padding, x.shape[-1]+2*self.padding, device=torch.device('cuda'))
        #     for k in range (x.shape[0]):
        #         for l in range(x.shape[1]):
        #             for i in range(self.padding, self.padding+x.shape[-2]):
        #                 for j in range(self.padding, self.padding+x.shape[-1]):
        #                     x_p[k][l][i][j]=x[k][l][i-self.padding][j-self.padding]
        # else:
        #     x_p=x
               
        # output_dim_x=(int)((x.shape[-2]-self.kernel_size+2*self.padding)/self.stride)+1
        # output_dim_y=(int)((x.shape[-1]-self.kernel_size+2*self.padding)/self.stride)+1
            
        # output = torch.zeros(x.shape[0], self.outchannel, output_dim_x, output_dim_y, device=torch.device('cuda'))
        # # sliding
        # for p in range(x.shape[0]):
        #     for i in range(self.outchannel):
        #         for j in range(output_dim_x):
        #             for k in range(output_dim_y):
        #                 x_p_ijk=x_p[p, :,  j*self.stride:j*self.stride+self.kernel_size,  k*self.stride:k*self.stride+self.kernel_size]
        #                 output[p][i][j][k]=(self.weights[i] * x_p_ijk).sum()
        #         if self.bias is not None:
        #             output[p][i]+=self.bias[i]  
        
        # output1=self.output(x)
        # for p in range(x.shape[0]):
        #     if self.bias is not None:
        #         output1[p]+=self.bias

          


        ''' 2. code with unfold/vectorization/matrix multiplication '''
        output_dim_x=(int)((x.shape[-2]-self.kernel_size+2*self.padding)/self.stride)+1
        output_dim_y=(int)((x.shape[-1]-self.kernel_size+2*self.padding)/self.stride)+1
        # 4 dim to 3 dim
        x_uf=F.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)

        output = torch.matmul(x_uf.transpose(1,2), self.weights.view(-1, self.weights.shape[0])).transpose(1,2).view(x.shape[0], self.outchannel, output_dim_x, output_dim_y) 
        if self.bias is not None: 
            output = output + self.bias
          
            


        ##############################################################
        #                       END OF YOUR CODE                     #
        ##############################################################
        

        return output
        
        



class MaxPool2D(nn.Module):
    
    def __init__(self, pooling_size):
        # assume pooling_size = kernel_size = stride
        
        super(MaxPool2D, self).__init__()
        
        self.pooling_size = pooling_size
        

    def forward(self, x):
        
        
        ##############################################################
        #                       YOUR CODE HERE                       #       
        ##############################################################

        # size after pooling
        itr_x=(int)(x.shape[-2]/self.pooling_size)
        itr_y=(int)(x.shape[-1]/self.pooling_size)


        ''' 1. code from scratch without unfold/vectorization '''
        # initialize, send to cuda
        # output=torch.zeros(x.shape[0],x.shape[1], itr_x, itr_y, device=torch.device('cuda'))
        # for p in range (x.shape[0]):
        #     for k in range (x.shape[1]):
        #         for i in range(itr_x):
        #             for j in range(itr_y):
        #                 # max of square
        #                 output[p][k][i][j]=torch.max(x[p, k, i*self.pooling_size:i*self.pooling_size+self.pooling_size, j*self.pooling_size:j*self.pooling_size+self.pooling_size])



        ''' 2. code with unfold '''
        x_uf=F.unfold(x, self.pooling_size, stride=self.pooling_size)
        x_uf=x_uf.view(x.shape[0], x.shape[1],-1, x_uf.shape[-1])
        output=torch.max(x_uf, dim=-2)[0].view(x.shape[0],x.shape[1], itr_x, itr_y)




        ##############################################################
        #                       END OF YOUR CODE                     #
        ##############################################################
                
        
        return output


# define resnet building blocks

class ResidualBlock(nn.Module): 
    def __init__(self, inchannel, outchannel, stride=1): 
        
        super(ResidualBlock, self).__init__() 
        
        self.left = nn.Sequential(Conv2D(inchannel, outchannel, kernel_size=3, 
                                         stride=stride, padding=1, bias=False), 
                                  nn.BatchNorm2d(outchannel), 
                                  nn.ReLU(inplace=True), 
                                  Conv2D(outchannel, outchannel, kernel_size=3, 
                                         stride=1, padding=1, bias=False), 
                                  nn.BatchNorm2d(outchannel)) 
        
        self.shortcut = nn.Sequential() 
        
        if stride != 1 or inchannel != outchannel: 
            
            self.shortcut = nn.Sequential(Conv2D(inchannel, outchannel, 
                                                 kernel_size=1, stride=stride, 
                                                 padding = 0, bias=False), 
                                          nn.BatchNorm2d(outchannel) ) 
            
    def forward(self, x): 
        
        out = self.left(x) 
        
        out += self.shortcut(x) 
        
        out = F.relu(out) 
        
        return out



# define resnet

class ResNet(nn.Module):
    
    def __init__(self, ResidualBlock, num_classes = 10):
        
        super(ResNet, self).__init__()
        
        self.inchannel = 64
        self.conv1 = nn.Sequential(Conv2D(3, 64, kernel_size = 3, stride = 1,
                                            padding = 1, bias = False), 
                                  nn.BatchNorm2d(64), 
                                  nn.ReLU())
        
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride = 1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride = 2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride = 2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride = 2)
        self.maxpool = MaxPool2D(4)
        self.fc = nn.Linear(512, num_classes)
        
    
    def make_layer(self, block, channels, num_blocks, stride):
        
        strides = [stride] + [1] * (num_blocks - 1)
        
        layers = []
        
        for stride in strides:
            
            layers.append(block(self.inchannel, channels, stride))
            
            self.inchannel = channels
            
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        
        x = self.conv1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.maxpool(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        
        return x
    
    
def ResNet18():
    return ResNet(ResidualBlock)







import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset

import numpy as np

import torchvision.transforms as T


transform = T.ToTensor()


# load data

NUM_TRAIN = 49000
print_every = 100


data_dir = './data'
cifar10_train = dset.CIFAR10(data_dir, train=True, download=True, transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=64, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

cifar10_val = dset.CIFAR10(data_dir, train=True, download=True, transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=64, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

cifar10_test = dset.CIFAR10(data_dir, train=False, download=True, transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=64)


USE_GPU = True
dtype = torch.float32 

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')








def check_accuracy(loader, model):
    # function for test accuracy on validation and test set
    
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


def train_part(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        print(len(loader_train))
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            loss.backward()

            # Update the parameters of the model using the gradients
            optimizer.step()

            if t % print_every == 0:
                print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                #check_accuracy(loader_val, model)
                print()




# code for optimising your network performance


# define and train the network
model = ResNet18()
optimizer = optim.Adam(model.parameters())

train_part(model, optimizer, epochs = 10)


# report test set accuracy

check_accuracy(loader_test, model)


# save the model
torch.save(model.state_dict(), 'model.pt')