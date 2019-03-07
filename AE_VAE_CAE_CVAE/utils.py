import numpy as np
import torch
import matplotlib.pyplot as plt


def interpolate(z1, z2, num=11):
    Z = np.zeros((z1.shape[0], num))
    for i in range(z1.shape[0]):
        Z[i, :] = np.linspace(z1[i], z2[i], num)
    return Z


def denorm_for_tanh(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

def denorm_for_sigmoid(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

def denorm_for_binary(x):
    x = x.clamp(0, 1)
    x = x>0.5
    x = x.view(x.size(0), 1, 28, 28)
    return x

def show(img):
    if torch.cuda.is_available():
        img = img.cpu()
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
