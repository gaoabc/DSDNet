import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def dismat(self, X):
    X = torch.squeeze(X)
    r = torch.sum(X * X, 1)
    r = r.view([-1, 1])
    a = torch.matmul(X, torch.transpose(X, 0, 1))
    D = r.expand_as(a) - 2 * a + torch.transpose(r, 0, 1).expand_as(a)
    D = torch.abs(D)
    return D


def kernelmat(self, X, sigma):
    # 核函数
    Dxx = self.dismat(X)
    Kx = torch.exp(-Dxx / (2 * sigma * sigma)).type(torch.FloatTensor)
    # Kx = torch.exp(-Dxx / (2 * sigma * sigma))
    return Kx


def hsic(self, X, Y, sigma=2):
    m = int(X.size()[0])
    m2 = torch.tensor(X.size()[1] - 1) * torch.tensor(X.size()[1] - 1)
    H = torch.eye(m) - (1. / m) * torch.ones([m, m])
    Kx = torch.mm(self.kernelmat(X, sigma), H)
    Ky = torch.mm(self.kernelmat(Y, sigma), H)
    HSIC = torch.trace(torch.matmul(Kx, Ky)) / m2
    return HSIC


def hsicloss(self, X, Y, Z, sigma=2, lam=3):
    HSICLOSS = self.hsic(Y, X, sigma) - lam * self.hsic(Y, Z, sigma)
    return HSICLOSS

def HSIC_LOSS(x,y,z):
    loss = hsicloss(X=x, Y=y, Z=z)
    return loss

