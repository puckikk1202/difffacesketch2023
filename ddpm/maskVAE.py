import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, nc, ngf, ndf, latent_variable_size):
        super(VAE, self).__init__()
        #self.cuda = True
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(nc, ndf, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.e3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.e4 = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf*8)

        self.e5 = nn.Conv2d(ndf*8, ndf*16, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(ndf*16)

        self.e6 = nn.Conv2d(ndf*16, ndf*32, 4, 2, 1)
        self.bn6 = nn.BatchNorm2d(ndf*32)

        # self.e7 = nn.Conv2d(ndf*32, ndf*64, 3, 2, 1)
        # self.bn7 = nn.BatchNorm2d(ndf*64)

        self.fc1 = nn.Linear(ndf*32*4*4, latent_variable_size)
        # self.fc2 = nn.Linear(ndf*64*4*4, latent_variable_size)

        # decoder
        # self.d1 = nn.Linear(latent_variable_size, ngf*64*4*4)

        # self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd1 = nn.ReplicationPad2d(1)
        # self.d2 = nn.Conv2d(ngf*64, ngf*32, 3, 1)
        # self.bn8 = nn.BatchNorm2d(ngf*32, 1.e-3)

        # self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd2 = nn.ReplicationPad2d(1)
        # self.d3 = nn.Conv2d(ngf*32, ngf*16, 3, 1)
        # self.bn9 = nn.BatchNorm2d(ngf*16, 1.e-3)

        # self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd3 = nn.ReplicationPad2d(1)
        # self.d4 = nn.Conv2d(ngf*16, ngf*8, 3, 1)
        # self.bn10 = nn.BatchNorm2d(ngf*8, 1.e-3)

        # self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd4 = nn.ReplicationPad2d(1)
        # self.d5 = nn.Conv2d(ngf*8, ngf*4, 3, 1)
        # self.bn11 = nn.BatchNorm2d(ngf*4, 1.e-3)

        # self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd5 = nn.ReplicationPad2d(1)
        # self.d6 = nn.Conv2d(ngf*4, ngf*2, 3, 1)
        # self.bn12 = nn.BatchNorm2d(ngf*2, 1.e-3)

        # self.up6 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd6 = nn.ReplicationPad2d(1)
        # self.d7 = nn.Conv2d(ngf*2, ngf, 3, 1)
        # self.bn13 = nn.BatchNorm2d(ngf, 1.e-3)

        # self.up7 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd7 = nn.ReplicationPad2d(1)
        # self.d8 = nn.Conv2d(ngf, nc, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        # self.relu = nn.ReLU()
        # #self.sigmoid = nn.Sigmoid()
        # self.maxpool = nn.MaxPool2d((2, 2), (2, 2))

    # def encode(self, x):
    def forward(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h6 = self.leakyrelu(self.bn6(self.e6(h5)))
        h6 = h6.view(-1, self.ndf*32*4*4) 
        # h6 = h6.view(128, -1) 
        # h7 = self.leakyrelu(self.bn7(self.e7(h6)))
        # h7 = h7.view(-1, self.ndf*64*4*4)

        hk = h6
        for i in range(16):
            hk = torch.mul(h6, hk)
            if i == 0:
                output = hk
            else:
                output = torch.cat((output, hk), 0)

        return self.fc1(output) #, self.fc2(h7)

    # def reparametrize(self, mu, logvar):
    #     std = logvar.mul(0.5).exp_()
    #     #if self.cuda:
    #     eps = torch.cuda.FloatTensor(std.size()).normal_()
    #     #else:
    #     #    eps = torch.FloatTensor(std.size()).normal_()
    #     eps = Variable(eps)
    #     return eps.mul(std).add_(mu)

    # def decode(self, z):
    #     h1 = self.relu(self.d1(z))
    #     h1 = h1.view(-1, self.ngf*64, 4, 4)
    #     h2 = self.leakyrelu(self.bn8(self.d2(self.pd1(self.up1(h1)))))
    #     h3 = self.leakyrelu(self.bn9(self.d3(self.pd2(self.up2(h2)))))
    #     h4 = self.leakyrelu(self.bn10(self.d4(self.pd3(self.up3(h3)))))
    #     h5 = self.leakyrelu(self.bn11(self.d5(self.pd4(self.up4(h4)))))
    #     h6 = self.leakyrelu(self.bn12(self.d6(self.pd5(self.up5(h5)))))
    #     h7 = self.leakyrelu(self.bn13(self.d7(self.pd6(self.up6(h6)))))
    #     return self.d8(self.pd7(self.up7(h7)))

    # def get_latent_var(self, x):
    #     mu, logvar = self.encode(x)
    #     z = self.reparametrize(mu, logvar)
    #     return z, mu, logvar.mul(0.5).exp_()

    # def forward(self, x):
        # mu, logvar = self.encode(x)
        # z = self.reparametrize(mu, logvar)
        # res = self.decode(z)
        
        # return res, x, mu, logvar