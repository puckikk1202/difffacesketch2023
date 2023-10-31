import numpy
import torch
import torch.nn as nn
from . import networks
import math
import torch.nn.functional as F
import numpy as np

class Combine_AE_dilate(nn.Module):
    def __init__(self, img_size, img_channels):
        super().__init__()

        self.img_size = img_size
        self.img_channels = img_channels

        self.eye1_encoder = networks.define_part_encoder(model='eye', 
                                                         input_nc = self.img_channels, 
                                                         norm='instance', 
                                                         latent_dim = 512)
        self.eye2_encoder = networks.define_part_encoder(model='eye', 
                                                         input_nc = self.img_channels, 
                                                         norm='instance', 
                                                         latent_dim = 512)
        self.nose_encoder = networks.define_part_encoder(model='nose', 
                                                         input_nc = self.img_channels, 
                                                         norm='instance', 
                                                         latent_dim = 512)
        self.mouth_encoder = networks.define_part_encoder(model='mouth', 
                                                         input_nc = self.img_channels, 
                                                         norm='instance', 
                                                         latent_dim = 512)      
        self.face_encoder = networks.define_part_encoder(model='face', 
                                                         input_nc = self.img_channels, 
                                                         norm='instance', 
                                                         latent_dim = 512)                                                                                           

        self.eye1_decoder = networks.define_part_decoder(model='eye', 
                                                         output_nc = self.img_channels, 
                                                         norm='instance', 
                                                         latent_dim = 512) 
        self.eye2_decoder = networks.define_part_decoder(model='eye', 
                                                         output_nc = self.img_channels, 
                                                         norm='instance', 
                                                         latent_dim = 512)                                             
        self.nose_decoder = networks.define_part_decoder(model='nose', 
                                                         output_nc = self.img_channels, 
                                                         norm='instance', 
                                                         latent_dim = 512) 
        self.mouth_decoder = networks.define_part_decoder(model='mouth', 
                                                         output_nc = self.img_channels, 
                                                         norm='instance', 
                                                         latent_dim = 512) 
        self.face_decoder = networks.define_part_decoder(model='face', 
                                                         output_nc = self.img_channels, 
                                                         norm='instance', 
                                                         latent_dim = 512) 

          # ------ sketch dilate --------    
        self.max_dilate = 21
    
        self.edgeSmooth1 = MyDilateBlur()
        self.edgedilate = ConditionalDilate(self.max_dilate, gpu=True)

    def encode(self, x):
        # print('encoding')
        eye1_x = x[:,:, 94:94+64, 54:54+64]
        eye2_x = x[:,:, 94:94+64, 128:128+64]
        nose_x = x[:,:, 116:116+96, 91:91+96]
        mouth_x = x[:,:, 151:151+96, 85:85+96]
        face_x = x

        # eye1_x = x[:,:, 188:188+128, 108:108+128]
        # eye2_x = x[:,:, 188:188+128, 256:256+128]
        # nose_x = x[:,:, 232:232+192, 182:182+192]
        # mouth_x = x[:,:, 302:302+192, 170:170+192]
        # face_x = x

        # ------- AE --------
        eye1_l = self.eye1_encoder(eye1_x)
        # print(eye1_l.shape)
        eye2_l = self.eye2_encoder(eye2_x)
        # print(eye2_l.shape)
        nose_l = self.nose_encoder(nose_x)
        # print(nose_l.shape)
        mouth_l = self.mouth_encoder(mouth_x)
        # print(mouth_l.shape)
        face_l = self.face_encoder(face_x)
        # print(face_l.shape)
        return eye1_l, eye2_l, nose_l, mouth_l, face_l

        # --------- VAE -------
        # eye1 = self.eye1_encoder(eye1_x)
        # # print(eye1_l.shape)
        # eye2 = self.eye2_encoder(eye2_x)
        # # print(eye2_l.shape)
        # nose = self.nose_encoder(nose_x)
        # # print(nose_l.shape)
        # mouth = self.mouth_encoder(mouth_x)
        # # print(mouth_l.shape)
        # face = self.face_encoder(face_x)
        # # print(face_l.shape)

        # return eye1, eye2, nose, mouth, face

    # ------- AE --------
    def decode(self, eye1_l, eye2_l, nose_l, mouth_l, face_l):
        # print('eye1_xddd', eye1_l.shape)
        eye1_x = self.eye1_decoder(eye1_l)
        eye2_x = self.eye2_decoder(eye2_l)
        nose_x = self.nose_decoder(nose_l)
        mouth_x = self.mouth_decoder(mouth_l)
        face_x = self.face_decoder(face_l)

        return eye1_x, eye2_x, nose_x, mouth_x, face_x
    
    # --------- VAE -------
    # def decode(self, eye1, eye2, nose, mouth, face):
    #     # print('eye1_xddd', eye1_l.shape)
    #     # print('decoding')
    #     eye1_x = self.eye1_decoder(eye1[2])
    #     eye2_x = self.eye2_decoder(eye2[2])
    #     nose_x = self.nose_decoder(nose[2])
    #     mouth_x = self.mouth_decoder(mouth[2])
    #     face_x = self.face_decoder(face[2])

    #     return eye1_x, eye2_x, nose_x, mouth_x, face_x

    def forward(self, x):
        # ------sketch dilate ----
        # print('origin',torch.min(x), torch.max(x))
        x = self.edgeSmooth1(x)
        # print('smooth',torch.min(x), torch.max(x))
        x = self.edgedilate(x, 1)
        # print('edged',torch.min(x), torch.max(x))

        dilate_x = x

        eye1, eye2, nose, mouth, face = self.encode(x)
        # print('eye1', len(eye1))
        # print('eye2', eye2_l.shape)
        # print('nose', nose_l.shape)
        # print('mouth', mouth_l.shape)
        # print('face', face_l.shape)

        # --------- VAE -------
        # eye1_kld_loss = torch.mean(-0.5 * torch.sum(1 + eye1[1] - eye1[0] ** 2 - eye1[1].exp(), dim = 1), dim = 0)
        # eye2_kld_loss = torch.mean(-0.5 * torch.sum(1 + eye2[1] - eye2[0] ** 2 - eye2[1].exp(), dim = 1), dim = 0)
        # nose_kld_loss = torch.mean(-0.5 * torch.sum(1 + nose[1] - nose[0] ** 2 - nose[1].exp(), dim = 1), dim = 0)
        # mouth_kld_loss = torch.mean(-0.5 * torch.sum(1 + mouth[1] - mouth[0] ** 2 - mouth[1].exp(), dim = 1), dim = 0)
        # face_kld_loss = torch.mean(-0.5 * torch.sum(1 + face[1] - face[0] ** 2 - face[1].exp(), dim = 1), dim = 0)

        eye1_x, eye2_x, nose_x, mouth_x, face_x = self.decode(eye1, eye2, nose, mouth, face)
        # print('eye1', eye1_x.shape)
        # print('eye2', eye2_x.shape)
        # print('nose', nose_x.shape)
        # print('mouth', mouth_x.shape)
        # print('face', face_x.shape)
        # return [eye1_x, eye1_kld_loss], [eye2_x, eye2_kld_loss], [nose_x, nose_kld_loss], [mouth_x, mouth_kld_loss], [face_x, face_kld_loss], dilate_x
        return eye1_x, eye2_x, nose_x, mouth_x, face_x, dilate_x

# ------- borrow from https://github.com/VITA-Group/DeepPS --------
class MyDilateBlur(nn.Module):
    def __init__(self, kernel_size=7, channels=1, sigma=0.8):
        super(MyDilateBlur, self).__init__()
        self.kernel_size=kernel_size
        self.channels = channels
        # Set these to whatever you want for your gaussian filter
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(self.kernel_size+0.)
        x_grid = x_cord.repeat(self.kernel_size).view(self.kernel_size, self.kernel_size)
        y_grid = x_grid.t()
        self.xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        self.mean = (self.kernel_size - 1)//2
        self.diff = -torch.sum((self.xy_grid - self.mean)**2., dim=-1)
        self.gaussian_filter = nn.Conv2d(in_channels=self.channels, out_channels=self.channels,
                                    kernel_size=self.kernel_size, groups=self.channels, bias=False)

        self.gaussian_filter.weight.requires_grad = False
        variance = sigma**2.
        gaussian_kernel = (1./(2.*math.pi*variance)) * torch.exp(self.diff /(2*variance))
        # note that normal gaussain_kernel use gaussian_kernel / torch.sum(gaussian_kernel)
        # here we multiply with 2 to make a small dilation
        gaussian_kernel = 2 * gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, self.kernel_size, self.kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(self.channels, 1, 1, 1)
        self.gaussian_filter.weight.data = gaussian_kernel
        
    def forward(self, x):        
        y = self.gaussian_filter(F.pad(1-x, (self.mean,self.mean,self.mean,self.mean), "replicate")) 
        # print('y', torch.min(y), torch.max(y))
        return 1 - 2 * torch.clamp(y, min=0, max=1)

class OneDilate(nn.Module):
    def __init__(self, kernel_size=10, channels=1, gpu=True):
        super(OneDilate, self).__init__()
        self.kernel_size=kernel_size
        self.channels = channels
        gaussian_kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(self.channels, 1, 1, 1)
        self.mean = (self.kernel_size - 1)//2
        self.gaussian_filter = nn.Conv2d(in_channels=self.channels, out_channels=self.channels,
                                    kernel_size=self.kernel_size, groups=self.channels, bias=False)
        if gpu:
            gaussian_kernel = gaussian_kernel.cuda()
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False
        
    def forward(self, x):
        x = F.pad((1-x)*0.5, (self.mean,self.mean,self.mean,self.mean), "replicate")
        return self.gaussian_filter(x)  

class ConditionalDilate(nn.Module):
    def __init__(self, max_kernel_size=21, channels=1, gpu=True):
        super(ConditionalDilate, self).__init__()
        
        self.max_kernel_size = max_kernel_size//2*2+1
        self.netBs = [OneDilate(i, gpu=gpu) for i in range(1,self.max_kernel_size+1,2)]
        
    def forward(self, x, l):
        l = min(self.max_kernel_size, max(1, l))
        lf = int(np.floor(l))
        if l == lf and l%2 == 1:
            out = self.netBs[(lf-1)//2](x)
        else:
            lf = lf - (lf+1)%2
            lc = lf + 2
            x1 = self.netBs[(lf-1)//2](x)
            x2 = self.netBs[(lc-1)//2](x)
            out = (x1 * (lc-l) + x2 * (l-lf))/2.0
        return 1 - 2 * torch.clamp(out, min=0, max=1) 