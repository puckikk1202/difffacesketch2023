import numpy
import torch
import torch.nn as nn
from . import networks

class Combine_AE(nn.Module):
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

    def encode(self, x):
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

    def decode(self, eye1_l, eye2_l, nose_l, mouth_l, face_l):
        # print('eye1_xddd', eye1_l.shape)
        eye1_x = self.eye1_decoder(eye1_l)
        eye2_x = self.eye2_decoder(eye2_l)
        nose_x = self.nose_decoder(nose_l)
        mouth_x = self.mouth_decoder(mouth_l)
        face_x = self.face_decoder(face_l)

        return eye1_x, eye2_x, nose_x, mouth_x, face_x

    def forward(self, x):
        eye1_l, eye2_l, nose_l, mouth_l, face_l = self.encode(x)
        # print('eye1', eye1_l.shape)
        # print('eye2', eye2_l.shape)
        # print('nose', nose_l.shape)
        # print('mouth', mouth_l.shape)
        # print('face', face_l.shape)
        eye1_x, eye2_x, nose_x, mouth_x, face_x = self.decode(eye1_l, eye2_l, nose_l, mouth_l, face_l)
        # print('eye1', eye1_x.shape)
        # print('eye2', eye2_x.shape)
        # print('nose', nose_x.shape)
        # print('mouth', mouth_x.shape)
        # print('face', face_x.shape)
        return eye1_x, eye2_x, nose_x, mouth_x, face_x


class One_AE(nn.Module):
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
                                                      

    def encode(self, x):
        # eye1_x = x[:,:, 94:94+64, 54:54+64]
        # eye2_x = x[:,:, 94:94+64, 128:128+64]
        # nose_x = x[:,:, 116:116+96, 91:91+96]
        # mouth_x = x[:,:, 151:151+96, 85:85+96]
        face_x = x

        # eye1_x = x[:,:, 188:188+128, 108:108+128]
        # eye2_x = x[:,:, 188:188+128, 256:256+128]
        # nose_x = x[:,:, 232:232+192, 182:182+192]
        # mouth_x = x[:,:, 302:302+192, 170:170+192]
        # face_x = x

        # eye1_l = self.eye1_encoder(eye1_x)
        # # print(eye1_l.shape)
        # eye2_l = self.eye2_encoder(eye2_x)
        # # print(eye2_l.shape)
        # nose_l = self.nose_encoder(nose_x)
        # # print(nose_l.shape)
        # mouth_l = self.mouth_encoder(mouth_x)
        # print(mouth_l.shape)
        face_l = self.face_encoder(face_x)
        # print(face_l.shape)

        return face_l

    def decode(self, eye1_l, eye2_l, nose_l, mouth_l, face_l):
        # print('eye1_xddd', eye1_l.shape)
        eye1_x = self.eye1_decoder(eye1_l)
        eye2_x = self.eye2_decoder(eye2_l)
        nose_x = self.nose_decoder(nose_l)
        mouth_x = self.mouth_decoder(mouth_l)
        face_x = self.face_decoder(face_l)

        return eye1_x, eye2_x, nose_x, mouth_x, face_x

    def forward(self, x):
        
        face_l = self.encode(x)
        # print('eye1', eye1_l.shape)
        # print('eye2', eye2_l.shape)
        # print('nose', nose_l.shape)
        # print('mouth', mouth_l.shape)
        # print('face', face_l.shape)
        face_x = self.decode(face_l)
        # print('eye1', eye1_x.shape)
        # print('eye2', eye2_x.shape)
        # print('nose', nose_x.shape)
        # print('mouth', mouth_x.shape)
        # print('face', face_x.shape)
        return face_x

