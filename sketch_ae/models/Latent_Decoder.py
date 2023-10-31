import numpy
import torch
import torch.nn as nn
from . import networks

class Latent_Decoder(nn.Module):
    def __init__(self, latent_size, latent_channels):
        super().__init__()

        self.latent_size = latent_size
        self.latent_channels = latent_channels                                                                                          

        self.eye1_decoder = networks.define_part_train_decoder(model='eye', 
                                                         output_nc = self.latent_channels, 
                                                         norm='instance', 
                                                         latent_dim = 512) 
        self.eye2_decoder = networks.define_part_train_decoder(model='eye', 
                                                         output_nc = self.latent_channels, 
                                                         norm='instance', 
                                                         latent_dim = 512)                                             
        self.nose_decoder = networks.define_part_train_decoder(model='nose', 
                                                         output_nc = self.latent_channels, 
                                                         norm='instance', 
                                                         latent_dim = 512) 
        self.mouth_decoder = networks.define_part_train_decoder(model='mouth', 
                                                         output_nc = self.latent_channels, 
                                                         norm='instance', 
                                                         latent_dim = 512) 
        self.face_decoder = networks.define_part_train_decoder(model='face', 
                                                         output_nc = self.latent_channels, 
                                                         norm='instance', 
                                                         latent_dim = 512) 

    # def encode(self, x):
    #     eye1_x = x[:,:, 94:94+64, 54:54+64]
    #     eye2_x = x[:,:, 94:94+64, 128:128+64]
    #     nose_x = x[:,:, 116:116+96, 91:91+96]
    #     mouth_x = x[:,:, 151:151+96, 85:85+96]
    #     face_x = x

    #     eye1_l = self.eye1_encoder(eye1_x)
    #     eye2_l = self.eye2_encoder(eye2_x)
    #     nose_l = self.nose_encoder(nose_x)
    #     mouth_l = self.mouth_encoder(mouth_x)
    #     face_l = self.face_encoder(face_x)

    #     return eye1_l, eye2_l, nose_l, mouth_l, face_l

    # def decode(self, eye1_l, eye2_l, nose_l, mouth_l, face_l):
    #     eye1_x = self.eye1_decoder(eye1_l)
    #     eye2_x = self.eye2_decoder(eye2_l)
    #     nose_x = self.nose_decoder(nose_l)
    #     mouth_x = self.mouth_decoder(mouth_l)
    #     face_x = self.face_decoder(face_l)

    #     return eye1_x, eye2_x, nose_x, mouth_x, face_x

    def decode(self, x):
        eye1_l, eye2_l, nose_l, mouth_l, face_l = x
        # print('eye1', eye1_l.shape)

        eye1_x = self.eye1_decoder(eye1_l)
        eye2_x = self.eye2_decoder(eye2_l)
        nose_x = self.nose_decoder(nose_l)
        mouth_x = self.mouth_decoder(mouth_l)
        face_x = self.face_decoder(face_l)
        return eye1_x, eye2_x, nose_x, mouth_x, face_x

    def forward(self, x):

        eye1_x, eye2_x, nose_x, mouth_x, face_x = self.decode(x)
        # print('eye1', eye1_x.shape)
        # print('eye2', eye2_x.shape)
        # print('nose', nose_x.shape)
        # print('mouth', mouth_x.shape)
        # print('face', face_x.shape)

        # testing
        output_x = face_x
        output_x[:,:, 37:37+24, 21:21+24] = mouth_x
        output_x[:,:, 29:29+24, 22:22+24] = nose_x
        output_x[:,:, 23:23+16, 13:13+16] = eye1_x
        output_x[:,:, 23:23+16, 32:32+16] = eye2_x

        # output_x = face_x
        # output_x[:,:, 74:74+48, 42:42+48] = mouth_x
        # output_x[:,:, 58:58+48, 44:44+48] = nose_x
        # output_x[:,:, 46:46+32, 26:26+32] = eye1_x
        # output_x[:,:, 46:46+32, 64:64+32] = eye2_x

        return output_x

class One_Latent_Decoder(nn.Module):
    def __init__(self, latent_size, latent_channels):
        super().__init__()

        self.latent_size = latent_size
        self.latent_channels = latent_channels                                                                                          

        self.face_decoder = networks.define_part_train_decoder(model='face', 
                                                         output_nc = self.latent_channels, 
                                                         norm='instance', 
                                                         latent_dim = 512) 


    def decode(self, x):
        eye1_l, eye2_l, nose_l, mouth_l, face_l = x
        # print('eye1', eye1_l.shape)


        face_x = self.face_decoder(face_l)
        return face_x

    def forward(self, x):

        face_x = self.decode(x)
        # print('eye1', eye1_x.shape)
        # print('eye2', eye2_x.shape)
        # print('nose', nose_x.shape)
        # print('mouth', mouth_x.shape)
        # print('face', face_x.shape)

        # testing
        output_x = face_x

        # output_x = face_x
        # output_x[:,:, 74:74+48, 42:42+48] = mouth_x
        # output_x[:,:, 58:58+48, 44:44+48] = nose_x
        # output_x[:,:, 46:46+32, 26:26+32] = eye1_x
        # output_x[:,:, 46:46+32, 64:64+32] = eye2_x

        return output_x
