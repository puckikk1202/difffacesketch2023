import torch 
import torch.nn as nn
import torch.nn.functional as F

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def get_norm_layer(norm_type='instance'):
    if (norm_type == 'batch'):
        norm_layer = nn.BatchNorm2d
    elif (norm_type == 'instance'):
        norm_layer = nn.InstanceNorm2d
    else:
        raise NotImplementedError(('normalization layer [%s] is not found' % norm_type))
    return norm_layer

def define_part_encoder(model='mouth', norm='instance', input_nc=1, latent_dim=512):
    norm_layer = get_norm_layer(norm_type=norm)
    image_size = 256
    if 'eye' in model:
        image_size = 64
    elif 'mouth' in model:
        image_size = 96
    elif 'nose' in model:
        image_size = 96
    elif 'face' in model:
        image_size = 256
    # image_size = 512
    # if 'eye' in model:
    #     image_size = 128
    # elif 'mouth' in model:
    #     image_size = 192
    # elif 'nose' in model:
    #     image_size = 192
    # elif 'face' in model:
    #     image_size = 512
    
    else:
        print("Whole Image !!")

    net_encoder = EncoderGenerator_Res(norm_layer,image_size,input_nc, latent_dim)  # input longsize 256 to 512*4*4    
    print("net_encoder of part "+model+" is:",image_size)

    return net_encoder

def define_part_decoder(model='mouth', norm='instance', output_nc=1, latent_dim=512):
    norm_layer = get_norm_layer(norm_type=norm)

    image_size = 256
    if 'eye' in model:
        image_size = 64
    elif 'mouth' in model:
        image_size = 96
    elif 'nose' in model:
        image_size = 96
    elif 'face' in model:
        image_size = 256
    # image_size = 512
    # if 'eye' in model:
    #     image_size = 128
    # elif 'mouth' in model:
    #     image_size = 192
    # elif 'nose' in model:
    #     image_size = 192
    # elif 'face' in model:
    #     image_size = 512
    else:
        print("Whole Image !!")

    net_decoder = DecoderGenerator_image_Res(norm_layer,image_size,output_nc, latent_dim)  # input longsize 256 to 512*4*4

    print("net_decoder to image of part "+model+" is:",image_size)

    return net_decoder

def define_part_train_decoder(model='mouth', norm='instance', output_nc=8, latent_dim=512):
    norm_layer = get_norm_layer(norm_type=norm)

    image_size = 64
    if 'eye' in model:
        image_size = 16
    elif 'mouth' in model:
        image_size = 24
    elif 'nose' in model:
        image_size = 24
    elif 'face' in model:
        image_size = 64
    # image_size = 128
    # if 'eye' in model:
    #     image_size = 32
    # elif 'mouth' in model:
    #     image_size = 48
    # elif 'nose' in model:
    #     image_size = 48
    # elif 'face' in model:
    #     image_size = 128
    else:
        print("Whole Image !!")

    net_decoder = DecoderGenerator_sketch_Res(norm_layer,image_size,output_nc, latent_dim)  # input longsize 256 to 512*4*4

    print("net_decoder to image of part "+model+" is:",image_size)

    return net_decoder


# decoder block (used in the decoder)
class DecoderBlock(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=4, padding=1, stride=2, output_padding=0, norelu=False):
        super(DecoderBlock, self).__init__()
        layers_list = []
        layers_list.append(nn.ConvTranspose2d(channel_in, channel_out, kernel_size, padding=padding, stride=stride, output_padding=output_padding))
        layers_list.append(nn.BatchNorm2d(channel_out, momentum=0.9))
        if (norelu == False):
            layers_list.append(nn.LeakyReLU(1))
        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        ten = self.conv(ten)
        return ten

# encoder block (used in encoder and discriminator)
class EncoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=7, padding=3, stride=4):
        super(EncoderBlock, self).__init__()
        # convolution to halve the dimensions
        self.conv = nn.Conv2d(channel_in, channel_out, kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)
        self.relu = nn.LeakyReLU(1)

    def forward(self, ten, out=False, t=False):
        # print('ten',ten.shape)
        # here we want to be able to take an intermediate output for reconstruction error
        if out:
            ten = self.conv(ten)
            ten_out = ten
            ten = self.bn(ten)
            ten = self.relu(ten)
            return (ten, ten_out)
        else:
            ten = self.conv(ten)
            ten = self.bn(ten)
            # print(ten.shape)
            ten = self.relu(ten)
            return ten

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if (padding_type == 'reflect'):
            conv_block += [nn.ReflectionPad2d(1)]
        elif (padding_type == 'replicate'):
            conv_block += [nn.ReplicationPad2d(1)]
        elif (padding_type == 'zero'):
            p = 1
        else:
            raise NotImplementedError(('padding [%s] is not implemented' % padding_type))
        conv_block += [nn.Conv2d(dim, dim, 3, padding=p), norm_layer(dim), activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if (padding_type == 'reflect'):
            conv_block += [nn.ReflectionPad2d(1)]
        elif (padding_type == 'replicate'):
            conv_block += [nn.ReplicationPad2d(1)]
        elif (padding_type == 'zero'):
            p = 1
        else:
            raise NotImplementedError(('padding [%s] is not implemented' % padding_type))
        conv_block += [nn.Conv2d(dim, dim, 3, padding=p), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        # print(x.shape)
        out = (x + self.conv_block(x))
        return out

class EncoderGenerator_Res(nn.Module):
    """docstring for  EncoderGenerator"""
    def __init__(self, norm_layer, image_size, input_nc, latent_dim=512):
        super( EncoderGenerator_Res, self).__init__()
        layers_list = []
        
        latent_size = int(image_size/32)
        longsize = 512*latent_size*latent_size
        self.longsize = longsize
        # print(image_size, latent_size, longsize)

        activation = nn.ReLU()
        padding_type='reflect'
        norm_layer=nn.BatchNorm2d

        # encode
        layers_list.append(EncoderBlock(channel_in=input_nc, channel_out=32, kernel_size=4, padding=1, stride=2))  # 176 176 

        dim_size = 32
        for i in range(4):
            layers_list.append(ResnetBlock(dim_size, padding_type=padding_type, activation=activation, norm_layer=norm_layer)) 
            layers_list.append(EncoderBlock(channel_in=dim_size, channel_out=dim_size*2, kernel_size=4, padding=1, stride=2)) 
            dim_size *= 2

        layers_list.append(ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  

        # final shape Bx256*7*6
        self.conv = nn.Sequential(*layers_list)
        self.fc_mu = nn.Sequential(nn.Linear(in_features=longsize, out_features=latent_dim))#,

        # self.fc_var = nn.Sequential(nn.Linear(in_features=longsize, out_features=latent_dim))#,

        for m in self.modules():
            weights_init_normal(m)

    def forward(self, ten):
        # ten = ten[:,:,:]
        # ten2 = jt.reshape(ten,[ten.size()[0],-1])
        # print(ten.shape, ten2.shape)
        ten = self.conv(ten)
        ten = torch.reshape(ten, (ten.size()[0],-1))
        # print(ten.shape,self.longsize)
        mu = self.fc_mu(ten)
        # logvar = self.fc_var(ten)
        # z = self.reparameterize(mu, logvar)
        # ----- AE ------
        return mu
        # # ----- VAE -----
        # if torch.isnan(mu).any():
        #     print('mu is nan')

        # if torch.isnan(logvar).any():
        #     print('logvar is nan')

        # if torch.isnan(z).any():
        #     print('z is nan')

        # if torch.isinf(mu).any():
        #     print('mu is inf')

        # if torch.isinf(logvar).any():
        #     print('logvar is inf')

        # if torch.isinf(z).any():
        #     print('z is inf')

        # return [mu, logvar, z]
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

class DecoderGenerator_image_Res(nn.Module):
    def __init__(self, norm_layer, image_size, output_nc, latent_dim=512):  
        super(DecoderGenerator_image_Res, self).__init__()
        # start from B*1024
        latent_size = int(image_size/32)
        self.latent_size = latent_size
        longsize = 512*latent_size*latent_size

        activation = nn.ReLU()
        padding_type='reflect'
        norm_layer=nn.BatchNorm2d

        self.fc = nn.Sequential(nn.Linear(in_features=latent_dim, out_features=longsize))
        layers_list = []

        layers_list.append(ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  # 176 176 
        
        dim_size = 256
        for i in range(4):
            layers_list.append(DecoderBlock(channel_in=dim_size*2, channel_out=dim_size, kernel_size=4, padding=1, stride=2, output_padding=0)) #latent*2
            layers_list.append(ResnetBlock(dim_size, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  
            dim_size = int(dim_size/2)

        layers_list.append(DecoderBlock(channel_in=32, channel_out=32, kernel_size=4, padding=1, stride=2, output_padding=0)) #352 352
        layers_list.append(ResnetBlock(32, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  # 176 176 

        # layers_list.append(DecoderBlock(channel_in=64, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0)) #96*160
        layers_list.append(nn.ReflectionPad2d(2))
        layers_list.append(nn.Conv2d(32,output_nc,kernel_size=5,padding=0))

        self.conv = nn.Sequential(*layers_list)

        for m in self.modules():
            weights_init_normal(m)

    def forward(self, ten):
        # print("in DecoderGenerator, print some shape ")
        # print(ten.size())
        ten = self.fc(ten)
        # print(ten.size())
        ten = torch.reshape(ten, (ten.size()[0],512, self.latent_size, self.latent_size))
        # print('ten', ten.size())
        ten = self.conv(ten)

        return ten    

class DecoderGenerator_sketch_Res(nn.Module):
    def __init__(self, norm_layer, image_size, output_nc, latent_dim=512):  
        super(DecoderGenerator_sketch_Res, self).__init__()
        # start from B*1024
        latent_size = int(image_size/8)
        self.latent_size = latent_size
        longsize = 512*latent_size*latent_size

        activation = nn.ReLU()
        padding_type='reflect'
        norm_layer=nn.BatchNorm2d

        self.fc = nn.Sequential(nn.Linear(in_features=latent_dim, out_features=longsize))
        layers_list = []

        layers_list.append(ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  # 176 176 
        
        dim_size = 128
        for i in range(3):
            layers_list.append(DecoderBlock(channel_in=dim_size*4, channel_out=dim_size, kernel_size=4, padding=1, stride=2, output_padding=0)) #latent*2
            layers_list.append(ResnetBlock(dim_size, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  
            dim_size = int(dim_size/4)

        layers_list.append(DecoderBlock(channel_in=8, channel_out=8, kernel_size=4, padding=1, stride=2, output_padding=0)) #352 352
        layers_list.append(ResnetBlock(8, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  # 176 176 

        # layers_list.append(DecoderBlock(channel_in=64, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0)) #96*160
        layers_list.append(nn.ReflectionPad2d(2))
        layers_list.append(nn.Conv2d(8,output_nc,kernel_size=5,padding=0,stride=2))

        self.conv = nn.Sequential(*layers_list)

        for m in self.modules():
            weights_init_normal(m)

    def forward(self, ten):
        # print("in DecoderGenerator, print some shape ")
        # print(ten.size())
        ten = self.fc(ten)
        # print(ten.size())
        ten = torch.reshape(ten, (ten.size()[0],512, self.latent_size, self.latent_size))
        # print('ten', ten.size())
        ten = self.conv(ten)

        return ten    