# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_activation_layer(inplanes, relu_type="relu"):
    if relu_type == "relu":
        act_func = nn.ReLU(inplace=True) 
    elif relu_type == "prelu":
        act_func = nn.PReLU()
    elif relu_type == "leaky":
        act_func = nn.LeakyReLU(0.2)
    else:
        print("Not support this type of acitvation function.")
        return 0
    return act_func


def make_a_conv_layer(inplanes, outplanes, ksize=3, stride=1, pad=1, bn=True):
    if bn:
        return nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=ksize, stride=stride, padding=pad, bias=False),
                             nn.BatchNorm2d(outplanes))
    else:
        return nn.Conv2d(inplanes, outplanes, kernel_size=ksize, stride=stride, padding=pad, bias=True)


def make_a_conv_relu_layer(inplanes, outplanes, ksize=3, stride=1, pad=1, bn=True, relu_type="relu"):
    return nn.Sequential(make_a_conv_layer(inplanes, outplanes, ksize=ksize, stride=stride, pad=pad, bn=bn),        
                         make_activation_layer(outplanes, relu_type=relu_type))


'''
Make a sequence of conv layers
'''
def make_conv_layers(repeats, in_dim, out_dim, make_layer=make_a_conv_relu_layer, relu_type="relu", expansion=1):
    layers = [make_layer(in_dim, out_dim, relu_type=relu_type)] # default 3x3@s1p1 with bn and relu
    for _ in range(1, repeats):
        layers.append(make_layer(out_dim, out_dim, relu_type=relu_type))
    return nn.Sequential(*layers)


'''
Make a residual block
'''
class make_a_res_block(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, expansion=1):
        super(make_a_res_block, self).__init__()
        self._expansion = expansion
        # (2a) increase #channel using 1x1 conv or halve feature map size
        self.branch2a = make_a_conv_relu_layer(in_dim, out_dim, ksize=1, stride=stride, pad=0)
        # (2b) regular conv layer
        self.branch2b = make_a_conv_relu_layer(out_dim, out_dim)
        self._out_dim = out_dim * expansion
        # (2c) decrease #channel using 1x1 conv
        self.branch2c = make_a_conv_layer(out_dim, self._out_dim, ksize=1, pad=0)
        # (1) increase #channel or halve feature map size
        self.branch1 =  make_a_conv_relu_layer(in_dim, self._out_dim, ksize=1, stride=stride, pad=0) \
                                        if stride != 1 or in_dim != self._out_dim else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch2 = self.branch2a(x)
        branch2 = self.branch2b(branch2)
        branch2 = self.branch2c(branch2)
        branch1 = self.branch1(x)
        y = self.relu(branch1 + branch2)
        return y


'''
Make a basic residual block
'''
class make_a_res_block_basic(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        super(make_a_res_block_basic, self).__init__()
        self.branch2a = make_a_conv_relu_layer(in_dim, out_dim, ksize=3, stride=stride)
        self.branch2b = make_a_conv_relu_layer(out_dim, out_dim, ksize=3)
        self.branch1 =  make_a_conv_layer(in_dim, out_dim, ksize=1, stride=stride) \
                                        if stride != 1 or in_dim != out_dim else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch2 = self.branch2a(x)
        branch2 = self.branch2b(branch2)
        branch1 = self.branch1(x)
        y = self.relu(branch1 + branch2)
        return y


'''
Make a sequence of residual layers
'''
def make_res_layers(repeats, in_dim, out_dim, make_layer=make_a_res_block, expansion=1, reverse=False, relu_type="relu"): 
    if reverse:
        layers = []
        for _ in range(1, repeats):
            layers.append(make_layer(in_dim, in_dim, expansion=expansion))
        layers.append(make_layer(in_dim, out_dim, expansion=expansion))
    else:
        layers = [make_layer(in_dim, out_dim, expansion=expansion)]
        for _ in range(1, repeats):
            layers.append(make_layer(out_dim, out_dim, expansion=expansion))
    return nn.Sequential(*layers)



'''
Make an upsampling layer
'''
class make_upsample_layer(nn.Module):
    def __init__(self):
        super(make_upsample_layer, self).__init__()
    
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2)


'''
Make a transposed conv layer
'''
def make_up_layer(layer_type="upsample", in_dims=0, out_dims=0):
    if layer_type == "upsample":
        return make_upsample_layer()
    elif layer_type == "transconv":
        return nn.ConvTranspose2d(in_dims, out_dims, kernel_size=2, stride=2)


'''
Make a pooling layer
'''
def make_pool_layer(pool_type='avg'):
    if pool_type == 'agv':
        return nn.AvgPool2d(2, stride=2)
    elif pool_type == 'max':
        return nn.MaxPool2d(2, stride=2, ceil_mode=False)


'''
Make a merge layer
'''
class mergeup(nn.Module):
    def __init__(self, merge_type):
        super(mergeup, self).__init__()
        self.merge_type = merge_type

    def forward(self, up1, up2):
        if self.merge_type == "add":
            return up1 + up2
        elif self.merge_type == "prod":
            return up1 * up2
        elif self.merge_type == "concat":
            return torch.cat((up1, up2), dim=1)


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad