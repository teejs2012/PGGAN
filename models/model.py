import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.init import kaiming_normal, calculate_gain
import numpy as np
import sys


def resize_activations(v, so):
    """
    Resize activation tensor 'v' of shape 'si' to match shape 'so'.
    :param v:
    :param so:
    :return:
    """
    si = list(v.size())
    so = list(so)
    assert len(si) == len(so) and si[0] == so[0]

    # Decrease feature maps.
    if si[1] > so[1]:
        v = v[:, :so[1]]

    # Shrink spatial axes.
    if len(si) == 4 and (si[2] > so[2] or si[3] > so[3]):
        assert si[2] % so[2] == 0 and si[3] % so[3] == 0
        ks = (si[2] // so[2], si[3] // so[3])
        v = F.avg_pool2d(v, kernel_size=ks, stride=ks, ceil_mode=False, padding=0, count_include_pad=False)

    # Extend spatial axes. Below is a wrong implementation
    # shape = [1, 1]
    # for i in range(2, len(si)):
    #     if si[i] < so[i]:
    #         assert so[i] % si[i] == 0
    #         shape += [so[i] // si[i]]
    #     else:
    #         shape += [1]
    # v = v.repeat(*shape)
    if si[2] < so[2]:
        assert so[2] % si[2] == 0 and so[2] / si[2] == so[3] / si[3]  # currently only support this case
        v = F.upsample(v, scale_factor=so[2] // si[2], mode='nearest')

    # Increase feature maps.
    if si[1] < so[1]:
        z = torch.zeros((v.shape[0], so[1] - si[1]) + so[2:])
        v = torch.cat([v, z], 1)
    return v

class MinibatchStatConcatLayer(nn.Module):
    """Minibatch stat concatenation layer.
    - averaging tells how much averaging to use ('all', 'spatial', 'none')
    """
    def __init__(self, averaging='all'):
        super(MinibatchStatConcatLayer, self).__init__()
        self.averaging = averaging.lower()
        if 'group' in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            assert self.averaging in ['all', 'flat', 'spatial', 'none', 'gpool'], 'Invalid averaging mode'%self.averaging
        self.adjusted_std = lambda x, **kwargs: torch.sqrt(torch.mean((x - torch.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8) #Tstdeps in the original implementation

    def forward(self, x):
        shape = list(x.size())
        target_shape = shape.copy()
        vals = self.adjusted_std(x, dim=0, keepdim=True)# per activation, over minibatch dim
        if self.averaging == 'all':  # average everything --> 1 value per minibatch
            target_shape[1] = 1
            vals = torch.mean(vals, dim=1, keepdim=True)#vals = torch.mean(vals, keepdim=True)

        elif self.averaging == 'spatial':  # average spatial locations
            if len(shape) == 4:
                vals = mean(vals, axis=[2,3], keepdim=True)  # torch.mean(torch.mean(vals, 2, keepdim=True), 3, keepdim=True)
        elif self.averaging == 'none':  # no averaging, pass on all information
            target_shape = [target_shape[0]] + [s for s in target_shape[1:]]
        elif self.averaging == 'gpool':  # EXPERIMENTAL: compute variance (func) over minibatch AND spatial locations.
            if len(shape) == 4:
                vals = mean(x, [0,2,3], keepdim=True)  # torch.mean(torch.mean(torch.mean(x, 2, keepdim=True), 3, keepdim=True), 0, keepdim=True)
        elif self.averaging == 'flat':  # variance of ALL activations --> 1 value per minibatch
            target_shape[1] = 1
            vals = torch.FloatTensor([self.adjusted_std(x)])
        else:  # self.averaging == 'group'  # average everything over n groups of feature maps --> n values per minibatch
            target_shape[1] = self.n
            vals = vals.view(self.n, self.shape[1]/self.n, self.shape[2], self.shape[3])
            vals = mean(vals, axis=0, keepdim=True).view(1, self.n, 1, 1)
        vals = vals.expand(*target_shape)
        return torch.cat([x, vals], 1) # feature-map concatanation

    def __repr__(self):
        return self.__class__.__name__ + '(averaging = %s)' % (self.averaging)


class MinibatchDiscriminationLayer(nn.Module):
    def __init__(self, num_kernels):
        super(MinibatchDiscriminationLayer, self).__init__()
        self.num_kernels = num_kernels

    def forward(self, x):
        pass

class LayerNormLayer(nn.Module):
    """
    Layer normalization. Custom reimplementation based on the paper: https://arxiv.org/abs/1607.06450
    """
    def __init__(self, incoming, eps=1e-4):
        super(LayerNormLayer, self).__init__()
        self.incoming = incoming
        self.eps = eps
        self.gain = Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.bias = None

        if self.incoming.bias is not None:
            self.bias = self.incoming.bias
            self.incoming.bias = None

    def forward(self, x):
        x = x - mean(x, axis=range(1, len(x.size())))
        x = x * 1.0/(torch.sqrt(mean(x**2, axis=range(1, len(x.size())), keepdim=True) + self.eps))
        x = x * self.gain
        if self.bias is not None:
            x += self.bias
        return x

    def __repr__(self):
        param_str = '(incoming = %s, eps = %s)' % (self.incoming.__class__.__name__, self.eps)
        return self.__class__.__name__ + param_str

class GDropLayer(nn.Module):
    """
    # Generalized dropout layer. Supports arbitrary subsets of axes and different
    # modes. Mainly used to inject multiplicative Gaussian noise in the network.
    """
    def __init__(self, mode='mul', strength=0.2, axes=(0,1), normalize=False):
        super(GDropLayer, self).__init__()
        self.mode = mode.lower()
        assert self.mode in ['mul', 'drop', 'prop'], 'Invalid GDropLayer mode'%mode
        self.strength = strength
        self.axes = [axes] if isinstance(axes, int) else list(axes)
        self.normalize = normalize
        self.gain = None

    def forward(self, x, deterministic=False):
        if deterministic or not self.strength:
            return x

        rnd_shape = [s if axis in self.axes else 1 for axis, s in enumerate(x.size())]  # [x.size(axis) for axis in self.axes]
        if self.mode == 'drop':
            p = 1 - self.strength
            rnd = np.random.binomial(1, p=p, size=rnd_shape) / p
        elif self.mode == 'mul':
            rnd = (1 + self.strength) ** np.random.normal(size=rnd_shape)
        else:
            coef = self.strength * x.size(1) ** 0.5
            rnd = np.random.normal(size=rnd_shape) * coef + 1

        if self.normalize:
            rnd = rnd / np.linalg.norm(rnd, keepdims=True)
        rnd = Variable(torch.from_numpy(rnd).type(x.data.type()))
        if x.is_cuda:
            rnd = rnd.cuda()
        return x * rnd

    def __repr__(self):
        param_str = '(mode = %s, strength = %s, axes = %s, normalize = %s)' % (self.mode, self.strength, self.axes, self.normalize)
        return self.__class__.__name__ + param_str

class PixelNormLayer(nn.Module):
    """
    Pixelwise feature vector normalization.
    """

    def __init__(self, eps=1e-8):
        super(PixelNormLayer, self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

    def __repr__(self):
        return self.__class__.__name__ + '(eps = %s)' % (self.eps)

class WScaleLayer(nn.Module):
    """
    Applies equalized learning rate to the preceding layer.
    """
    def __init__(self, incoming):
        super(WScaleLayer, self).__init__()
        self.incoming = incoming
        self.scale = (torch.mean(self.incoming.weight.data ** 2)) ** 0.5
        self.incoming.weight.data.copy_(self.incoming.weight.data / self.scale)
        self.bias = None
        if self.incoming.bias is not None:
            self.bias = self.incoming.bias
            self.incoming.bias = None

    def forward(self, x):
        x = self.scale * x
        if self.bias is not None:
            x += self.bias.view(1, self.bias.size()[0], 1, 1)
        return x

    def __repr__(self):
        param_str = '(incoming = %s)' % (self.incoming.__class__.__name__)
        return self.__class__.__name__ + param_str

class GSelectLayer(nn.Module):
    def __init__(self, pre, chain, post):
        super(GSelectLayer, self).__init__()
        assert len(chain) == len(post)
        self.pre = pre
        self.chain = chain
        self.post = post
        self.N = len(self.chain)

    def forward(self, x, y=None, cur_level=None, insert_y_at=None):
        if cur_level is None:
            cur_level = self.N  # cur_level: physical index
        if y is not None:
            assert insert_y_at is not None

        min_level, max_level = int(np.floor(cur_level - 1)), int(np.ceil(cur_level - 1))
        min_level_weight, max_level_weight = int(cur_level + 1) - cur_level, cur_level - int(cur_level)

        _from, _to, _step = 0, max_level + 1, 1

        if self.pre is not None:
            x = self.pre(x)

        out = {}

        for level in range(_from, _to, _step):
            if level == insert_y_at:
                x = self.chain[level](x, y)
            else:
                x = self.chain[level](x)

            if level == min_level:
                out['min_level'] = self.post[level](x)
            if level == max_level:
                out['max_level'] = self.post[level](x)
                x = resize_activations(out['min_level'], out['max_level'].size()) * min_level_weight + \
                    out['max_level'] * max_level_weight

        return x

class DSelectLayer(nn.Module):
    def __init__(self, chain, inputs):
        super(DSelectLayer, self).__init__()
        assert len(chain) == len(inputs)
        self.chain = chain
        self.inputs = inputs
        self.N = len(self.chain)

    def forward(self, x, y=None, cur_level=None, insert_y_at=None):
        if cur_level is None:
            cur_level = self.N  # cur_level: physical index
        if y is not None:
            assert insert_y_at is not None

        max_level, min_level = int(np.floor(self.N - cur_level)), int(np.ceil(self.N - cur_level))
        min_level_weight, max_level_weight = int(cur_level + 1) - cur_level, cur_level - int(cur_level)

        _from, _to, _step = min_level + 1, self.N, 1

        if max_level == min_level:
            x = self.inputs[max_level](x)
            if max_level == insert_y_at:
                x = self.chain[max_level](x, y)
            else:
                x = self.chain[max_level](x)
        else:
            out = {}
            tmp = self.inputs[max_level](x)
            if max_level == insert_y_at:
                tmp = self.chain[max_level](tmp, y)
            else:
                tmp = self.chain[max_level](tmp)
            out['max_level'] = tmp
            out['min_level'] = self.inputs[min_level](x)
            x = resize_activations(out['min_level'], out['max_level'].size()) * min_level_weight + \
                out['max_level'] * max_level_weight
            if min_level == insert_y_at:
                x = self.chain[min_level](x, y)
            else:
                x = self.chain[min_level](x)

        for level in range(_from, _to, _step):
            if level == insert_y_at:
                x = self.chain[level](x, y)
            else:
                x = self.chain[level](x)

        return x


def he_init(layer, nonlinearity='conv2d', param=None):
    nonlinearity = nonlinearity.lower()
    if nonlinearity not in ['linear', 'conv1d', 'conv2d', 'conv3d', 'relu', 'leaky_relu', 'sigmoid', 'tanh']:
        if not hasattr(layer, 'gain') or layer.gain is None:
            gain = 0  # default
        else:
            gain = layer.gain
    elif nonlinearity == 'leaky_relu':
        assert param is not None, 'Negative_slope(param) should be given.'
        gain = calculate_gain(nonlinearity, param)
    else:
        gain = calculate_gain(nonlinearity)
    kaiming_normal(layer.weight, a=gain)

def G_conv(incoming, in_channels, out_channels, kernel_size, padding, nonlinearity, init,
        to_sequential=True, use_wscale=True, use_pixelnorm=True, stride=1):
    layers = incoming
    layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)]
    he_init(layers[-1], init, 0.2)  # init layers
    if use_wscale:
        layers += [WScaleLayer(layers[-1])]
    layers += [nonlinearity]
    # if use_batchnorm:
    #     layers += [nn.BatchNorm2d(out_channels)]
    if use_pixelnorm:
        layers += [PixelNormLayer()]
    if to_sequential:
        return nn.Sequential(*layers)
    else:
        return layers


def NINLayer(incoming, in_channels, out_channels, nonlinearity, init, param=None,
            to_sequential=True, use_wscale=True):
    layers = incoming
    layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)]  # NINLayer in lasagne
    he_init(layers[-1], init, param)  # init layers
    if use_wscale:
        layers += [WScaleLayer(layers[-1])]
    if not (nonlinearity == 'linear'):
        layers += [nonlinearity]
    if to_sequential:
        return nn.Sequential(*layers)
    else:
        return layers

class Generator(nn.Module):
    def __init__(self,
                 num_channels=3,  # Overridden based on dataset.
                 resolution=256,  # Overridden based on dataset.
                 label_size=0,  # Overridden based on dataset.
                 fmap_base=4096,
                 fmap_decay=1.0,
                 fmap_max=256,
                 latent_size=None,
                 use_wscale=True,
                 use_pixelnorm=True,
                 use_batchnorm=False,
                 tanh_at_end=None):
        super(Generator, self).__init__()
        self.num_channels = num_channels
        self.resolution = resolution
        self.label_size = label_size
        self.fmap_base = fmap_base
        self.fmap_decay = fmap_decay
        self.fmap_max = fmap_max
        self.latent_size = latent_size
        self.use_wscale = use_wscale
        self.use_pixelnorm = use_pixelnorm
        self.tanh_at_end = tanh_at_end

        R = int(np.log2(resolution))
        assert resolution == 2 ** R and resolution >= 4

        act = nn.ReLU()
        iact = 'relu'
        output_act = nn.Tanh() if self.tanh_at_end else 'linear'
        output_iact = 'tanh' if self.tanh_at_end else 'linear'


        pre = [nn.ReflectionPad2d(3),
               nn.Conv2d(num_channels, self.get_nf(R), kernel_size=7, padding=0),
               nn.InstanceNorm2d(self.get_nf(R)),
               nn.ReLU(True)]
        for i in range(R):
            ic,oc = self.get_nf(R-i), self.get_nf(R-i-1)
            pre += [nn.Conv2d(ic, oc, kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(oc, oc, kernel_size=2, stride=1, padding=1),
                    nn.InstanceNorm2d(self.get_nf(R)),
                    nn.ReLU(True)]

        pre_model = nn.Sequential(*pre)
        pre_model = init_weights(pre_model)

        lods = nn.ModuleList()
        nins = nn.ModuleList()

        oc = self.get_nf(1)
        layers = [nn.Conv2d(latent_size, oc, kernel_size=4, stride=1, padding=3),
                    nn.Conv2d(oc, oc, kernel_size=3, stride=1, padding=1),
                    nn.InstanceNorm2d(oc),
                    nn.ReLU(True)]
        net = nn.Sequential(*layers)

        lods.append(net)
        nins.append(nn.Sequential(nn.ReflectionPad2d(3),
                                      nn.Conv2d(oc, self.num_channels, kernel_size=7, padding=0),
                                      nn.Tanh()))  # to_rgb layer

        for I in range(2, R):  # following blocks
            ic, oc = self.get_nf(I - 1), self.get_nf(I)
            layers = [nn.Upsample(scale_factor=2, mode='nearest'),
                      nn.Conv2d(ic, oc, kernel_size=3, stride=1, padding=1),
                      nn.Conv2d(oc, oc, kernel_size=3, stride=1, padding=1),
                      nn.InstanceNorm2d(oc),
                      nn.ReLU(True)]
            net = nn.Sequential(*layers)
            net = init_weights(net)
            lods.append(net)
            nin = nn.Sequential(nn.ReflectionPad2d(3),
                                      nn.Conv2d(oc, self.num_channels, kernel_size=7, padding=0),
                                      nn.Tanh())
            nin = init_weights(nin)
            nins.append(nin)  # to_rgb layer

        self.output_layer = GSelectLayer(pre_model, lods, nins)

    def get_nf(self, stage):
        return min(int(self.fmap_base / (2.0 ** (stage * self.fmap_decay))), self.fmap_max)

    def forward(self, x, y=None, cur_level=None, insert_y_at=None):
        return self.output_layer(x, y, cur_level, insert_y_at)

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

#     print('initialize network with %s' % init_type)
    net.apply(init_func)
    return net

def D_conv(incoming, in_channels, out_channels, kernel_size, padding, nonlinearity, init, param=None,
        to_sequential=True, use_wscale=True, use_gdrop=True, use_layernorm=False, gdrop_param=dict()):
    layers = incoming
    if use_gdrop:
        layers += [GDropLayer(**gdrop_param)]
    layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
    he_init(layers[-1], init, param)  # init layers
    if use_wscale:
        layers += [WScaleLayer(layers[-1])]
    layers += [nonlinearity]
    if use_layernorm:
        layers += [LayerNormLayer()]  # TODO: requires incoming layer
    if to_sequential:
        return nn.Sequential(*layers)
    else:
        return layers


class Discriminator(nn.Module):
    def __init__(self,
                num_channels    = 3,        # Overridden based on dataset.
                resolution      = 32,       # Overridden based on dataset.
                label_size      = 0,        # Overridden based on dataset.
                fmap_base       = 4096,
                fmap_decay      = 1.0,
                fmap_max        = 256,
                mbstat_avg      = 'all',
                mbdisc_kernels  = None,
                use_wscale      = True,
                use_gdrop       = True,
                use_layernorm   = False,
                sigmoid_at_end  = False):
        super(Discriminator, self).__init__()
        self.num_channels = num_channels
        self.resolution = resolution
        self.label_size = label_size
        self.fmap_base = fmap_base
        self.fmap_decay = fmap_decay
        self.fmap_max = fmap_max
        self.mbstat_avg = mbstat_avg
        self.mbdisc_kernels = mbdisc_kernels
        self.use_wscale = use_wscale
        self.use_gdrop = use_gdrop
        self.use_layernorm = use_layernorm
        self.sigmoid_at_end = sigmoid_at_end

        R = int(np.log2(resolution))
        assert resolution == 2**R and resolution >= 4
        gdrop_strength = 0.0

        negative_slope = 0.2
        act = nn.LeakyReLU(negative_slope=negative_slope)
        # input activation
        iact = 'leaky_relu'
        # output activation
        output_act = nn.Sigmoid() if self.sigmoid_at_end else 'linear'
        output_iact = 'sigmoid' if self.sigmoid_at_end else 'linear'
        gdrop_param = {'mode': 'prop', 'strength': gdrop_strength}

        nins = nn.ModuleList()
        lods = nn.ModuleList()

        nin = nn.Conv2d(self.num_channels, self.get_nf(R-1), kernel_size=7, stride=1, padding=3)
        nin = init_weights(nin)
        nins.append(nin)

        for I in range(R-1, 1, -1):
            ic, oc = self.get_nf(I), self.get_nf(I-1)
            net = [nn.Conv2d(ic, ic, kernel_size=3, stride=1, padding=1),
                   nn.InstanceNorm2d(ic),
                   nn.LeakyReLU(negative_slope, True),
                   nn.Conv2d(ic, oc, kernel_size=3, stride=1, padding=1),
                   nn.InstanceNorm2d(oc),
                   nn.LeakyReLU(negative_slope, True),
                   nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)]
            net = init_weights(net)
            lods.append(nn.Sequential(*net))

            nin = nn.Conv2d(self.num_channels, oc, kernel_size=7, stride=1, padding=3)
            nin = init_weights(nin)
            nins.append(nin)

        ic, oc = self.get_nf(1), self.get_nf(0)
        net = nn.Sequential(nn.Conv2d(ic, ic, kernel_size=3, stride=1, padding=1),
                           nn.InstanceNorm2d(ic),
                           nn.LeakyReLU(negative_slope, True),
                           nn.Conv2d(ic, oc, kernel_size=3, stride=1, padding=1),
                           nn.InstanceNorm2d(oc),
                           nn.LeakyReLU(negative_slope, True),
                           nn.Conv2d(oc,self.num_channels,kernel_size=1,stride=1))
        net = init_weights(net)
        lods.append(net)

        self.output_layer = DSelectLayer(lods, nins)

    def get_nf(self, stage):
        return min(int(self.fmap_base / (2.0 ** (stage * self.fmap_decay))), self.fmap_max)

    def forward(self, x, y=None, cur_level=None, insert_y_at=None, gdrop_strength=0.0):
        for module in self.modules():
            if hasattr(module, 'strength'):
                module.strength = gdrop_strength
        return self.output_layer(x, y, cur_level, insert_y_at)

