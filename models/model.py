import torch
import torch.nn as nn
from torch.autograd import Variable
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
        if DEBUG:
            print('G: level=%s, size=%s' % ('in', x.size()))
        for level in range(_from, _to, _step):
            if level == insert_y_at:
                x = self.chain[level](x, y)
            else:
                x = self.chain[level](x)

            if DEBUG:
                print('G: level=%d, size=%s' % (level, x.size()))

            if level == min_level:
                out['min_level'] = self.post[level](x)
            if level == max_level:
                out['max_level'] = self.post[level](x)
                x = resize_activations(out['min_level'], out['max_level'].size()) * min_level_weight + \
                    out['max_level'] * max_level_weight
        if DEBUG:
            print('G:', x.size())
        return x

class DSelectLayer(nn.Module):
    def __init__(self, pre, chain, inputs):
        super(DSelectLayer, self).__init__()
        assert len(chain) == len(inputs)
        self.pre = pre
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

        if self.pre is not None:
            x = self.pre(x)

        if DEBUG:
            print('D: level=%s, size=%s, max_level=%s, min_level=%s' % ('in', x.size(), max_level, min_level))

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

            if DEBUG:
                print('D: level=%d, size=%s' % (level, x.size()))
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


        pre = G_conv([], num_channels, self.get_nf(R), 7, 3, act, iact,
               False, self.use_wscale, self.use_pixelnorm)
        for i in range(R):
            ic,oc = self.get_nf(R-i), self.get_nf(R-i-1)
            pre = G_conv(pre, ic, oc, 3, 1, act, iact,
                        False, self.use_wscale, self.use_pixelnorm,stride=2)
            pre = G_conv(pre, oc, oc, 3, 1, act, iact,
                        False, self.use_wscale, self.use_pixelnorm)
        pre_model = nn.Sequential(*pre)
        lods = nn.ModuleList()
        nins = nn.ModuleList()

        layers = G_conv([], latent_size, self.get_nf(1), 4, 3, act, iact,
                        False, self.use_wscale, self.use_pixelnorm)
        net = G_conv(layers, latent_size, self.get_nf(1), 3, 1, act, iact,
                     True, self.use_wscale, self.use_pixelnorm)  # first block

        lods.append(net)
        nins.append(NINLayer([], self.get_nf(1), self.num_channels, output_act, output_iact, None, True,
                             self.use_wscale))  # to_rgb layer

        for I in range(2, R):  # following blocks
            ic, oc = self.get_nf(I - 1), self.get_nf(I)
            layers = [nn.Upsample(scale_factor=2, mode='nearest')]  # upsample
            layers = G_conv(layers, ic, oc, 3, 1, act, iact, False, self.use_wscale,
                            self.use_pixelnorm)
            net = G_conv(layers, oc, oc, 3, 1, act, iact, True, self.use_wscale,
                         self.use_pixelnorm)
            lods.append(net)
            nins.append(NINLayer([], oc, self.num_channels, output_act, output_iact, None, True,
                                 self.use_wscale))  # to_rgb layer

        self.output_layer = GSelectLayer(pre_model, lods, nins)

    def get_nf(self, stage):
        return min(int(self.fmap_base / (2.0 ** (stage * self.fmap_decay))), self.fmap_max)

    def forward(self, x, y=None, cur_level=None, insert_y_at=None):
        return self.output_layer(x, y, cur_level, insert_y_at)

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
                num_channels    = 1,        # Overridden based on dataset.
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
        pre = None

        nins.append(NINLayer([], self.num_channels, self.get_nf(R-1), act, iact, negative_slope, True, self.use_wscale))

        for I in range(R-1, 1, -1):
            ic, oc = self.get_nf(I), self.get_nf(I-1)
            net = D_conv([], ic, ic, 3, 1, act, iact, negative_slope, False,
                        self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
            net = D_conv(net, ic, oc, 3, 1, act, iact, negative_slope, False,
                        self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
            net += [nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)]
            lods.append(nn.Sequential(*net))
            # nin = [nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)]
            nin = []
            nin = NINLayer(nin, self.num_channels, oc, act, iact, negative_slope, True, self.use_wscale)
            nins.append(nin)

        net = []
        ic = oc = self.get_nf(1)
        if self.mbstat_avg is not None:
            net += [MinibatchStatConcatLayer(averaging=self.mbstat_avg)]
            ic += 1
        net = D_conv(net, ic, oc, 3, 1, act, iact, negative_slope, False,
                    self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
        net = D_conv(net, oc, self.get_nf(0), 4, 0, act, iact, negative_slope, False,
                    self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)

        # Increasing Variation Using MINIBATCH Standard Deviation
        if self.mbdisc_kernels:
            net += [MinibatchDiscriminationLayer(num_kernels=self.mbdisc_kernels)]

        oc = 1 + self.label_size
        # lods.append(NINLayer(net, self.get_nf(0), oc, 'linear', 'linear', None, True, self.use_wscale))
        lods.append(NINLayer(net, self.get_nf(0), oc, output_act, output_iact, None, True, self.use_wscale))

        self.output_layer = DSelectLayer(pre, lods, nins)

    def get_nf(self, stage):
        return min(int(self.fmap_base / (2.0 ** (stage * self.fmap_decay))), self.fmap_max)

    def forward(self, x, y=None, cur_level=None, insert_y_at=None, gdrop_strength=0.0):
        for module in self.modules():
            if hasattr(module, 'strength'):
                module.strength = gdrop_strength
        return self.output_layer(x, y, cur_level, insert_y_at)

