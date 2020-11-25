import torch
import torch.nn as nn

OPS = {
    'mbconv_k3_t1': lambda C_in, C_out, stride, use_se, affine, track_running_stats: MBConv(C_in, C_out, 3, stride, 1, t=1, use_se=use_se, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k3_t3': lambda C_in, C_out, stride, use_se, affine, track_running_stats: MBConv(C_in, C_out, 3, stride, 1, t=3, use_se=use_se, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k3_t6': lambda C_in, C_out, stride, use_se, affine, track_running_stats: MBConv(C_in, C_out, 3, stride, 1, t=6, use_se=use_se, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k5_t1': lambda C_in, C_out, stride, use_se, affine, track_running_stats: MBConv(C_in, C_out, 5, stride, 2, t=1, use_se=use_se, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k5_t3': lambda C_in, C_out, stride, use_se, affine, track_running_stats: MBConv(C_in, C_out, 5, stride, 2, t=3, use_se=use_se, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k5_t6': lambda C_in, C_out, stride, use_se, affine, track_running_stats: MBConv(C_in, C_out, 5, stride, 2, t=6, use_se=use_se, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k7_t1': lambda C_in, C_out, stride, use_se, affine, track_running_stats: MBConv(C_in, C_out, 7, stride, 3, t=1, use_se=use_se, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k7_t3': lambda C_in, C_out, stride, use_se, affine, track_running_stats: MBConv(C_in, C_out, 7, stride, 3, t=3, use_se=use_se, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k7_t6': lambda C_in, C_out, stride, use_se, affine, track_running_stats: MBConv(C_in, C_out, 7, stride, 3, t=6, use_se=use_se, affine=affine, track_running_stats=track_running_stats),
    'basic_block': lambda C_in, C_out, stride, use_se, affine, track_running_stats: BasicBlock(C_in, C_out, stride, affine=affine, track_running_stats=track_running_stats),
    'bottle_neck': lambda C_in, C_out, stride, use_se, affine, track_running_stats: Bottleneck(C_in, C_out, stride, affine=affine, track_running_stats=track_running_stats),
    'skip_connect': lambda C_in, C_out, stride, use_se, affine, track_running_stats: Skip(C_in, C_out, stride, affine=affine, track_running_stats=track_running_stats),
}


class MBConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, t=3, affine=True, 
                    track_running_stats=True, use_se=False):
        super(MBConv, self).__init__()
        if t > 1:
            C_hidden = C_in*t
            self._expand_conv = nn.Sequential(
                nn.Conv2d(C_in, C_hidden, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(C_hidden, affine=affine, track_running_stats=track_running_stats),
                nn.ReLU6(inplace=True))

            self._depthwise_conv = nn.Sequential(
                nn.Conv2d(C_hidden, C_hidden, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_hidden, bias=False),
                nn.BatchNorm2d(C_hidden, affine=affine, track_running_stats=track_running_stats),
                nn.ReLU6(inplace=True))

            self._project_conv = nn.Sequential(
                nn.Conv2d(C_hidden, C_out, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats))
        else:
            C_hidden = C_in
            self._expand_conv = None

            self._depthwise_conv = nn.Sequential(
                nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
                nn.BatchNorm2d(C_in, affine=affine, track_running_stats=track_running_stats),
                nn.ReLU6(inplace=True))

            self._project_conv = nn.Sequential(
                nn.Conv2d(C_in, C_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(C_out))

        self.se = SELayer(C_hidden) if use_se else None

    def forward(self, x):
        input_data = x
        if self._expand_conv is not None:
            x = self._expand_conv(x)
        x = self._depthwise_conv(x)
        if self.se is not None:
            x = self.se(x)
        out_data = self._project_conv(x)

        if out_data.shape == input_data.shape:
            return out_data + input_data
        else:
            return out_data


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 affine=True, track_running_stats=True):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, affine=affine, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, affine=affine, track_running_stats=track_running_stats)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes, affine=affine, track_running_stats=track_running_stats),
            )

    def forward(self, x):  
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, affine=True, track_running_stats=True):
        super(Bottleneck, self).__init__()
        if inplanes != 32:
            inplanes *= 4
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample = None
        if stride != 1 or inplanes != planes*4:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * 4, stride),
                nn.BatchNorm2d(planes * 4, affine=affine, track_running_stats=track_running_stats),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Skip(nn.Module):
    def __init__(self, C_in, C_out, stride, affine=True, track_running_stats=True):
        super(Skip, self).__init__()
        if C_in!=C_out:
            skip_conv = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size=1, stride=stride, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats))
            stride = 1
        self.op=Identity(stride)

        if C_in!=C_out:
            self.op=nn.Sequential(skip_conv, self.op)

    def forward(self,x):
        return self.op(x)

class Identity(nn.Module):
    def __init__(self, stride):
        super(Identity, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x
        else:
            return x[:, :, ::self.stride, ::self.stride]
