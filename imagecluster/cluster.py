# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, 2018-11-12 20:14:49
# ***
# ************************************************************************************/


from torch import nn
from torch.autograd import Function
import torch
import random
import colorsys
from PIL import Image

from imagecluster import cluster_cpp


class RGB565(object):
    """RGB888 <---> RGB565."""

    @staticmethod
    def NO(r, g, b):
        return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)

    @staticmethod
    def R(x16):
        return (((x16 >> 11) & 0x1f) << 3)

    @staticmethod
    def G(x16):
        return (((x16 >> 5) & 0x3f) << 2)

    @staticmethod
    def B(x16):
        return ((x16 & 0x1f) << 3)


def random_colors(nums, bright=True, shuffle=True):
    """Generate colors from HSV space to RGB."""
    brightness = 1.0 if bright else 0.7
    hsv = [(i / nums, 1, brightness) for i in range(nums)]
    fcolors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    if shuffle:
        random.shuffle(fcolors)
    return torch.Tensor(fcolors)


def check_input(input):
    """Check Bx3xHxW tensor."""
    assert input.dim() == 4 and input.size(1) == 3, "input should be Bx3xHxW Tensor"


def check_label(label):
    """Check Bx1xHxW tensor."""
    assert label.dim() == 4 and label.size(1) == 1, "label should be Bx1xHxW Tensor"


def check_mask(mask):
    """Check Bx1xHxW tensor."""
    assert mask.dim() == 4 and mask.size(1) == 1, "mask should be Bx3xHxW Tensor"


def check_output(output):
    """Check Bx3xHxW tensor."""
    assert output.dim() == 4 and output.size(1) == 3, "output should be Bx3xHxW Tensor"


class ClusterFunction(Function):
    """Cluster Function."""

    @staticmethod
    def forward(ctx, input, index, center):
        """Forward."""
        check_input(input)
        output = input.clone()

        # label is a tensor with dim: Bx1xHxW, class per an element
        label = torch.IntTensor(input.size(0), 1, input.size(2), input.size(3))
        cluster_cpp.forward(input, index, center, output, label)

        return output, label

    @staticmethod
    def backward(ctx, grad_output, label):
        """Backward."""
        grad_input = grad_output.new()
        cluster_cpp.backward(grad_output, input, grad_input)
        return grad_input


class Cluster(nn.Module):
    """Cluster Class."""

    def __init__(self, imagefiles, K, maxloops=256):
        """Init."""
        super(Cluster, self).__init__()
        assert K >= 2, "Cluster numbers must be greater than 2"
        self.K = K
        self.maxloops = maxloops
        hist = self.histogram(imagefiles)
        self.index = torch.IntTensor(65536,)
        self.center = torch.FloatTensor(K, 4)   # r, g, b, w
        cluster_cpp.cluster(hist, K, self.maxloops, self.index, self.center)

    def forward(self, input):
        """Forward."""
        check_input(input)
        output, label = ClusterFunction.apply(input.cpu(), self.index, self.center)
        return output, label

    def histogram(self, imagefiles):
        """Simple histogram."""
        count = [0 for i in range(65536)]
        for i in range(len(imagefiles)):
            img = Image.open(imagefiles[i])
            w, h = img.size
            pixels = img.load()
            for i in range(h):
                for j in range(w):
                    (r, g, b) = pixels[j, i]
                    count[RGB565.NO(r, g, b)] += 1
        hist = torch.FloatTensor(count)
        sum = hist.sum()
        hist /= sum
        return hist


def Segment(label, radius=2):
    """Segment, radius==2 means 5x5 neighbours."""
    check_label(label)
    assert radius > 0, "radius must be greater than 0"
    label_clone = label.clone()
    segment_mask = torch.zeros_like(label)
    cluster_cpp.segment(label_clone, radius, segment_mask)
    return segment_mask


def ColorMask(mask):
    """Mask is Nx1xHxW."""
    check_mask(mask)
    NCOLORS = 32
    colors = random_colors(NCOLORS)
    output = torch.Tensor(mask.size(0), 3, mask.size(2), mask.size(3))
    check_output(output)
    cluster_cpp.colormask(mask, colors, output)
    return output


def AdjMatrix(mask, radius=2):
    """Mask is Nx1xHxW, radius==2 means 5x5 neighbours, same as segment."""
    check_mask(mask)
    n = mask.max().item() + 1   # 0 -- Background ?
    output = torch.IntTensor(n, n)
    cluster_cpp.adjmatrix(mask, radius, output)
    return output
