# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, 2018-11-12 20:14:49
# ***
# ************************************************************************************/


from torch import nn
from torch.autograd import Function
import torch
from PIL import Image

from imagecluster import cluster_cpp


class RGB565(object):
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


class ClusterFunction(Function):
    @staticmethod
    def forward(ctx, input, index, center):
        assert input.dim() == 4, "input should be BxCxHxW Tensor"
        output = input.clone()

        # label is a tensor with dim: Bx1xHxW, every element means one class
        label = torch.IntTensor(input.size(0), 1, input.size(2), input.size(3))
        cluster_cpp.forward(input, index, center, output, label)

        return output, label

    @staticmethod
    def backward(ctx, grad_output, label):
        grad_input = grad_output.new()
        cluster_cpp.backward(grad_output, input, grad_input)
        return grad_input


class Cluster(nn.Module):
    def __init__(self, imagefiles, K, maxloops=256):
        super(Cluster, self).__init__()
        assert K >= 2, "Cluster numbers must be greater than 2"
        self.maxloops = maxloops
        hist = self.histogram(imagefiles)
        self.index = torch.IntTensor(65536,)
        self.center = torch.FloatTensor(K, 4)   # r, g, b, w
        cluster_cpp.cluster(hist, K, self.maxloops, self.index, self.center)

    def forward(self, input):
        output, label = ClusterFunction.apply(input.cpu(), self.index, self.center)
        return output, label

    def histogram(self, imagefiles):
        count = [0 for i in range(65536)]
        for i in range(len(imagefiles)):
            img = Image.open(imagefiles[i])
            w, h = img.size
            pixels = img.load()
            for i in range(h):
                for j in range(w):
                    (r, g, b) = pixels[i, j]
                    count[RGB565.NO(r, g, b)] += 1
        hist = torch.FloatTensor(count)
        sum = hist.sum()
        hist /= sum
        return hist
