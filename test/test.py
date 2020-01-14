# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, , 2018-10-17 01:50:34
# ***
# ************************************************************************************/

import torch
from PIL import Image
import torchvision
from torchvision import transforms
import imagecluster as ic

import time
import sys
import pdb
import colorsys
import random


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


def image_open(filename):
    return Image.open(filename).convert('RGB')


def to_tensor(image):
    """
    return 1xCxHxW tensor
    """
    transform = transforms.Compose([transforms.ToTensor()])
    t = transform(image)
    return t.unsqueeze(0).to(device)


def random_colors(nums, bright=True, shuffle=True):
    """Generate colors from HSV space to RGB."""
    brightness = 1.0 if bright else 0.7
    hsv = [(i / nums, 1, brightness) for i in range(nums)]
    fcolors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    pdb.set_trace()

    colors = []
    for (r, g, b) in fcolors:
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    if shuffle:
        random.shuffle(colors)
    pdb.set_trace()

    return colors


def from_tensor(tensor):
    """
    tensor format: 1xCxHxW
    """
    transform = transforms.Compose([transforms.ToPILImage()])
    return transform(tensor.squeeze(0).cpu())


if __name__ == "__main__":
    img = image_open(sys.argv[1])
    input = to_tensor(img)

    model = ic.Cluster([sys.argv[1]], 256, 256)
    model = model.to(device)
    model.eval()

    input = input.to(device)
    start = time.time()
    output, label = model(input)
    print("Cluster spend ", time.time() - start, " s.")

    start = time.time()
    mask = ic.Segment(label, 2)

    print("Segment spend ", time.time() - start, " s.")

    start = time.time()
    colormask = ic.ColorMask(mask)
    print("Color Mask spend ", time.time() - start, " s.")

    alpha = 0.2
    blend = (1.0 - alpha) * input.cpu() + alpha * colormask

    torchvision.utils.save_image(blend, "blend.jpg")

    start = time.time()
    matrix = ic.AdjMatrix(mask, 2)
    print("Adjmatrix spend ", time.time() - start, " s.")

    # pdb.set_trace()


    result = from_tensor(output)

    result.show()
