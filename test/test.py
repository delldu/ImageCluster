# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, , 2018-10-17 01:50:34
# ***
# ************************************************************************************/

import torch
from PIL import Image
from torchvision import transforms
import imagecluster as ic

import time
import sys
import pdb

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


def from_tensor(tensor):
    """
    tensor format: 1xCxHxW
    """
    transform = transforms.Compose([transforms.ToPILImage()])
    return transform(tensor.squeeze(0).cpu())


if __name__ == "__main__":
    img = image_open(sys.argv[1])
    input = to_tensor(img)

    model = ic.Cluster([sys.argv[1]], 128, 256)
    model = model.to(device)

    input = input.to(device)
    start = time.time()
    output, label = model(input)
    print("Cluster spend ", time.time() - start, " s.")

    start = time.time()
    segment_result = model.segment(label, 2)
    print("Segment spend ", time.time() - start, " s.")

    maskimg = ic.Blend(img, segment_result, 0.1)
    maskimg.show()


    result = from_tensor(output)

    result.show()
