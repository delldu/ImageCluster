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


img = image_open("girl.jpg")
input = to_tensor(img)

model = ic.Cluster(["girl.jpg"], 128, 256)
model = model.to(device)

input = input.to(device)
start = time.time()


for i in range(1000):
    output, _= model(input)

print("Time spend ", time.time() - start, " s for 1000 times.")
result = from_tensor(output)

result.show()
