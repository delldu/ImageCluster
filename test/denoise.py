# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, , 2018-10-17 01:50:34
# ***
# ************************************************************************************/

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import sys
import glob
import os
from tqdm import tqdm
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def gauss_conv(size=5, channels=3):
    """Create Gaussian Kernel, default is 5x5."""
    assert size % 2 == 1, "size should be odd integer."
    n = size // 2  # n -- radius
    sigma = 1.0 * n / 3.0  # 3*sigma == window radius
    delta = 1.0 / (2.0 * sigma * sigma)
    weight = torch.FloatTensor([-i * i * delta for i in range(-n, n + 1)]).exp().view(2 * n + 1, 1)
    weight = weight.mm(weight.t())
    weight /= weight.sum().item()
    m = nn.Conv2d(channels, channels, kernel_size=2 * n + 1, groups=channels, padding=n, bias=False)
    m.weight.data[:, :] = weight
    return m


# def box_kernel(size):
#     """Create Box Kernel, for an example: 3 means 3x3."""
#     assert size % 2 == 1, "size should be odd integer."
#     n = size // 2  # n -- radius
#     weight = torch.ones(size, size)
#     weight /= weight.sum().item()
#     m = nn.Conv2d(3, 3, kernel_size=2 * n + 1, groups=3, padding=n, bias=False)
#     m.weight.data[:, :] = weight
#     return m


def image_totensor(image):
    """PIL image [0, 255] to 1xCxHxW tensor, data range[0.0, 1.0]."""
    return transforms.ToTensor()(image).unsqueeze(0)


def tensor_toimage(tensor):
    """Tensor 1xCxHxW, data range[0.0, 1.0] ==> PIL image, [0, 255]."""
    return transforms.ToTensor()(tensor.squeeze(0).cpu())


def crop_patch(tensor, row, col, size):
    """Crop patch from tensor, 1xHxW."""
    radius = size // 2
    beg_row, end_row = row - radius, row + radius
    beg_col, end_col = col - radius, col + radius

    return tensor[:, beg_row:end_row + 1, beg_col:end_col + 1].clone()


def denoise(image):
    """Denoise with good similar patches."""
    # return image

    # Define parameters
    patch_size = 3
    similar_topk = 4
    conv_func = gauss_conv(size=patch_size, channels=1).cuda()
    dist_func = nn.CosineSimilarity().cuda()

    # Main functions ...
    half_size = patch_size // 2
    tensor = image_totensor(image.convert('L')).cuda()
    tensor = conv_func(tensor).squeeze(0)  # CxHxW, here C == 1

    height, width = tensor.size(1), tensor.size(2)
    for i in range(half_size, height - half_size):
        print("i ---- = ", i)
        for j in range(half_size, width - half_size):
            current_patch = crop_patch(tensor, i, j, patch_size)

            currrent_value = tensor[0][i][j].item()
            # Topk Most Similar
            weight, position = tensor.sub(currrent_value).abs().view(-1).topk(
                similar_topk, largest=False)
            weight = weight.tolist()
            position = position.tolist()
            # First Normal Weight
            weight = [w / max(w, currrent_value) for w in weight]
            rows = [pos // width for pos in position]
            cols = [pos % width for pos in position]

            for index in range(len(weight)):
                row = rows[index]
                col = cols[index]
                if row < half_size or row >= height - half_size or col < half_size or col >= width - half_size:
                    weight[index] = 0.0
                    continue

                patch = crop_patch(tensor, row, col, patch_size)
                # pdb.set_trace()
                distance = dist_func(current_patch.view(1, -1), patch.view(1, -1))
                # pdb.set_trace()
                weight[index] *= distance.item()
            # Second Normal Weight
            total_weight = sum(weight)
            if total_weight > 1e-6:
                weight = [w / total_weight for w in weight]

            for index in range(len(weight)):
                row = rows[index]
                col = cols[index]
                # color element replace ...
    return image


if __name__ == "__main__":
    files = sorted(glob.glob(sys.argv[1] + "/*"))

    with tqdm(total=len(files)) as t:
        for src in files:
            dst = "tmp/" + os.path.basename(src)

            img = Image.open(src).convert('RGB')
            output = denoise(img)

            output.save(dst)

            t.update(1)
