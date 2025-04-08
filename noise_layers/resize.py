import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from noise_layers.crop import random_float


class Resize(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, ratio, interpolation_method='nearest'):
        super(Resize, self).__init__()
        self.resize_ratio_min = ratio
        self.resize_ratio_max = ratio
        self.interpolation_method = interpolation_method


    def forward(self, noised_and_cover):

        resize_ratio = random_float(self.resize_ratio_min, self.resize_ratio_max)
        ori_size = noised_and_cover.size(-1)
        noised_image = noised_and_cover
        noised_image = F.interpolate(
                                    noised_image,
                                    scale_factor=(resize_ratio, resize_ratio),
                                    mode=self.interpolation_method)

        noised_image = F.interpolate(noised_image, ori_size)

        return noised_image
