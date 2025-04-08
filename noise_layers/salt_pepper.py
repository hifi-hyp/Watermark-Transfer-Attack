import torch.nn as nn
import random
import numpy as np
import torch


#
class Salt_Pepper(nn.Module):
    """"
    The matrix of (h,w,1) is masked with certain probability using three numbers 0, 1, 2: mask 0 (original image) with probability signal_pct, mask 1 (salt noise) with probability noise_pct/2. and mask 2 (pepper noise) with probability noise_pct/2.
    """

    def __init__(self):
        super(Salt_Pepper, self).__init__()
        # pepper and salt
        self.snr = 0.9
        self.p = 1.0
        # self.device = device

    def forward(self, noised_and_cover):
        encoded = noised_and_cover[0]
        if random.uniform(0, 1) < self.p:
            #
            b, c, h, w = encoded.shape
            signal_pct = self.snr
            noise_pct = 1 - self.snr
            mask = torch.Tensor(np.random.choice((0, 1, 2), size=(b, 1, h, w), p=[signal_pct, noise_pct/2., noise_pct/2.])).cuda()
            mask = mask.repeat(1, c, 1, 1)
            #

            noise = torch.zeros((b, c, h, w)).cuda()
            noise[mask == 1] = 1
            noise[mask == 2] = -1
            encoded = encoded + noise
            encoded = torch.clamp(encoded, 0, 1)

            # encoded[mask == 1] = 1      # salt
            # encoded[mask == 2] = 0     # pepper
            noised_and_cover[0] = encoded

            return noised_and_cover
        else:
            print('salt_pepper error!')
            exit()