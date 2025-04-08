import torch
import torch.nn as nn
import numpy as np


def project(param_data, backup, epsilon):
    r = param_data - backup
    r = epsilon * r
    return backup + r

class Adversarial(nn.Module):

    def __init__(self, Decoder, bound):
        super(Adversarial, self).__init__()
        self.Decoder = Decoder
        self.bound = bound

    def forward(self, container_img):
        per_bound = self.bound
        lr = 0.1
        epsilon_prime = 0.01
        criterion = nn.MSELoss().cuda()

        # Random
        random_message = np.random.choice([0, 1], (container_img.shape[0], 30))
        target_message = torch.from_numpy(random_message).cuda().float()
        container_img_cloned = container_img.clone()

        for i in range(100):

            container_img = container_img.requires_grad_(True)
            min_value, max_value = torch.min(container_img), torch.max(container_img)
            decoded_message = self.Decoder(container_img)

            loss = criterion(decoded_message, target_message)
            grads = torch.autograd.grad(loss, container_img)

            with torch.no_grad():
                container_img = container_img - lr * grads[0]
                container_img = torch.clamp(container_img, min_value.item(), max_value.item())

            # Projection
            perturbation_norm = torch.norm(container_img - container_img_cloned, float('inf'))
            if perturbation_norm.cpu().detach().numpy() >= per_bound:
                epsilon = per_bound / perturbation_norm
                container_img = project(container_img, container_img_cloned, epsilon)
                break

            # Early Stopping
            decoded_message = self.Decoder(container_img)
            decoded_rounded = decoded_message.detach().cpu().numpy().round().clip(0, 1)
            bit_acc = 1 - np.sum(np.abs(decoded_rounded - target_message.cpu().numpy())) / (container_img.shape[0] * 30)
            if bit_acc >= 1 - epsilon_prime:
                break

        return container_img