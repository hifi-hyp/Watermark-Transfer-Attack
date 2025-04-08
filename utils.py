import numpy as np
import os
import re
import csv
import time
import pickle
import logging
import random

# DiffusionDB is implemented by datasets lib
import datasets
from datasets import load_dataset

import torch
from torchvision import datasets, transforms
import torchvision.utils
from torch.utils import data
import torch.nn.functional as F

from options import HiDDenConfiguration, TrainingOptions
from model.hidden import Hidden


'''
add new datasets for DiffusionDB
'''
def get_data_loaders_DB(hidden_config: HiDDenConfiguration, train_options: TrainingOptions, dataset: str, train: bool):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((hidden_config.H, hidden_config.W), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop((hidden_config.H, hidden_config.W)),
            # transforms.Resize((hidden_config.H, hidden_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    # print(dataset)
    data = load_dataset('poloclub/diffusiondb', dataset, split='train', cache_dir='./data/DB_image_sample')
    if train:
        images = data.map(lambda item: {'image': data_transforms['train'](item['image'])})
    else:
        images = data.map(lambda item: {'image': data_transforms['test'](item['image'])})

    images.set_format(type='torch', columns=['image', 'prompt'])

    data_loader = torch.utils.data.DataLoader(images, batch_size=train_options.batch_size,
                                              shuffle=train, num_workers=4)
    return data_loader

def get_data_loaders_DALLE(hidden_config: HiDDenConfiguration, train_options: TrainingOptions, train: bool, idx=None):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((hidden_config.H, hidden_config.W), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop((hidden_config.H, hidden_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    if train:
        dataset = datasets.ImageFolder('./data/DALLE2/train', data_transforms['train'])
        indices = torch.load('./data/DALLE2/indice/indices_seed_' + str(idx) + '.pt')
        dataset = torch.utils.data.Subset(dataset, indices)
    else:
        dataset = datasets.ImageFolder('./data/DALLE2/val', data_transforms['test'])

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=train_options.batch_size,
                                              shuffle=train, num_workers=4)
    
    return data_loader

def get_data_loaders_midjourney(hidden_config: HiDDenConfiguration,train_options:TrainingOptions,dataset:str,train:bool):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((hidden_config.H, hidden_config.W), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop((hidden_config.H, hidden_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }
    if train:
        train_images = datasets.ImageFolder('./data/midjourney/train', data_transforms['train'])
    else:
        train_images = datasets.ImageFolder('./data/midjourney/val', data_transforms['test'])

    data_loader = torch.utils.data.DataLoader(train_images, batch_size=train_options.batch_size,
                                                    shuffle=train, num_workers=4)
    return data_loader


def get_data_loaders_nlb(hidden_config: HiDDenConfiguration,train_options:TrainingOptions,dataset:str):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((hidden_config.H, hidden_config.W), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_images = datasets.ImageFolder('./data/nlb_mj_image_128/small_dataset', data_transforms['test'])

    data_loader = torch.utils.data.DataLoader(train_images, batch_size=train_options.batch_size,
                                                    shuffle=False, num_workers=4)
    return data_loader


def get_data_loaders_stablesign(model_config: HiDDenConfiguration,train_options:TrainingOptions):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {
        'test': transforms.Compose([
            # transforms.CenterCrop((256, 256)),
            transforms.Resize((model_config.H, model_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_images = datasets.ImageFolder('/home/yh351/code/stable_signature/comp_image_w_encoder_imagenet_final_1000/wm_images', data_transforms['test'])

    data_loader = torch.utils.data.DataLoader(train_images, batch_size=train_options.batch_size,
                                                    shuffle=False, num_workers=4)
    return data_loader


def get_data_loaders_treering(model_config: HiDDenConfiguration,train_options:TrainingOptions):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {
        'test': transforms.Compose([
            transforms.CenterCrop((model_config.H, model_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_images = datasets.ImageFolder('/home/yh351/code/tree-ring-watermark-main/wm_images', data_transforms['test'])

    data_loader = torch.utils.data.DataLoader(train_images, batch_size=train_options.batch_size,
                                                    shuffle=False, num_workers=4)
    return data_loader


def image_to_tensor(image):
    """
    Transforms a numpy-image into torch tensor
    :param image: (batch_size x height x width x channels) uint8 array
    :return: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    """
    image_tensor = torch.Tensor(image)
    image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.permute(0, 3, 1, 2)
    image_tensor = image_tensor / 127.5 - 1
    return image_tensor


def tensor_to_image(tensor):
    """
    Transforms a torch tensor into numpy uint8 array (image)
    :param tensor: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    :return: (batch_size x height x width x channels) uint8 array
    """
    image = tensor.permute(0, 2, 3, 1).cpu().numpy()
    image = (image + 1) * 127.5
    return np.clip(image, 0, 255).astype(np.uint8)


def save_images(original_images, watermarked_images, epoch, folder, resize_to=None):
    images = original_images[:original_images.shape[0], :, :, :].cpu()
    watermarked_images = watermarked_images[:watermarked_images.shape[0], :, :, :].cpu()

    # scale values to range [0, 1] from original range of [-1, 1]
    images = (images + 1) / 2
    watermarked_images = (watermarked_images + 1) / 2

    if resize_to is not None:
        images = F.interpolate(images, size=resize_to)
        watermarked_images = F.interpolate(watermarked_images, size=resize_to)

    stacked_images = torch.cat([images, watermarked_images], dim=0)
    filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))
    torchvision.utils.save_image(stacked_images, filename, original_images.shape[0], normalize=False)


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def last_checkpoint_from_folder(folder: str):
    last_file = sorted_nicely(os.listdir(folder))[-1]
    last_file = os.path.join(folder, last_file)
    return last_file


def save_checkpoint(model: Hidden, experiment_name: str, epoch: int, checkpoint_folder: str):
    """ Saves a checkpoint at the end of an epoch. """
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint_filename = f'epoch-{epoch}.pth'
    checkpoint_filename = os.path.join(checkpoint_folder, checkpoint_filename)
    logging.info('Saving checkpoint to {}'.format(checkpoint_filename))
    checkpoint = {
        # 'enc-dec-model': model.encoder_decoder.state_dict(),
        # 'enc-dec-optim': model.optimizer_enc_dec.state_dict(),
        ###
        'enc-model': model.encoder.state_dict(),
        'dec-model': model.decoder.state_dict(),
        'enc-optim': model.optimizer_enc.state_dict(),
        'dec-optim': model.optimizer_dec.state_dict(),
        ###
        'discrim-model': model.discriminator.state_dict(),
        'discrim-optim': model.optimizer_discrim.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, checkpoint_filename)
    logging.info('Saving checkpoint done.')


# def load_checkpoint(hidden_net: Hidden, options: Options, this_run_folder: str):
def load_last_checkpoint(checkpoint_folder):
    """ Load the last checkpoint from the given folder """
    last_checkpoint_file = last_checkpoint_from_folder(checkpoint_folder)
    checkpoint = torch.load(last_checkpoint_file)

    return checkpoint, last_checkpoint_file


def load_specific_checkpoint(checkpoint_folder, epoch):
    """ Load the last checkpoint from the given folder """
    checkpoint = torch.load(os.path.join(checkpoint_folder, 'epoch-' + str(epoch) + '.pth'))

    return checkpoint


def model_from_checkpoint(hidden_net, checkpoint):
    """ Restores the hidden_net object from a checkpoint object """
    # hidden_net.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'])
    # hidden_net.optimizer_enc_dec.load_state_dict(checkpoint['enc-dec-optim'])
    ###
    hidden_net.encoder.load_state_dict(checkpoint['enc-model'])
    hidden_net.decoder.load_state_dict(checkpoint['dec-model'])
    hidden_net.optimizer_enc.load_state_dict(checkpoint['enc-optim'])
    hidden_net.optimizer_dec.load_state_dict(checkpoint['dec-optim'])
    ###
    hidden_net.discriminator.load_state_dict(checkpoint['discrim-model'])
    hidden_net.optimizer_discrim.load_state_dict(checkpoint['discrim-optim'])


def load_options(options_file_name) -> (TrainingOptions, HiDDenConfiguration, dict):
    """ Loads the training, model, and noise configurations from the given folder """
    with open(os.path.join(options_file_name), 'rb') as f:
        train_options = pickle.load(f)
        noise_config = pickle.load(f)
        hidden_config = pickle.load(f)
        # for backward-capability. Some models were trained and saved before .enable_fp16 was added
        if not hasattr(hidden_config, 'enable_fp16'):
            setattr(hidden_config, 'enable_fp16', False)

    return train_options, hidden_config, noise_config


def get_data_loaders(hidden_config: HiDDenConfiguration, train_options: TrainingOptions):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((hidden_config.H, hidden_config.W), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop((hidden_config.H, hidden_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    # print(train_options.train_folder)
    train_images = datasets.ImageFolder(train_options.train_folder, data_transforms['train'])
    # train_images = datasets.ImageFolder('./data/train', data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=train_options.batch_size, shuffle=True,
                                               num_workers=4)

    validation_images = datasets.ImageFolder(train_options.validation_folder, data_transforms['test'])
    # validation_images = datasets.ImageFolder('./data/val', data_transforms['test'])
    validation_loader = torch.utils.data.DataLoader(validation_images, batch_size=train_options.batch_size,
                                                    shuffle=False, num_workers=4)

    return train_loader, validation_loader


def log_progress(losses_accu):
    log_print_helper(losses_accu, logging.info)


def print_progress(losses_accu):
    log_print_helper(losses_accu, print)


def log_print_helper(losses_accu, log_or_print_func):
    max_len = max([len(loss_name) for loss_name in losses_accu])
    for loss_name, loss_value in losses_accu.items():
        log_or_print_func(loss_name.ljust(max_len + 4) + '{:.4f}'.format(loss_value.avg))


def create_folder_for_run(runs_folder, experiment_name):
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)

    this_run_folder = os.path.join(runs_folder, f'{experiment_name} {time.strftime("%Y.%m.%d--%H-%M-%S")}')

    os.makedirs(this_run_folder)
    os.makedirs(os.path.join(this_run_folder, 'checkpoints'))
    os.makedirs(os.path.join(this_run_folder, 'images'))

    return this_run_folder


def write_losses(file_name, losses_accu, epoch, duration):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()] + ['duration']
            writer.writerow(row_to_write)
        row_to_write = [epoch] + ['{:.4f}'.format(loss_avg.avg) for loss_avg in losses_accu.values()] + [
            '{:.0f}'.format(duration)]
        writer.writerow(row_to_write)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def transform_image(image, device):
    # For HiDDeN watermarking method, image pixel value range should be [-1, 1]. Transform an image into [-1, 1] range.
    cloned_encoded_images = (image + 1) / 2  # for HiDDeN watermarking method only
    cloned_encoded_images = cloned_encoded_images.mul(255).clamp_(0, 255)

    cloned_encoded_images = cloned_encoded_images / 255
    cloned_encoded_images = cloned_encoded_images * 2 - 1  # for HiDDeN watermarking method only
    image = cloned_encoded_images.to(device)

    return image


def transform_image_stablesign(image, device):
    # For HiDDeN watermarking method, image pixel value range should be [-1, 1]. Transform an image into [-1, 1] range.
    cloned_encoded_images = (image + 1) / 2  # for HiDDeN watermarking method only
    cloned_encoded_images = cloned_encoded_images.mul(255).clamp_(0, 255)

    cloned_encoded_images = cloned_encoded_images / 255
    data_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    cloned_encoded_images = data_transforms(cloned_encoded_images)
    image = cloned_encoded_images.to(device)

    return image


def str2msg(str):
    return [1 if el=='1' else 0 for el in str]