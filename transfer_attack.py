from model.hidden import Hidden
from model.stega import stegamodel
from noise_layers.noiser import Noiser
import torch
import argparse
from noise_argparser import NoiseArgParser
from options import *
import os
import utils
from attack_func import test_tfattk_hidden
from attack_theory_flip_all import test_tfattk_DB_theory
import logging
import torchvision.transforms as transforms


def main():
    '''
    ========================
    set all parameters here!
    ========================
    '''
    device = torch.device('cuda:5') if torch.cuda.is_available() else torch.device('cpu')
    # device = 'cuda:' + str(device)
    seed = 42
    data_dir = ''
    batch_size = 100
    epochs = 200
    num_models = 10
    name = '30bits_AT_DALLE_200epochs_50maintrain'
    size = 128
    message = 64
    train_dataset = 'large_random_10k'
    val_dataset = 'large_random_1k'
    tensorboard = False
    enable_fp16 = False
    noise = None
    train_type = 'AT'
    data_name = 'DB'
    wm_method = 'hidden'
    target = 'hidden'
    model_type = 'resnet'
    white = False
    smooth = False



    start_epoch = 1
    train_options = TrainingOptions(
        batch_size=batch_size,
        number_of_epochs=epochs,
        train_folder=os.path.join(data_dir, 'train'),
        validation_folder=os.path.join(data_dir, 'val'),
        runs_folder=os.path.join('.', 'runs'),
        start_epoch=start_epoch,
        experiment_name=name)

    noise_config = noise if noise is not None else []
    if wm_method == 'hidden':
        sur_config = HiDDenConfiguration(H=size, W=size,
                                            message_length=message,
                                            encoder_blocks=4, encoder_channels=64,
                                            decoder_blocks=7, decoder_channels=64,
                                            use_discriminator=True,
                                            use_vgg=False,
                                            discriminator_blocks=3, discriminator_channels=64,
                                            decoder_loss=1,
                                            encoder_loss=0.7,
                                            adversarial_loss=1e-3,
                                            enable_fp16=enable_fp16
                                            )

    if target == 'hidden':
        if model_type == 'cnn':
            target_config = HiDDenConfiguration(H=size, W=size,
                                                message_length=message,
                                                encoder_blocks=4, encoder_channels=64,
                                                decoder_blocks=7, decoder_channels=64,
                                                use_discriminator=True,
                                                use_vgg=False,
                                                discriminator_blocks=3, discriminator_channels=64,
                                                decoder_loss=1,
                                                encoder_loss=0.7,
                                                adversarial_loss=1e-3,
                                                enable_fp16=enable_fp16
                                                )
        elif model_type == 'resnet':
            target_config = HiDDenConfiguration(H=size, W=size,
                                                message_length=message,
                                                encoder_blocks=7, encoder_channels=64,
                                                decoder_blocks=7, decoder_channels=64,
                                                use_discriminator=True,
                                                use_vgg=False,
                                                discriminator_blocks=3, discriminator_channels=64,
                                                decoder_loss=1,
                                                encoder_loss=0.7,
                                                adversarial_loss=1e-3,
                                                enable_fp16=enable_fp16
                                                )

    # Model
    noiser = Noiser(noise_config, device)
    model = Hidden(target_config, device, noiser, model_type)

    if 'DB' in data_name:
        if target == 'hidden':
            target_cp_file = './target model/' + target + '_' + train_type + '/' + str(message) + 'bits_' + model_type + '_' + train_type + '.pth'
    elif 'midjourney' in data_name:
        if target == 'hidden':
            target_cp_file = './target model/' + str(message) + 'bits_' + model_type + '_AT_midjourney.pth'

    sur_cp_folder = './surrogate model/' + wm_method + '/' + train_type + '/'

    sur_cp_list = []
    for idx in range(num_models):
        sur_cp_list.append(sur_cp_folder + 'model_' + str(idx + 1) + '.pth')
    
    if target == 'hidden':
        target_cp = torch.load(target_cp_file, map_location='cpu')
        utils.model_from_checkpoint(model, target_cp)

    sur_model_list = []
    for idx in range(len(sur_cp_list)):
        if wm_method == 'hidden':
            sur_model_list.append(Hidden(sur_config, device, noiser, 'cnn'))
        if white:
            linear = torch.nn.Linear(30, 30, bias=True)
            sur_model_list[idx].decoder = torch.nn.Sequential(sur_model_list[idx].decoder, linear.to(device))
            cp = torch.load(sur_cp_list[idx].replace(".pth", "_whit.pth"), map_location='cpu')
        else:
            cp = torch.load(sur_cp_list[idx], map_location='cpu')
        utils.model_from_checkpoint(sur_model_list[idx], cp)

    test_tfattk_hidden(model, sur_model_list, device, target_config, train_options, val_dataset, train_type, model_type, attk_param, data_name, pp, wm_method, target, smooth)




if __name__ == '__main__':
    utils.setup_seed(42)
    main()
