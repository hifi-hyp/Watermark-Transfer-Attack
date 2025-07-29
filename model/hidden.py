import numpy as np
import torch
import torch.nn as nn
import random

from options import HiDDenConfiguration
from model.discriminator import Discriminator
from model.encoder_decoder import EncoderDecoder
from vgg_loss import VGGLoss
from noise_layers.noiser import Noiser

from model.encoder import Encoder
from model.decoder import Decoder
from model.densenet import DenseNet
from model.resnet18 import ResNet

###
import sys
sys.path.append("..")

from noise_layers.identity import Identity
from noise_layers.diff_jpeg import DiffJPEG
from noise_layers.gaussian import Gaussian
from noise_layers.crop import Crop
from noise_layers.resize import Resize
from noise_layers.brightness import Brightness
from noise_layers.gaussian_blur import GaussianBlur
from noise_layers.adversarial import Adversarial
###


class Hidden:
    def __init__(self, configuration: HiDDenConfiguration, device: torch.device, noiser: Noiser, model_type: str):
        """
        :param configuration: Configuration for the net, such as the size of the input image, number of channels in the intermediate layers, etc.
        :param device: torch.device object, CPU or GPU
        :param noiser: Object representing stacked noise layers.
        :param tb_logger: Optional TensorboardX logger object, if specified -- enables Tensorboard logging
        """
        super(Hidden, self).__init__()

        # self.encoder_decoder = EncoderDecoder(configuration, noiser).to(device)
        ###
        self.encoder = Encoder(configuration).to(device)
        self.noiser_train = noiser
        self.noiser_test = Identity()
        if model_type == 'cnn':
            self.decoder = Decoder(configuration).to(device)
        elif model_type == 'resnet':
            self.decoder = ResNet(configuration).to(device)
        # self.decoder = DenseNet(configuration).to(device)
        # self.decoder = ResNet(configuration).to(device)
        ##
        self.discriminator = Discriminator(configuration).to(device)
        # self.optimizer_enc_dec = torch.optim.Adam(self.encoder_decoder.parameters())
        ###
        self.optimizer_enc = torch.optim.Adam(self.encoder.parameters())
        self.optimizer_dec = torch.optim.Adam(self.decoder.parameters())
        ###
        self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters())

        if configuration.use_vgg:
            self.vgg_loss = VGGLoss(3, 1, False)
            self.vgg_loss.to(device)
        else:
            self.vgg_loss = None

        self.config = configuration
        self.device = device

        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)

        # Defined the labels used for training the discriminator/adversarial loss
        self.cover_label = 1
        self.encoded_label = 0

        self.tb_logger = None


    def train_on_batch(self, batch: list, epoch):
        """
        Trains the network on a single batch consisting of images and messages
        :param batch: batch of training data, in the form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        images, messages = batch

        batch_size = images.shape[0]
        # self.encoder_decoder.train()
        ###
        self.encoder.train()
        self.decoder.train()
        ###
        self.discriminator.train()
        with torch.enable_grad():
            # ---------------- Train the discriminator -----------------------------
            self.optimizer_discrim.zero_grad()
            # train on cover
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover.float())
            d_loss_on_cover.backward()

            # train on fake
            # encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
            ###
            encoded_images = self.encoder(images, messages)
            if epoch >= 50:
                noise_list = [0,1,2,3,4]
                for i in range(encoded_images.shape[0]):
                    choice = random.choice(noise_list)
                    if choice == 0:
                        noise_layers = Identity()
                    elif choice == 1:
                        noise_layers = DiffJPEG(random.randint(50, 99), self.device)
                    elif choice == 2:
                        noise_layers = Gaussian(random.uniform(0, 0.1))
                    elif choice == 3:
                        # noise_layers = GaussianBlur(std=random.uniform(0, 2.0))
                        noise_layers = Crop(random.uniform(0.3, 0.7))
                    elif choice == 4:
                        noise_layers = Resize(random.uniform(0.3, 0.7))
                        # noise_layers = Brightness(random.uniform(1.0, 3))
                    # elif choice == 5:
                    #     noise_layers = Crop(random.uniform(0.8, 0.9))
                    # elif choice == 5:
                    #     noise_layers = Adversarial(self.decoder, random.uniform(0, 0.05))
                    img_noised = noise_layers(encoded_images[i:i + 1, :, :, :])
                    if i == 0:
                        noised_images = img_noised
                    else:
                        noised_images = torch.cat((noised_images, img_noised), 0)
            else:
                noised_images = self.noiser_train(encoded_images)
            decoded_messages = self.decoder(noised_images)
            ###
            d_on_encoded = self.discriminator(encoded_images.detach())
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded.float())

            d_loss_on_encoded.backward()
            self.optimizer_discrim.step()

            # --------------Train the generator (encoder-decoder) ---------------------
            # self.optimizer_enc_dec.zero_grad()
            ###
            self.optimizer_enc.zero_grad()
            self.optimizer_dec.zero_grad()
            ###
            # target label for encoded images should be 'cover', because we want to fool the discriminator
            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded.float())

            if self.vgg_loss == None:
                g_loss_enc = self.mse_loss(encoded_images, images)
            else:
                vgg_on_cov = self.vgg_loss(images)
                vgg_on_enc = self.vgg_loss(encoded_images)
                g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

            g_loss_dec = self.mse_loss(decoded_messages, messages)
            g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc \
                     + self.config.decoder_loss * g_loss_dec

            g_loss.backward()
            # self.optimizer_enc_dec.step()
            ###
            self.optimizer_enc.step()
            self.optimizer_dec.step()
            ###

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-acc    ': 1-bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item()
        }
        # return losses, (encoded_images, noised_images, decoded_messages)
        ###
        return losses, (encoded_images, decoded_messages)
        ###

    def validate_on_batch(self, batch: list, test_noiser: str):
        """
        Runs validation on a single batch of data consisting of images and messages
        :param batch: batch of validation data, in form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """

        ###
        if test_noiser == 'Identity':
            self.noiser_test = Identity()

        # if TensorboardX logging is enabled, save some of the tensors.
        if self.tb_logger is not None:
            # encoder_final = self.encoder_decoder.encoder._modules['final_layer']
            # decoder_final = self.encoder_decoder.decoder._modules['linear']
            ###
            encoder_final = self.encoder._modules['final_layer']
            decoder_final = self.decoder._modules['linear']
            ###
            self.tb_logger.add_tensor('weights/encoder_out', encoder_final.weight)
            self.tb_logger.add_tensor('weights/decoder_out', decoder_final.weight)
            discrim_final = self.discriminator._modules['linear']
            self.tb_logger.add_tensor('weights/discrim_out', discrim_final.weight)

        images, messages = batch

        batch_size = images.shape[0]

        # self.encoder_decoder.eval()
        ###
        self.encoder.eval()
        self.decoder.eval()
        ###
        self.discriminator.eval()
        with torch.no_grad():
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover.float())

            # encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
            ###
            encoded_images = self.encoder(images, messages)
            noised_images = self.noiser_test(encoded_images)
            decoded_messages = self.decoder(noised_images)
            ###

            d_on_encoded = self.discriminator(encoded_images)
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded.float())

            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded.float())

            if self.vgg_loss is None:
                g_loss_enc = self.mse_loss(encoded_images, images)
            else:
                vgg_on_cov = self.vgg_loss(images)
                vgg_on_enc = self.vgg_loss(encoded_images)
                g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

            g_loss_dec = self.mse_loss(decoded_messages, messages)
            g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc \
                     + self.config.decoder_loss * g_loss_dec

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-acc    ': 1-bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item()
        }
        return losses, (encoded_images, noised_images, decoded_messages)

    def to_stirng(self):
        # return '{}\n{}'.format(str(self.encoder_decoder), str(self.discriminator))
        ###
        return '{}\n{}\n{}'.format(str(self.encoder), str(self.decoder), str(self.discriminator))
        ###
