from torchvision.datasets import ImageFolder, mnist
import torch
import os

from dataset import BreastCancerDataset
from autoencoder import VAE
from classifier import EncoderClassifier
from utils import Utils, TransForms
from traintest_autoencoder import TrainTestVAE
from traintest_classifier import TrainTestCls
from config import config

if __name__ == '__main__':
    tfms = TransForms(config.img_size)
    utils = Utils()
    train_ds = ImageFolder(os.path.join('breast_cancer', 'train'))
    test_ds = ImageFolder(os.path.join('breast_cancer', 'test'))
    train_dataset = BreastCancerDataset(config, train_ds, transforms=tfms.train_tfms)
    test_dataset = BreastCancerDataset(config, test_ds, transforms=tfms.test_tfms)
    if config.mode == 'autoencoder':
        print('Start training The VAE')
        model = VAE()
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of train images: {}\nNumber of test images: {}\nNumber of model trainable parameters: {}".format(
            len(train_dataset.imgs), len(test_dataset.imgs), num_params
        ))
        train_test = TrainTestVAE(config, model, train_dataset, test_dataset, utils)
        train_test.train()

    elif config.mode == 'classification':
        print('Start training the classifier from trained VAE')
        autoencoder = VAE()
        autoencoder.load_state_dict(torch.load('checkpoints/VAE/ckpt_200.pth'))
        model = EncoderClassifier(autoencoder)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of train images: {}\nNumber of test images: {}\nNumber of model trainable parameters: {}".format(
            len(train_dataset.imgs), len(test_dataset.imgs), num_params
        ))
        train_test = TrainTestCls(config, model, train_dataset, test_dataset, utils)
        train_test.train()

    elif config.mode == 'mnist_vae':
        tfms = TransForms(config.img_size)
        utils = Utils()
        train_dataset = mnist.MNIST('mnist', transform=tfms.train_tfms, download=True)
        test_dataset = mnist.MNIST('mnist', train=False, transform=tfms.test_tfms, download=True)

        model = VAE()
        train_test = TrainTestVAE(config, model, train_dataset, test_dataset, utils)
        train_test.train()