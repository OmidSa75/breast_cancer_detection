from torchvision.datasets import ImageFolder
import os
from dataset import BreastCancerDataset
from autoencoder import VAE
from utils import Utils, TransForms
from traintest_autoencoder import TrainTest
from config import config

if __name__ == '__main__':
    tfms = TransForms(config.img_size)
    utils = Utils()
    train_ds = ImageFolder(os.path.join('breast_cancer', 'train'))
    test_ds = ImageFolder(os.path.join('breast_cancer', 'test'))
    train_dataset = BreastCancerDataset(config, train_ds, transforms=tfms.train_tfms)
    sample = train_dataset[1]
    test_dataset = BreastCancerDataset(config, test_ds, transforms=tfms.test_tfms)
    model = VAE()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of train images: {}\nNumber of test images: {}\nNumber of model trainable parameters: {}".format(
        len(train_dataset.imgs), len(test_dataset.imgs), num_params
    ))
    train_test = TrainTest(config, model, train_dataset, test_dataset, utils)
    train_test.train()
