from easydict import EasyDict
import os

config = EasyDict()
config.visdom = False
config.lr = 0.001
config.batch_size = 16
config.num_epochs = 200
config.num_worker = 2
config.save_iteration = 20
config.ckpt_dir = os.path.join('checkpoints')
config.img_size = (32, 32)
config.test_iteration = 5
config.optim = 'adam'  # sgd or adam

'''AutoEncoder Setting'''
config.save_gen_images_dir = 'generated_images'
config.save_gen_images = 5  # after this epochs , save the generated images.

config.patch_size = 32

config.mode = 'classification'  # classification or autoencoder, mnist_cls, mnist_vae

os.makedirs(config.save_gen_images_dir, exist_ok=True)
