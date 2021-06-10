from easydict import EasyDict
import os

config = EasyDict()
config.lr = 0.01
config.batch_size = 16
config.num_epochs = 200
config.num_worker = 4
config.save_iteration = 20
config.ckpt_dir = os.path.join('checkpoints')
config.img_size = (230, 350)
config.test_iteration = 5

'''AutoEncoder Setting'''
config.save_gen_images_dir = 'generated_images'
config.save_gen_images = 10  # after this epochs , save the generated images.

config.patch_size = 32

config.mode = 'classification'  # classification or autoencoder

os.makedirs(config.save_gen_images_dir, exist_ok=True)
