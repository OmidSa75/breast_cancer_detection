import torch
from torch.utils.data import Dataset
from PIL import Image


class BreastCancerDataset(Dataset):
    def __init__(self, args, ds, transforms):
        self.args = args
        self.imgs = ds.imgs
        self.classes = ds.classes
        self.tfms = transforms

    def __getitem__(self, index):
        img, lbl = self.imgs[index]
        patch_size = self.args.patch_size

        image = Image.open(img).convert('L')


        image = self.tfms(image)

        _, h, w = image.shape
        num_w = w // patch_size
        # pad_w = w % patch_size
        num_h = h // patch_size
        # pad_h = h % patch_size
        patch_image = torch.zeros(0, patch_size, patch_size)
        for i in range(num_h):
            for j in range(num_w):
                patch_image = torch.cat([patch_image, image[:, i * patch_size:(i + 1) * patch_size,
                                                      j * patch_size:(j + 1) * patch_size]])
        return patch_image[:64, :, :], lbl

    def __len__(self):
        return len(self.imgs)
