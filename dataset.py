import torchvision
import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import zipfile
# print("Worker here")


class ImageDataset(Dataset):
    """Image dataset ."""

    def __init__(self, image_zip, image_extensions, transform=None):
        """
        Arguments:
            pillow_images (list): a list of pillow Image objects (matching image_filenames)
            image_filenames (list): a list of image filenames.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # pillow_images = [Image.open(imgzip.open(image_filename)) for image_filename in name_list]
        imgzip = zipfile.ZipFile(image_zip)

        self.imgzip = imgzip
        self.name_list = [name for name in imgzip.namelist(
        ) if name.endswith(image_extensions)]
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        #! figure out indexing multiple?

        # print("TAKING INDEX", idx)
        img = Image.open(self.imgzip.open(self.name_list[idx]))

        # Load an image
        processed_img = img.convert("RGB")  # Ensure it's in RGB mode

        # Preprocess the image
        if self.transform:
            processed_img = self.transform(processed_img)

        return processed_img, self.name_list[idx]
