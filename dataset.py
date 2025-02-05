import torchvision
import os
from PIL import Image
from torch.utils.data import Dataset
import torch
# print("Worker here")


class ImageDataset(Dataset):
    """Image dataset ."""

    def __init__(self, pillow_images, image_filenames, transform=None):
        """
        Arguments:
            pillow_images (list): a list of pillow Image objects (matching image_filenames)
            image_filenames (list): a list of image filenames.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pillow_images = pillow_images
        self.filenames = image_filenames
        self.transform = transform

    def __len__(self):
        return len(self.pillow_images)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        #! figure out indexing multiple?

        # print("TAKING INDEX", idx)
        img = self.pillow_images[idx]

        # Load an image
        processed_img = img.convert("RGB")  # Ensure it's in RGB mode

        # Preprocess the image
        if self.transform:
            processed_img = self.transform(processed_img)

        return processed_img, self.filenames[idx]
