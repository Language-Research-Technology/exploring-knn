import torchvision
import os
from PIL import Image
from torch.utils.data import Dataset
import torch
# print("Worker here")


class ImageDataset(Dataset):
    """Image dataset ."""

    def __init__(self, image_filenames, root_dir, transform=None):
        """
        Arguments:
            image_filenames (list): a list of image filenames.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.filenames = image_filenames
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        #! figure out indexing multiple?

        print("TAKING INDEX", idx)

        with Image.open(os.path.join(self.root_dir,
                                     self.filenames[idx])) as img:

            # Load an image
            processed_img = img.convert("RGB")  # Ensure it's in RGB mode

            # Preprocess the image
            if self.transform:
                processed_img = self.transform(processed_img)

        return processed_img, self.filenames[idx]
