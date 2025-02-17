import torchvision
import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import zipfile
from typing import Tuple, Optional, Callable, Union


class ImageDataset(Dataset):
    """
    A PyTorch Dataset for loading images from a zip archive.

    This dataset reads images from a zip file, applies optional transformations,
    and returns the processed images along with their filenames.

    Attributes:
        imgzip (zipfile.ZipFile): The opened zip file containing images.
        name_list (List[str]): A list of image filenames that match the given extensions
                               and are not ignored.
        transform (Optional[Callable]): An optional transformation function to apply to images.
    """

    def __init__(self,
                 image_zip: str,
                 image_extensions: Tuple[str, ...],
                 image_ignore: Tuple[str, ...],
                 transform: Optional[Callable] = None):
        """
        Initializes the ImageDataset.

        Args:
            image_zip (str): Path to the zip file containing images.
            image_extensions (Tuple[str, ...]): A tuple of allowed image file extensions
                                                (e.g., ('jpg', 'png')).
            image_ignore (Tuple[str, ...]): A tuple of filename prefixes to exclude from processing.
            transform (Optional[Callable], optional): A transformation function to apply to
                                                      each image (e.g., torchvision transforms).
                                                      Defaults to None.
        """
        # pillow_images = [Image.open(imgzip.open(image_filename)) for image_filename in name_list]
        with zipfile.ZipFile(image_zip) as imgzip:

            self.imgzip = imgzip
            self.name_list = [name for name in imgzip.namelist(
            ) if name.endswith(image_extensions) and not name.split("/")[-1].startswith(image_ignore)]
        self.transform = transform

    def __len__(self) -> int:
        """
        Returns:
            int: The total number of valid images in the dataset.
        """
        return len(self.name_list)

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> Tuple[Image.Image, str]:
        """
        Retrieves an image and its filename from the dataset.

        Args:
            idx (Union[int, torch.Tensor]): Index of the image in the dataset. 
                                            If it's a tensor, it will be converted to a list.

        Returns:
            Tuple[Image.Image, str]: A tuple containing:
                - The processed image (as a PIL Image or transformed version).
                - The corresponding filename.
        """
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
