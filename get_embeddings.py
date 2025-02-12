import zipfile
from PIL import Image  # Pillow

from concurrent.futures import ThreadPoolExecutor  # for parallel processing
from tqdm import tqdm  # for progress bars

import numpy as np

import torch
# Clip ## # Meta clip
# ensure using huggingface-hub v0.25.0 or earlier to prevent import issue
from imgbeddings import imgbeddings
from torch.utils.data import DataLoader
import torchvision
from typing import Tuple, Callable


### Basic no batch embedding ###
#! use scikit image for sift
# def get_descriptors_opencv(f=cv.SIFT_create()):
#     # https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format

#     def embed(img: "pillow image"):
#         open_cv_img = np.array(img.convert('RGB'))[:, :, ::-1].copy()

#         # Use sift as surf patented
#         # surf = cv.SURF(400) #? add way to change paramts
#         _, des = f.detectAndCompute(open_cv_img,None)
#         #! add clustering on the outputs (do single layer feature network first)

#         # Convert RGB to BGR
#         return np.array(des).flatten()

#     return embed

def get_clip_embedding(image_zip: str, image_extensions: tuple[str], batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """ 
    A function to get CLIP embeddings in batches.

    Params:
        - `image_zip`: Path to the zip file containing all images, e.g., "./images.zip".
        - `image_extensions`: A tuple of image file extensions you want to include (e.g., ('jpg',)).
        - `batch_size`: Integer size of the image batches.

    Returns:
        - `embedding`: A NumPy array of CLIP embeddings where the ith row corresponds to 
          the ith image_filename in `image_filenames`.
        - `image_filenames`: A NumPy array of image filenames.
    """
    # without batching
    # imgzip = zipfile.ZipFile(image_zip)
    # # imgzip.namelist()
    # name_list = [name for name in imgzip.namelist(
    # ) if name.endswith(image_extensions)]
    # # pillow_images = [Image.open(imgzip.open(image_filename))
    # #                  for image_filename in name_list]
    # # # want to feed it in batches
    # # # images = [Image.open(os.path.join(IMAGE_FOLDER, image_filename)) for image_filename in image_filenames] # try as generator
    # # # for
    # del imgzip
    # ibed = imgbeddings()
    # embedding = ibed.to_embeddings(name_list)
    # # del images # save memory
    # return np.array(embedding), np.array(name_list)

    # take 2 with batching

    imgzip = zipfile.ZipFile(image_zip)
    # imgzip.namelist()
    name_list = [name for name in imgzip.namelist(
    ) if name.endswith(image_extensions)]

    ibed = imgbeddings()
    # do first one
    name_batch = name_list[0:batch_size]
    pillow_batch = [Image.open(imgzip.open(image_filename))
                    for image_filename in name_batch]
    embeddings = np.array(ibed.to_embeddings(pillow_batch))

    for i in tqdm(range(batch_size, len(name_list), batch_size)):
        name_batch = name_list[i:i+batch_size]
        pillow_batch = [Image.open(imgzip.open(image_filename))
                        for image_filename in name_batch]
        embedding = ibed.to_embeddings(pillow_batch)
        # print(i, len(name_batch), len(pillow_batch), len(embedding), len(embeddings))
        embeddings = np.concatenate((embeddings, embedding))
        del name_batch, pillow_batch, embedding

    return embeddings, np.array(name_list)


#! look into different colour spaces with luminance
#! integrate scikit image
def get_normalised_histogram(img: Image.Image) -> np.ndarray:
    """ 
    Returns a pixel-standardized Pillow color histogram array from a Pillow image.

    Params:
        - `img`: A Pillow image object.

    Returns:
        - `np.ndarray`: A NumPy array representing the normalized histogram of the image.
    """
    w, h = img.size
    pixels = w*h
    return np.array(img.histogram())/pixels

# embedding_function takes a batch of images, get_embedding takes batch_size=


def get_embedding(image_zip: str,
                  image_extensions: Tuple[str],
                  embedding_function: Callable[[
                      Image.Image], np.ndarray] = get_normalised_histogram
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """ 
    A function to get embeddings using a user-defined embedding function.

    This method does not work well for intensive embeddings that benefit from batching.

    Params:
        - `image_zip`: Path to the zip file containing all images (e.g., "./images.zip").
        - `image_extensions`: A tuple containing image file extensions you want to include (e.g., ('jpg',)).
        - `embedding_function`: A function which takes in a Pillow image and returns a 1D array of embeddings.

    Returns:
        - `embedding`: A NumPy array of embeddings where the ith row corresponds to 
          the ith image_filename in `image_filenames`.
        - `image_filenames`: A NumPy array of image filenames.
    """
    imgzip = zipfile.ZipFile(image_zip)

    name_list = [name for name in imgzip.namelist(
    ) if name.endswith(image_extensions)]

    def process_image(image_filename):

        with Image.open(imgzip.open(image_filename)) as img:
            # add the embedding of the image to the given function
            return embedding_function(img)

    with ThreadPoolExecutor() as executor:
        embedding = list(
            tqdm(executor.map(process_image, name_list), total=len(name_list)))

    # tqdm does not work without the list()
    return np.array(embedding), np.array(name_list)


### Batch embedding ###

# = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to("cpu")):
def get_embeddings_pytorch(dataloader: DataLoader, pytorch_model: torchvision.models) -> tuple[np.ndarray, np.ndarray]:
    """ 
    A function to get embeddings on a PyTorch dataset from a PyTorch model, 
    processes the data in batches as specified by the dataloader.

    Params:
        - `dataloader`: A PyTorch DataLoader object that provides batches of images and corresponding filenames.
        - `pytorch_model`: A PyTorch model used to compute the embeddings (or predictions).

    Returns:
        - `embedding`: A NumPy array of embeddings, where each row corresponds to 
          the embeddings of an individual image in the batch.
        - `image_filenames`: A NumPy array of filenames corresponding to the images in the batches.
    """

    pytorch_model.eval()  # optimization (disable unneeded things)
    torch.set_grad_enabled(False)  # optimization (disable unneeded things)

    batch_embeddings = []  # Using a list to hold embeddings for each image
    batch_filenames = []

    total_batches = len(dataloader)
    print(f"Total batches to process: {total_batches}")

    for i_batch, (batch, filenames) in enumerate(dataloader):
        # show progress of batches
        print(f"Processing batch {i_batch + 1}/{total_batches}...")

        with torch.no_grad():  # Disable gradient calculations for inference
            output = pytorch_model(batch)

        # The output is logits; apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)

        # Convert to numpy array
        probabilities_numpy = probabilities.detach().cpu().numpy()

        # Store each batch's probabilities with corresponding filenames
        # Add probabilities for each image
        batch_embeddings.extend(probabilities_numpy)
        batch_filenames.extend(filenames)  # Add corresponding filenames

    print("\nBatch processing complete.")  # Final completion message

    del probabilities, probabilities_numpy  # cleaning memory

    return np.array(batch_embeddings), np.array(batch_filenames)
