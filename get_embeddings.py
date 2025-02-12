import zipfile
from PIL import Image  # Pillow

from concurrent.futures import ThreadPoolExecutor  # for parallel processing
from tqdm import tqdm  # for progress bars

import numpy as np

import torch
# Clip ## # Meta clip
# ensure using huggingface-hub v0.25.0 or earlier to prevent import issue
from imgbeddings import imgbeddings


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

def get_clip_embedding(image_zip, image_extensions, batch_size=32):
    """ 
    A function to get clip embeddings.

    Params:
        - pillow_images: list of pillow Image objects.
        - embedding_function: a function which takes in <class 'PIL.JpegImagePlugin.JpegImageFile'>
        and returns a 1D array of floats (by defauly creates colour histogram).

    Returns:
        - embedding: list of embedding where the ith row in embedding.
        corresponds to the ith image_filename in image_filenames.

    """

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
def get_normalised_histogram(img):
    """Returns a pixel standardised pillow colour histogram array."""
    w, h = img.size
    pixels = w*h
    return np.array(img.histogram())/pixels

# embedding_function takes a batch of images, get_embedding takes batch_size=


def get_embedding(image_zip, image_extensions, embedding_function=get_normalised_histogram) -> "[embedding_function(img_0),...]":
    """ 
    A function to get embeddings, this does not work well for intensive embeddings which benefit from batching.

    Params:
        - pillow_images: list of pillow Image objects.
        - embedding_function: a function which takes in <class 'PIL.JpegImagePlugin.JpegImageFile'>
        and returns a 1D array of floats (by defauly creates colour histogram).

    Returns:
        - embedding: list of embedding where the ith row in embedding.
        corresponds to the ith image_filename in image_filenames.

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
def get_embeddings_pytorch(dataloader, pytorch_model):
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
