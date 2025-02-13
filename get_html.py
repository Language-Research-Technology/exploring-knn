import zipfile
from PIL import Image  # Pillow
import numpy as np

from matplotlib import colormaps


import json

# adding images to html
import base64
from io import BytesIO
# for incorporating plotly
import requests

from embeddings_analysis import get_knn, get_cluster_labels, get_count_neighbour_occurances, rank_neighbour_occurances, get_mean_distance
from force_directed_diagram import get_diagram_data

from typing import Optional, Tuple, Union

# for flexible image loading


# This function is to allow you to refetch all the needed html rendering requirements for a given embedding
def get_html_rendering_details(embedding_name: str,
                               image_zip: str,
                               image_extensions: Tuple[str],
                               image_ignore: Tuple[str],
                               number_of_neighbours: int = 10,
                               n_clusters: int = 10
                               ) -> Optional[Tuple[np.ndarray, np.ndarray, zipfile.ZipFile, str, str, np.ndarray]]:
    """
    Retrieves the necessary details for rendering the HTML visualization based on an embedding.

    This function loads image filenames from a zip file, matches them with corresponding embedding files, 
    calculates the nearest neighbors and cluster labels, and generates the diagram data for the 
    force-directed graph.

    Args:
        embedding_name (str): The name of the embedding (matching the files of the format 
            'image_embedding_{embedding_name}.npy' and 'image_filenames_{embedding_name}.npy').
        image_zip (str): The path to the zip file containing images.
        image_extensions (Tuple[str]): The extension(s) of the image files to load from the zip.
        image_ignore (Tuple[str]): A tuple of filename prefixes to exclude from processing.
        number_of_neighbours (int, optional): The number of nearest neighbors to compute. Defaults to 10.
        n_clusters (int, optional): The number of clusters for the clustering algorithm. Defaults to 10.

    Returns:
        Union[Tuple[np.ndarray, np.ndarray, zipfile.ZipFile, Tuple[str], str, np.ndarray], None]: 
        A tuple containing:
            - nearest_neighbours_array: A 2D array of shape (n_samples, n_neighbours).
            - neighbours_distance_array: A 2D array of shape (n_samples, n_neighbours).
            - image_zip: The zip file object.
            - image_extensions: The image extensions to match.
            - diagram_data: JSON-encoded data for rendering the force-directed diagram.
            - cluster_labels: Array of cluster labels.
        If the embedding file cannot be found, None will be returned.
    """

    # load the images for the zip file, check if the zip file matches the embedding file
    imgzip = zipfile.ZipFile(image_zip)
    # inflist = imgzip.infolist()
    # Image.open(imgzip.open(imgzip.namelist()[3]))
    pillow_image_filenames = [name for name in imgzip.namelist(
    ) if name.endswith(image_extensions) and not name.split("/")[-1].startswith(image_ignore)]

    # pillow_images =
    try:
        embedding = np.load(f"./output/image_embedding_{embedding_name}.npy")
        embedding_image_filenames = np.load(
            f"./output/image_filenames_{embedding_name}.npy")
    except:
        print('Embeddings not saved. ')  # ! todo make it render?
        return

    if any(embedding_image_filenames != pillow_image_filenames):
        raise "Saved filenames and current filenames don't match"

    nearest_neighbours_array, neighbours_distance_array = get_knn(
        embedding, number_of_neighbours)
    cluster_labels = get_cluster_labels(embedding, n_clusters=n_clusters)

    diagram_data = get_diagram_data(nearest_neighbours_array, neighbours_distance_array,
                                    cluster_labels, embedding_image_filenames, return_json=True)

    return nearest_neighbours_array, neighbours_distance_array, image_zip, image_extensions, diagram_data, cluster_labels


# regarding colour_map: If you are using above 20 clusters, try "viridis" or a similar continuous map
# This lists the avaliable colormaps for the cluster colouring, recommended tab10 or tab20 for good colour differentiation
# list(colormaps)
def generate_html(
    nearest_neighbours_array: np.ndarray,
    neighbours_distance_array: np.ndarray,
    image_zip: str,
    image_extensions: str,
    diagram_data: str,
    cluster_labels: np.ndarray,
    group_clusters: Union[bool, str] = False,
    colour_map: str = "tab20"
) -> str:
    """
    Generates an HTML string that visualizes the images and relevant information in a table format.

    This function renders a table of images, their nearest neighbors, and related statistics in an HTML format.
    The images are displayed as resized base64 encoded strings, and the table is sorted based on the group_clusters option.

    Args:
        nearest_neighbours_array (np.ndarray): A 2D NumPy array containing the indices of the nearest neighbors for each image.
        neighbours_distance_array (np.ndarray): A 2D NumPy array containing the distances to the nearest neighbors.
        image_zip (str): The path to the zip file containing images.
        image_extensions (Tuple[str]): The extension(s) of the image files to load from the zip.
        diagram_data (str): JSON-encoded diagram data, typically from get_diagram_data.
        cluster_labels (np.ndarray): A 1D NumPy array of cluster labels corresponding to each image.
        group_clusters (Union[bool, str], optional): A flag or string indicating how to group images (default is False). 
            Options are:
            - False: Sort by rank normally.
            - True: Order by cluster label order.
            - "asc": Sort by ascending cluster size.
            - "desc": Sort by descending cluster size.
            - "rank": Sort by the highest ranked image in each cluster.
        colour_map (str, optional): The name of the colormap to use for color differentiation between clusters (default is "tab20").

    Returns:
        str: A string containing the HTML code for rendering the image visualization.
    """

    imgzip = zipfile.ZipFile(image_zip)
    image_filenames = [name for name in imgzip.namelist(
    ) if name.endswith(image_extensions)]

    # make it so that instead of image src pointing to an image, it contains the image

    def get_resized_b64(image, set_height=150):
        def resize_image(image, set_height):
            # Calculate the new height keeping the aspect ratio
            width, height = image.size
            new_width = int((set_height / height) * width)

            # Resize image
            resized_image = image.resize((new_width, set_height))
            return resized_image

        im_file = BytesIO()
        resized_img = resize_image(image, 150)
        resized_img.save(im_file, format="JPEG")
        im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
        im_b64 = base64.b64encode(im_bytes)
        return im_b64.decode("utf-8")

    diagram_data_dict = json.loads(diagram_data)
    diagram_data_dict['image_filenames'] = [get_resized_b64(
        Image.open(imgzip.open(filename)).convert("RGB")) for filename in image_filenames]
    diagram_data = json.dumps(diagram_data_dict)
    del diagram_data_dict
    image_filepath = "data:image/jpeg;base64,"

    # add plotly
    # response = requests.get("https://cdn.jsdelivr.net/npm/plotly.js-dist-min")
    # plotly_data = response.text
    # del response

    count_neighbour_occurances_array = get_count_neighbour_occurances(
        nearest_neighbours_array)
    ranked_indices = rank_neighbour_occurances(
        count_neighbour_occurances_array, ascending=False)
    number_of_neighbours = len(nearest_neighbours_array[0])
    number_of_clusters = len(set(cluster_labels))

    colours = [colormaps.get_cmap(colour_map)(value)
               for value in np.linspace(0, 1, number_of_clusters)]

    neighbours_mean_distance_array = get_mean_distance(
        neighbours_distance_array)

    max_neighbour_occurances = max(
        count_neighbour_occurances_array) - 1  # -1 to exclude self
    max_mean_distance = max(neighbours_mean_distance_array)

    row_template = """ 
            <tr id="{}">
                <td><div style="width: 100%; background-color: rgba({},{},{},{}); height: 150px;">{}. ({}) [{}]</div></td>
                <td><div style="width: {}%; background-color: DarkSeaGreen; height: 150px;text-align:right;white-space: nowrap;">{} ({}%)</div></td>
                <td><div style="width: {}%; background-color: CornflowerBlue; height: 150px;text-align:right;white-space: nowrap;">{} ({}%)</div></td>
                <td>{}</td>
                <td>{}</td>
                <td><img id="image_{}" alt="Image" style="width:auto; height:150px;"></td>
            </tr>"""

    table_rows = """"""
    # iterate through the ranks
    # rank = 1 means this image has the most neighbours

    def generate_nearest_neighbours_html(nearest_neighbours):
        link_template = '<a href="#{}">{}</a>'
        formatted_nearest_neighbours = [link_template.format(
            idx, idx) for idx in nearest_neighbours]
        nearest_neighbours_html = ", ".join(formatted_nearest_neighbours)
        return nearest_neighbours_html

    def generate_html_image_module(index, rank):
        """This will generate the html for a given image"""
        # can use index to get filename, nearest_neighbours, and count of nearest neighbours

        filename = image_filenames[index]
        nearest_neighbours = nearest_neighbours_array[index]
        # -1 to exclude self
        count_neighbour_occurances = count_neighbour_occurances_array[index] - 1
        mean_distance = neighbours_mean_distance_array[index]
        mean_distance_percentage = round(
            mean_distance*100/max_mean_distance, 2)
        nearest_neighbours_html = generate_nearest_neighbours_html(
            nearest_neighbours)
        neighbour_occurance_percentage = round(
            count_neighbour_occurances*100/max_neighbour_occurances, 2)
        # as cluster_label includes 0 can use as an index
        cluster_label = cluster_labels[index]
        # need to convert to rgba format (* by 255)
        colour = colours[cluster_label]

        return row_template.format(index,
                                   # assign rgba colours
                                   colour[0]*255, colour[1]*255, colour[2] *
                                   255, colour[3], rank, index, cluster_label,
                                   neighbour_occurance_percentage, count_neighbour_occurances, neighbour_occurance_percentage,
                                   # scientific notation as some embedding types have very small distances
                                   mean_distance_percentage, "{:.2e}".format(
                                       mean_distance), mean_distance_percentage,
                                   filename,
                                   nearest_neighbours_html,
                                   index)  # image_filepath+filename)

    # idx in this case is the item in the list, not the index of the item in the list
    # this is an index referencing the other lists

    if group_clusters:  # group_clusters can be "asc" "desc" "rank", any other truthy value will sort by cluster name

        # Sort by either Asc or Desc based off the highest ranking RKNN image in the cluster
        if group_clusters == "asc" or group_clusters == "desc":
            _, counts = np.unique(cluster_labels, return_counts=True)
            # argsort
            ordered_cluster_labels = np.argsort(counts)
            if group_clusters == "desc":
                ordered_cluster_labels = ordered_cluster_labels[::-1]

        elif group_clusters == "rank":
            ordered_cluster_labels = []
            for index in ranked_indices:
                if cluster_labels[index] not in ordered_cluster_labels:
                    ordered_cluster_labels.append(cluster_labels[index])
                    if len(ordered_cluster_labels) == len(set(cluster_labels)):
                        break  # if we have all cluster labels break

        else:  # Any other truthy value
            ordered_cluster_labels = set(cluster_labels)

        # this gets all of cluster 0 in rank order then all of cluster 1 in rank order etc.
        for idx, rank in [(index, rank+1) for target_cluster in ordered_cluster_labels
                          for rank, index in enumerate(ranked_indices)
                          if cluster_labels[index] == target_cluster]:
            table_rows += generate_html_image_module(idx, rank)
        cluster_text = " (and grouped by cluster)"
    else:
        for rank, idx in enumerate(ranked_indices):
            # rank starts at 0
            table_rows += generate_html_image_module(idx, rank+1)

        cluster_text = ""
    with open("./html_text/html_file_template.txt", 'r') as file:
        with open("./html_text/plotly_js.txt", 'r') as plotly_data:
            return str(file.read()).format(
                number_of_clusters=number_of_clusters,  # Replace with actual value
                number_of_neighbours=number_of_neighbours,  # Replace with actual value
                table_rows=table_rows,
                data_json=diagram_data,
                cluster_text=cluster_text,
                image_path=image_filepath,
                plotly_data=plotly_data.read(),
            )
