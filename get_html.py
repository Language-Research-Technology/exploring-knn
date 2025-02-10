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

# for flexible image loading


# This function is to allow you to refetch all the needed html rendering requirements for a given embedding
def get_html_rendering_details(embedding_name, image_zip, image_extensions,  number_of_neighbours=10, n_clusters=10):
    """The image_filenames passed in is to ensure that the pillow image, image filenames, and embeddings are all compatibile """

    # load the images for the zip file, check if the zip file matches the embedding file
    imgzip = zipfile.ZipFile(image_zip)
    # inflist = imgzip.infolist()
    Image.open(imgzip.open(imgzip.namelist()[3]))
    pillow_image_filenames = [name for name in imgzip.namelist(
    ) if name.endswith(image_extensions)]

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
def generate_html(nearest_neighbours_array, neighbours_distance_array, image_zip, image_extensions, diagram_data, cluster_labels, group_clusters=False, image_filepath="../images/", colour_map="tab20"):
    """Generates string html file to display the images and relevant information.
    group_cluster options:
    - `False`: Sort by rank normally
    - `True`: Order by Cluster label order
    - `asc`: Sort by ascending cluster size order
    - `desc`: Sort by descending cluster size order
    - `rank`: Sort by the highest ranked image in each cluster

    It is assumed that cluster_labels is a list of consecutive integers beginning with 0 (So they may be used as indices).
    Do not change encode_data from True, this was a legacy feature."""

    plotly_data = ""

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
        Image.open(imgzip.open(filename))) for filename in image_filenames]
    diagram_data = json.dumps(diagram_data_dict)
    del diagram_data_dict
    image_filepath = "data:image/jpeg;base64,"

    # add plotly
    response = requests.get("https://cdn.jsdelivr.net/npm/plotly.js-dist-min")
    plotly_data = response.text
    del response

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
    with open("html_file_template.txt", 'r') as file:
        return str(file.read()).format(
            number_of_clusters=number_of_clusters,  # Replace with actual value
            number_of_neighbours=number_of_neighbours,  # Replace with actual value
            table_rows=table_rows,
            data_json=diagram_data,
            cluster_text=cluster_text,
            image_path=image_filepath,
            plotly_data=plotly_data,
        )
