
import numpy as np
# Force directed diagram #
import networkx as nx
from scipy.stats import gaussian_kde
import pandas as pd
import json
from matplotlib import colormaps
from typing import Union


def get_position_df(nearest_neighbours_array: np.ndarray,
                    neighbours_distance_array: np.ndarray,
                    iterations: int = 100,
                    seed: int = 42,
                    k: float = None
                    ) -> pd.DataFrame:
    """
    Computes positions for each image based on a force-directed diagram simulation.

    This function calculates the 2D positions of images using a spring layout (force-directed graph). 
    The images are treated as nodes connected to their nearest neighbors, with forces applied 
    to either repel or attract them based on their nearest neighbor distances.

    Args:
        nearest_neighbours_array (np.ndarray): A 2D NumPy array of shape (n_samples, n_neighbours), 
            where the ith row contains the indices of the nearest neighbors for each sample.
        neighbours_distance_array (np.ndarray): A 2D NumPy array of shape (n_samples, n_neighbours), 
            where the ith row contains the distances to the nearest neighbors for each sample.
        iterations (int, optional): The maximum number of iterations for the spring simulation. Defaults to 100.
        seed (int, optional): The random seed for the spring simulation. Defaults to 42.
        k (float, optional): The optimal distance between nodes. If None, the distance is set to 1/sqrt(n)
            where n is the number of nodes. Defaults to None.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the computed 2D positions of the images with the following columns:
            - index: The index of the image corresponding to its filename.
            - x: The 'x' coordinate of the image in the force-directed diagram.
            - y: The 'y' coordinate of the image in the force-directed diagram.
    """
    # Create the graph
    G = nx.Graph()
    for i, neighbors in enumerate(nearest_neighbours_array):
        for neighbor in neighbors:
            if i != neighbor:  # Avoid self-loops
                # add a very small number to prevent divide by 0, is there a better way to handle this?
                weight = 1 / \
                    ((neighbours_distance_array[i][np.where(
                        nearest_neighbours_array[i] == neighbor)[0][0]])+1e-20)
                G.add_edge(i, neighbor, weight=weight)

    # Visualize the graph
    # key value pair of index and position
    # Use spring layout for positioning
    pos = nx.spring_layout(G, iterations=iterations,
                           seed=seed, k=k, weight='weight')
    # nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=100)
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))

    del G

    return pd.DataFrame(data={"index": list(pos.keys()), "x": np.array(list(pos.values()))[:, 0], "y": np.array(list(pos.values()))[:, 1]})

# position_df = get_position_df(nearest_neighbours_array, neighbours_distance_array)


def get_diagram_data(
        nearest_neighbours_array: np.ndarray,
        neighbours_distance_array: np.ndarray,
        cluster_labels: np.ndarray,
        image_filenames: np.ndarray[str],
        return_json: bool = True
) -> Union[str, dict]:
    """
    Computes and returns the data required to visualize the force-directed diagram with kernel density estimation (KDE).

    This function computes a force-directed layout of images based on their nearest neighbor distances, 
    applies kernel density estimation (KDE) on the 2D positions of the images, and generates color-coded cluster 
    labels for each image. It prepares the data for visualization, either as a dictionary or a JSON string.

    Args:
        nearest_neighbours_array (np.ndarray): A 2D NumPy array of shape (n_samples, n_neighbours), 
            where the ith row contains the indices of the nearest neighbors for each sample.
        neighbours_distance_array (np.ndarray): A 2D NumPy array of shape (n_samples, n_neighbours), 
            where the ith row contains the distances to the nearest neighbors for each sample.
        cluster_labels (np.ndarray): A 1D NumPy array containing the cluster labels for each image.
        image_filenames (np.ndarray): A 1D NumPy array containing the filenames of the images.
        return_json (bool, optional): If True, the function returns the diagram data as a JSON string. 
            If False, the function returns the data as a dictionary. Defaults to True.

    Returns:
        Union[dict, str]: A dictionary or JSON string containing the following keys:
            - grid: The grid values used for the KDE plot.
            - Z: The KDE evaluation over the grid.
            - x: The x-coordinates of the images in the diagram.
            - y: The y-coordinates of the images in the diagram.
            - index: The indices of the images.
            - rgba_colours: A list of RGBA color values for each cluster label.
            - image_filenames: A list of image filenames.
    """

    position_df = get_position_df(
        nearest_neighbours_array, neighbours_distance_array)

    # Compute the KDE using scipy
    kde = gaussian_kde([position_df['x'], position_df['y']])
    grid = np.linspace(min(position_df['x'].min(), position_df['y'].min()),
                       max(position_df['x'].max(), position_df['y'].max()),
                       100)  # done this way to make square plot

    X, Y = np.meshgrid(grid, grid)

    # Evaluate the KDE on the grid
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    colours = [colormaps.get_cmap("tab20")(
        value) for value in np.linspace(0, 1, len(set(cluster_labels)))]
    colours = np.hstack(
        (np.array(colours)[:, 0:3]*255, np.array(colours)[:, 3:4]))
    rgba_colours = [
        f"rgba({','.join(str(int(num)) for num in colours[label])})"
        for label in cluster_labels
    ]

    diagram_data = {"grid": grid.tolist(),
                    "Z": Z.tolist(),
                    "x": position_df['x'].tolist(),
                    "y": position_df['y'].tolist(),
                    "index": position_df['index'].tolist(),
                    "rgba_colours": rgba_colours,
                    "image_filenames": image_filenames.tolist()}

    if return_json:
        return json.dumps(diagram_data)

    return diagram_data


# ? This is code you can add into main.ipynb to see some extra visualisations,
# the reason these were removed from main.ipynb is as effectively the samne
# visualisations can be seen in the export html files. (this code is the prototypes)


# # Compute the KDE using scipy
# kde = gaussian_kde([position_df['x'], position_df['y']])
# grid = np.linspace(min(position_df['x'].min(), position_df['y'].min()),
#                    max(position_df['x'].max(), position_df['y'].max()),
#                    100) # done this way to make square plot
# # xgrid = np.linspace(position_df['x'].min(), position_df['x'].max(), 100)
# # ygrid = np.linspace(position_df['y'].min(), position_df['y'].max(), 100)
# X, Y = np.meshgrid(grid, grid)

# # # Evaluate the KDE on the grid
# Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

# colours = [colormaps.get_cmap("tab20")(value) for value in np.linspace(0,1,len(set(cluster_labels)))]
# colours = np.hstack((np.array(colours)[:,0:3]*255, np.array(colours)[:,3:4]))
# rgba_colours = [
#     f"rgba({','.join(str(int(num)) for num in colours[label])})"
#     for label in cluster_labels
# ]

# # data: grid, Z, x,y, index, rgba_clours,

# # fig = go.Figure()
# fig = go.FigureWidget()


# fig.add_trace(go.Contour(
#     z=Z, x=grid, y=grid, colorscale='Inferno',
#     contours=dict(
#         showlabels=True, labelfont=dict(size=12), coloring='heatmap'
#     ),
#     colorbar=dict(title='Density')
# ))

# fig.add_trace(go.Scattergl(x=position_df['x'],
#                          y=position_df['y'],
#                         #  hoverinfo='text',
#                          text=position_df['index'].tolist(),
#                          mode='markers',
#                          marker=dict(
#         symbol='circle',
#         opacity=0.5,
#         color=rgba_colours,
#         size=4,
#         line=dict(width=1),
#     )))


# fig.update_layout(
#     xaxis=dict( ticks='', showgrid=False, zeroline=False, nticks=20 ),
#     yaxis=dict( ticks='', showgrid=False, zeroline=False, nticks=20 ),
#     title="Spring Visualisation for Image embeddings based off Nearest Neighbour Distance",
#     autosize=False,
#     height=800,
#     width=800,
#     hovermode='closest'
# )
# # fig.write_image("output/springplot_clip.png")

# def resize_image(image, set_width):
#     # Calculate the new height keeping the aspect ratio
#     width, height = image.size
#     new_height = int((set_width / width) * height)

#     # Resize image
#     resized_image = image.resize((set_width, new_height))
#     return resized_image

# # Function to paste all images into one new image
# def paste_images_to_new_image(filenames, set_width, images_per_row=5):
#     # Load images and resize them
#     images = [resize_image(Image.open(os.path.join(IMAGE_FOLDER, filename)), set_width) for filename in filenames]

#     total_width = set_width * images_per_row

#     # Calculate total height: total height of all images divided by images per row
#     total_height = 0
#     row_height = 0
#     for i, image in enumerate(images):
#         row_height = max(row_height, image.size[1])  # Max height in the current row

#         # When reaching the end of a row (after images_per_row images), move to the next row
#         if (i + 1) % images_per_row == 0 or (i + 1) == len(images):
#             total_height += row_height
#             row_height = 0  # Reset row height for the next row

#     new_image = Image.new("RGB", (total_width, total_height), (255, 255, 255))

#     # Paste each image onto the new image
#     x_offset = 0
#     y_offset = 0
#     row_height = 0  # Track height of the current row

#     for i, image in enumerate(images):
#         new_image.paste(image, (x_offset, y_offset))  # Paste image at the offset

#         # Update the offset for the next image
#         x_offset += image.size[0]

#         # Track the row height (the tallest image in the row)
#         row_height = max(row_height, image.size[1])

#         # When the row is full (i.e., we’ve placed `images_per_row` images), move to the next row
#         if (i + 1) % images_per_row == 0 or (i + 1) == len(images):
#             # Update y_offset to the next row
#             y_offset += row_height
#             x_offset = 0  # Reset x_offset for the next row
#             row_height = 0  # Reset row height for the next row

#     return new_image


# def get_index_given_coord(x, y):
#     # assumes no two points are at the same coordinate (which shouldn't be possible due to the nature of spring simulation)
#     return position_df[(position_df['x']==x) & (position_df['y']==y)]['index'].iloc[0]

# def click_handler(trace, points, selector):
#     print("clicked something")

# def selection_handler(trace, points, selector):
#     print("Selected", end='\r')
#     xs = points._xs
#     ys = points._ys
#     indices = [get_index_given_coord(*coord) for coord in zip(xs,ys)]
#     filenames = [image_filenames[index] for index in indices]
#     print(f"Selected {len(filenames)} images.")
#     # To keep the grid roughly square:
#     images_per_row = int(np.ceil(np.sqrt(len(filenames))))
#     grid_image = paste_images_to_new_image(filenames, 150, images_per_row=images_per_row)
#     grid_image.show()

# sca = fig.data[1]
# sca.on_click(click_handler)
# sca.on_selection(selection_handler)

# for the html
# diagram_data = {"grid":grid.tolist(),
#         "Z":Z.tolist(),
#         "x":position_df['x'].tolist(),
#         "y":position_df['y'].tolist(),
#         "index":position_df['index'].tolist(),
#         "rgba_colours":rgba_colours,
#         "image_filenames": image_filenames.tolist()}

# diagram_data = json.dumps(diagram_data)

# # fig.show()
# # fig.write_html("./output/spring_visualization_with_selection.html", include_plotlyjs="cdn", full_html=True)

# fig

#! Diagram for each cluster


# subplot_rows = 5
# subplot_cols = 4

# # Compute the KDE using scipy
# kde = gaussian_kde([position_df['x'], position_df['y']])
# grid = np.linspace(min(position_df['x'].min(), position_df['y'].min()),
#                    max(position_df['x'].max(), position_df['y'].max()),
#                    100) # done this way to make square plot
# # xgrid = np.linspace(position_df['x'].min(), position_df['x'].max(), 100)
# # ygrid = np.linspace(position_df['y'].min(), position_df['y'].max(), 100)
# X, Y = np.meshgrid(grid, grid)

# # Evaluate the KDE on the grid
# Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)


# colours = [colormaps.get_cmap("tab20")(value) for value in np.linspace(0,1,len(set(cluster_labels)))]
# colours = np.hstack((np.array(colours)[:,0:3]*255, np.array(colours)[:,3:4]))

# fig = make_subplots(rows=subplot_rows, cols=subplot_cols, subplot_titles=[f"Cluster {num}" for num in set(cluster_labels)])

# location = (1,1)


# for target_cluster in set(cluster_labels):


#     rgba_colours = [
#         f"rgba({','.join(str(num) for num in [255,255,255,1])})"
#         if label == target_cluster else f"rgba({','.join(str(num) for num in [0,0,0,0])})"
#         for label in cluster_labels
#     ]


#     fig.add_trace(go.Contour(
#         z=Z, x=grid, y=grid, colorscale='Inferno',
#         contours=dict(
#             showlabels=True, labelfont=dict(size=12), coloring='heatmap'
#         ),
#         colorbar=dict(title='Density')
#     ), row = location[0], col=location[1])

#     fig.add_trace(go.Scatter(x=position_df['x'],
#                             y=position_df['y'],
#                             hovertext=position_df['index'],
#                             mode='markers',
#                             marker=dict(
#             symbol='circle',
#             opacity=0.5,
#             color=rgba_colours,
#             size=1,
#             line=dict(width=0),
#         )), row = location[0], col=location[1])


#     if location[1] < subplot_cols:
#         location = (location[0], location[1]+1)
#     else: # go to new col
#         location = (location[0]+1, 1)

# fig.update_layout(
#     # xaxis=dict( ticks='', showgrid=False, zeroline=False, nticks=20 ),
#     # yaxis=dict( ticks='', showgrid=False, zeroline=False, nticks=20 ),
#     title="Force-Directed Graphs with Kernel Density Estimate (KDE) Plots for VGG19 Image Embeddings: Visualizing Nearest-Neighbor Relationships and K-Means Clusters",
#     autosize=False,
#     height=1000,
#     width=700,
#     hovermode='closest',
#     showlegend=False,
#     margin=dict(l=30, r=30, t=80, b=40)

# )

# fig.show()
