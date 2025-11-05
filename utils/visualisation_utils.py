import asyncio
import json
import math
import os
import time
from copy import deepcopy

import numpy as np
from bokeh.embed import components

from sompy.aux_fun import aux_fun
from sompy.visualization.viz_functions import Visualization_func
from utils.utils import training_data, get_user_visualisations_directory, remove_script_tags, get_user_directory, \
    SAVE_DIRECTORY


def train_async_in_background(app, current_user, data, request_id, train_length, extend_training):
    asyncio.run(train_async(app, current_user, data, request_id, train_length, extend_training))


async def train_async(app, current_user, df, request_id, train_length, extend_training):
    with app.app_context():
        message = "Processing visualisation."
        training_data[request_id]['message'] = message
        try:
            inputs = training_data[request_id]['variables']
            outputs = (df.shape[1] - inputs)
            training_data[request_id]['total_steps'] = training_data[request_id]['total_steps'] + (outputs * train_length) + 2

            n_array = np.array(df)
            comp_names = list(df.columns)
            df_outputs = df.iloc[:, inputs:]

            column_classifiers = []

            for idx in range(df_outputs.shape[1]):
                column_name = df_outputs.columns[idx]
                unique_values = np.sort(df_outputs[column_name].dropna().unique())
                column_classifiers.append((idx, column_name, unique_values))

            n_array_data = n_array[:, :n_array.shape[1] - outputs]
            cnames_data = comp_names[0:len(comp_names) - outputs]

            # Calculation of weights
            ranges = np.max(n_array[:, -outputs:], axis=0) - np.min(n_array[:, -outputs:], axis=0)
            sum_ranges = sum(ranges)
            weights = ranges / sum_ranges

            n_array0 = np.hstack(
                (n_array_data, np.dot(weights, n_array[:, -outputs:].T).reshape(n_array.shape[0], 1)))  # Weighted average
            cnames0 = [*cnames_data, 'Favg']

            for i in range(1, outputs + 1):
                # Dynamically assign variables using globals()
                globals()[f"n_array{i}"] = np.hstack((n_array_data, n_array[:, inputs - 1 + i].reshape(n_array.shape[0], 1)))
                globals()[f"cnames{i}"] = [*cnames_data, comp_names[inputs - 1 + i]]

            map_size_n = max(20, int(math.sqrt(math.sqrt(training_data[request_id]['instances']) * 5)))

            from sompy.sompy_isom import SOMFactory as smf

            sm0 = smf.build(n_array0, normalization='range', initialization='pca', mapshape="planar", lattice="hexa",
                            neighborhood='gaussian',
                            training='batch', mapsize=[map_size_n, map_size_n], component_names=cnames0)
            sm0.som_lininit()

            if extend_training == True and 'som_objects' in training_data[request_id]:
                som_objects = training_data[request_id]['som_objects']

                for (index, column_name, unique_values), sm in zip(column_classifiers, som_objects):
                    sm.set_initialized_matrix(sm.codebook.matrix)
                    sm.train(n_job=1, shared_memory=False, verbose='info', request_id=request_id,
                             train_rough_len=0, train_finetune_len=train_length, train_len_factor=1)
            else:
                if extend_training:
                    sm0.set_initialized_matrix(sm0.codebook.matrix)
                    train_rough_length = 0
                    train_finetune_len = train_length
                else:
                    train_rough_length = 100
                    train_finetune_len = train_length - 100

                som_models = []  # List to store all trained SOM models

                # Train SOM models for each output
                for i in range(1, outputs + 1):
                    # Get n_array and cnames dynamically
                    n_array = globals()[f"n_array{i}"]
                    cnames = globals()[f"cnames{i}"]

                    # Build the SOM model
                    som = smf.build(
                        n_array,
                        mapsize=sm0.codebook.mapsize,
                        component_names=cnames
                    )

                    # Set initialized matrix
                    som.set_initialized_matrix(sm0.initialized_matrix)

                    # Train the SOM model
                    som.train(
                        request_id=request_id,
                        n_job=1,
                        shared_memory=False,
                        verbose='info',
                        train_rough_len=train_rough_length,
                        train_finetune_len=train_finetune_len,
                        train_len_factor=1
                    )

                    # Append the trained SOM model to the list
                    som_models.append(som)

                for j in range(1, outputs + 1):
                    globals()[f"sm{j}"] = som_models[j - 1]

                som_objects = [globals()[f'sm{i}'] for i in range(1, outputs + 1)]

            for (index, column_name, unique_values), sm in zip(column_classifiers, som_objects):
                matrix = sm.codebook.matrix
                last_column = matrix[:, -1]

                num_unique = len(unique_values)

                # Binary classifier
                if num_unique == 2:
                    rounded_column = np.where(last_column < 0.5, unique_values[0], unique_values[1])

                # Ternary classifier
                elif num_unique == 3:
                    sorted_values = np.sort(unique_values)
                    thresholds = [0.33, 0.67]
                    rounded_column = np.digitize(last_column, bins=thresholds, right=True)
                    rounded_column = sorted_values[rounded_column]

                    # Quaternary classifier
                elif num_unique == 4:
                    sorted_values = np.sort(unique_values)
                    thresholds = [0.25, 0.5, 0.75]
                    rounded_column = np.digitize(last_column, bins=thresholds, right=True)
                    rounded_column = sorted_values[rounded_column]

                    # Continuous or larger classifiers
                else:
                    rounded_column = last_column  # No rounding

                modified_matrix = matrix.copy()
                modified_matrix[:, -1] = rounded_column

                sm.set_initialized_matrix(modified_matrix)

            if training_data[request_id]['status'] != 'Cancelled':
                training_data[request_id]['som_objects'] = som_objects
                save_visualisation_metrics(current_user, request_id, column_classifiers)
                save_visualisation_plots(current_user, request_id, sm0)
                training_data[request_id]['status'] = 'Completed'
                message = "Training completed."
        except Exception as e:
            message = "Visualisation training failed. Please check your input and try again."
            training_data[request_id]['status'] = 'Failed'
            print(str(e))

        if training_data[request_id]['status'] != 'Cancelled':
            training_data[request_id]['message'] = message
            training_data[request_id]['completed_steps'] = training_data[request_id]['total_steps']

        training_data[request_id]['updated_at'] = int(time.time())
        save_training_data(current_user, request_id)


def calculate_metrics(request_id, column_classifiers):
    metrics = []
    axf = aux_fun()
    for (index, column_name, unique_values), smi in zip(column_classifiers, training_data[request_id]['som_objects']):
        metric = {
            "quantization_error": np.mean(smi._bmu[1]),
            "topographic_error": axf.topographic_error(smi),
            "column_name": column_name
        }
        metrics.append(metric)

    return metrics


def get_visualisations(request_id, sm0):
    fig_objs = []
    hit_objs = []
    axf = aux_fun()
    som_objects = training_data[request_id]['som_objects']
    for sm in som_objects:
        vis = Visualization_func(sm)
        fig_obj = vis.get_som_cplanes()
        fig_objs.append(fig_obj)

        hits = axf.som_bmus2(sm, sm._data)
        hit_obj = vis.get_som_hitmap(hits, comp='all', clr='#000000')
        hit_objs.append(hit_obj)

    inputs = training_data[request_id]['variables']
    outputs = len(som_objects)
    final_fig_objs = []
    final_hit_objs = []
    for i in range(0, outputs):
        final_fig_objs.append(fig_objs[i][0])
        final_hit_objs.append(hit_objs[i][0])

    for i in range(0, inputs):
        final_fig_objs.append(fig_objs[0][i+1])
        final_hit_objs.append(hit_objs[0][i+1])


    # grid plots
    gridplots = vis.plot_cplanes(final_fig_objs)

    hitmap = vis.plot_hitmap(final_hit_objs, comp='all')

    som_umatrix_data = som_objects[0].codebook.matrix
    for i in range(1, outputs):
        last_column = som_objects[i].codebook.matrix[:, -1]
        som_umatrix_data = np.column_stack((som_umatrix_data, last_column))

    som_4umatrix = deepcopy(sm0)
    som_4umatrix.set_codebook_matrix(som_umatrix_data)

    axf = aux_fun()
    U = axf.som_umat(som_4umatrix, comp='all')
    umatrix_plots = vis.plot_umat(U)  # This will return U-matrix plot

    return gridplots, umatrix_plots, hitmap


def save_training_data(current_user, request_id):
    title = training_data[request_id]['title']
    vis_dir = get_user_visualisations_directory(current_user, request_id)
    file_name = title + ".json"
    file_path = os.path.join(vis_dir, file_name)
    data = {key: value for key, value in training_data[request_id].items() if key != 'som_objects'}
    data['request_id'] = request_id
    # Save the dictionary to the JSON file
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def save_visualisation_plots(current_user, request_id, sm0):
    grid_plots, umatrix_plots, hitmap = get_visualisations(request_id, sm0)

    comp_planes_script, comp_planes_div = components(grid_plots)
    comp_planes_script_modified = remove_script_tags(comp_planes_script)

    hitmap_script, hitmap_div = components(hitmap)
    hitmap_script_modified = remove_script_tags(hitmap_script)

    umatrix_script, umatrix_div = components(umatrix_plots)
    umatrix_script_modified = remove_script_tags(umatrix_script)

    plots = {
        "comp_planes_script": comp_planes_script_modified,
        "comp_planes_div": comp_planes_div,
        "umatrix_script": umatrix_script_modified,
        "umatrix_div": umatrix_div,
        "hitmap_script": hitmap_script_modified,
        "hitmap_div": hitmap_div
    }

    vis_dir = get_user_visualisations_directory(current_user, request_id)
    vis_plots_file_path = os.path.join(vis_dir, 'plots.json')
    with open(vis_plots_file_path, "w") as plots_file:
        json.dump(plots, plots_file, indent=4)


def save_visualisation_metrics(current_user, request_id, column_classifiers):
    metrics = calculate_metrics(request_id, column_classifiers)
    vis_dir = get_user_visualisations_directory(current_user, request_id)
    vis_metrics_file_path = os.path.join(vis_dir, 'metrics.json')
    with open(vis_metrics_file_path, "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=4)


def load_all_visualisations(directory):
    vis_data = []

    # Loop through all visualisation subdirectories in the user directory
    for root, dirs, files in os.walk(directory):
        for subdir in dirs:
            vis_file = os.path.join(root, subdir, subdir + '.json')
            if os.path.exists(vis_file):
                request_id = None
                with open(vis_file, 'r') as f:
                    data_dict = json.load(f)
                    vis_data.append(data_dict)

                    request_id_key = 'request_id'
                    request_id = data_dict.get(request_id_key)
                    if request_id and request_id not in training_data:
                        training_data[request_id] = {}
                        for item in data_dict.keys():
                            if item != request_id_key:
                                training_data[request_id][item] = data_dict[item]

    vis_data = sorted(vis_data, key=lambda vis: vis["updated_at"], reverse=True)
    return vis_data


def fix_inprogress_visualisations_on_server_start():
    save_dir = os.getenv("SAVE_DIRECTORY")
    for dirpath, dirnames, filenames in os.walk(save_dir):
        for filename in filenames:
            # Process only JSON files with a name that matches their parent folder's name
            if filename.endswith('.json'):
                parent_folder_name = os.path.basename(dirpath)
                json_file_name = os.path.splitext(filename)[0]  # Get the filename without extension

                # Check if JSON file name matches parent folder name
                if json_file_name == parent_folder_name:
                    file_path = os.path.join(dirpath, filename)
                    try:
                        # Open the JSON file
                        with open(file_path, 'r') as json_file:
                            data = json.load(json_file)

                        # Fix visualisations which are stuck in InProgress state
                        if data['status'] == 'InProgress':
                            data['status'] = 'Failed'
                            data['completed_steps'] = data['total_steps']
                            data['message'] = 'Unexpected server error.'

                            # Write the updated data back to the file
                            with open(file_path, 'w') as json_file:
                                json.dump(data, json_file, indent=4)
                    except Exception as e:
                        print(f"Error updating {file_path}: {e}")


def load_full_visualisation_detail(directory, title):
    vis_file = os.path.join(directory, title, f'{title}.json')
    if os.path.exists(vis_file):
        with open(vis_file, 'r') as f:
            visualisation = json.load(f)
            request_id_key = 'request_id'
            request_id = visualisation.get(request_id_key)
            if request_id and request_id not in training_data:
                training_data[request_id] = {}
                for item in visualisation.keys():
                    if item != visualisation:
                        training_data[request_id][item] = visualisation[item]

        if visualisation is not None:
            plots_file = os.path.join(directory, title, 'plots.json')
            if os.path.exists(plots_file):
                with open(plots_file, 'r') as f:
                    visualisation['plots'] = json.load(f)

            metrics_file = os.path.join(directory, title, 'metrics.json')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    visualisation['metrics'] = json.load(f)
    return visualisation


def get_input_file_data(vis_dir, title):
    vis_file = os.path.join(vis_dir, title, f'{title}.json')
    if os.path.exists(vis_file):
        with open(vis_file, 'r') as f:
            visualisation = json.load(f)
            if visualisation and 'file' in visualisation.keys():
                csv_file_path = os.path.join(vis_dir, title, visualisation['file'])
                with open(csv_file_path, mode='r') as file:
                    return file.read()  # Read the raw text from the file
    return None

def load_visualisation_detail(current_user, title):
    user_dir = get_user_directory(current_user)
    vis_file = os.path.join(user_dir, title, f'{title}.json')
    if os.path.exists(vis_file):
        with open(vis_file, 'r') as f:
            return json.load(f)

