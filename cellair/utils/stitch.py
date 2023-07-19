import cv2
import os
import re

import pandas as pd

from openpyxl.utils.cell import coordinate_from_string, column_index_from_string, get_column_letter
# xy = coordinate_from_string('A4') # returns ('A',4)
# col = column_index_from_string(xy[0]) # returns 1
# row = xy[1]

import numpy as np

from hehuprofiler.utils.constants import dataroot

def rescale_image(image):
    image_min = image.min()
    image = image - image_min
    image_max = image.max()
    image /= image_max
    
    return (image * 255).astype(np.uint8)


def field_to_row_column(field, pattern_table):
    
    # field_row = field // 3
    
    # row_order = [1, -1][int(field_row%2)]
    
    # field_column = np.arange(3, dtype=int)[::row_order][field%3]
    
    status = False
    field_row = None
    field_col = None
    for i, row in enumerate(pattern_table):
        for j, cell in enumerate(row):
            if cell == field:
                status = True
                field_row = i
                field_col = j
                break
        
    
    # print("field_row: ", field_row)
    # print("field_col: ", field_col)
    if status:
       msg = "Success"
    else:
       msg = "Pattern Table Is Incorrect"
    
    return status, field_row, field_col, msg


def paths2paths_df(paths, pattern_table):
    path_info = []
    for p in paths:
        # print(re.findall(r"_(z\d+)_\d+_([\w]+\d+)(f\d+)(d\d+)", p))
        if len(re.findall(r"_z(\d+)_\d+_([\w]+\d+)(f\d+)(d\d+)", p)) <= 0:
            continue
        # print(re.findall(r"_(z\d+)_\d+_([\w]+\d+)(f\d+)(d\d+)", p))
        depth, well, field, channel = re.findall(r"_z(\d+)_\d+_([\w]+\d+)f(\d+)(d\d+)", p)[0]
        xy = coordinate_from_string(well)
        well_row = column_index_from_string(xy[0])
        well_column = xy[1]
        
        status, field_row, field_column, msg = field_to_row_column(int(field), pattern_table)
        
        if not status:
            return False, [], msg
        
        path_info.append((p, int(depth), well_row, well_column, field_row, field_column, channel))
        

    paths_df = pd.DataFrame(path_info, columns=["path", "depth", "well_row", "well_column", "field_row", "field_column", "channel"])

    cat = pd.Categorical(paths_df["depth"], ordered=True)
    codes, uniques = pd.factorize(cat, sort=True)
    paths_df["depth"] = codes

    cat = pd.Categorical(paths_df["well_row"], ordered=True)
    codes, uniques = pd.factorize(cat, sort=True)
    paths_df["well_row"] = codes

    cat = pd.Categorical(paths_df["well_column"], ordered=True)
    codes, uniques = pd.factorize(cat, sort=True)
    paths_df["well_column"] = codes

    return True, paths_df, "Success"

def stitch_evos_images(raw_dir, stitched_dir, pattern_table):
    raw_root = os.path.join(dataroot, "0_raw")
    raw_path = os.path.join(raw_root, raw_dir)

    paths = os.listdir(raw_path)

    # print("paths: ", paths)

    paths = [os.path.join(raw_path, p) for p in paths]

    status, paths_df, msg = paths2paths_df(paths, pattern_table)
    if not status:
        return False, [], msg

    # print(paths_df)
    
    n_depth = pd.unique(paths_df["depth"]).shape[-1]
    n_row = pd.unique(paths_df["well_row"]).shape[-1]
    n_column = pd.unique(paths_df["well_column"]).shape[-1]
    n_field_row = pd.unique(paths_df["field_row"]).shape[-1]
    n_field_column = pd.unique(paths_df["field_column"]).shape[-1]

    # print("path: ", paths_df.loc[0, "path"])
    height, width = image_size = cv2.imread(paths_df.loc[0, "path"], cv2.IMREAD_UNCHANGED).shape


    stitch_per_channels = np.zeros((n_depth, n_row, n_column, n_field_row*height, n_field_column*width))

    for row_idx in range(paths_df.shape[0]):
        path = paths_df.loc[row_idx, "path"]
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        depth = paths_df.loc[row_idx, "depth"]
        well_row = paths_df.loc[row_idx, "well_row"]
        well_column = paths_df.loc[row_idx, "well_column"]
        field_row = paths_df.loc[row_idx, "field_row"]
        field_column = paths_df.loc[row_idx, "field_column"]
        channel = paths_df.loc[row_idx, "channel"]
        
        
        # field_height = n_field_row * height
        # field_width = n_field_column * width
        
        # pos_x = well_row * field_height + field_row * height
        # pos_y = well_column * field_width + field_column * width
        pos_x = field_row * height
        pos_y = field_column * width
        
        stitch_per_channels[depth, well_row, well_column, pos_x:pos_x+height, pos_y:pos_y+width] = img
        
        # stitch_per_channels[depth, well_row, field_row, :, :, field_column, well_column, :] = img

    stitched_root = os.path.join(dataroot, "0_original", stitched_dir)
    os.makedirs(stitched_root, exist_ok=True)
    stitched_mean_root = os.path.join(stitched_root, f"{stitched_dir}_mean")
    os.makedirs(stitched_mean_root, exist_ok=True)
    for j in range(n_row):
        for k in range(n_column):
            for i in range(n_depth):
                # column_letter = get_column_letter(k)
                path_to_stitch = os.path.join(stitched_root, f"Well-{k}-{j}-Depth-{i}.png")
                cv2.imwrite(path_to_stitch, stitch_per_channels[i, j, k].astype(np.uint16))

            path_to_mean = os.path.join(stitched_mean_root, f"Well-{k}-{j}-Depth-mean.png")
            cv2.imwrite(path_to_mean, rescale_image(np.mean(stitch_per_channels[:, j, k], axis=0)))

    paths = os.listdir(stitched_mean_root)
    paths = [os.path.join("data/0_original", stitched_dir, f"{stitched_dir}_mean", p) for p in paths]

    return True, paths, "Success"