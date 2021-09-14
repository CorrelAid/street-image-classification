import os
import numpy as np
import pandas as pd
import geopandas
from mapillary_image_classification.data.osm import define_categories


def split_data(df: geopandas.GeoDataFrame, num_parts: int = 4):
    """
    Split a dataframe into num_parts chunks.

    This can be used to produce multiple dataset files and download the data concurrently
    on multiple computers.
    """
    return np.array_split(df, num_parts)


def balance_data(df: geopandas.GeoDataFrame, group_size, group_cols = ['surface_category', 'smoothness_category']):
    """
    Undersample groups of a dataframe so they have a maximum size of group_size.
    """
    g = df.groupby(group_cols)
    print(g.size())
    smaller_groups_mask = g.size() < group_size
    if sum(smaller_groups_mask) > 0: # if there are groups with smaller size than group_size
        df_smaller = pd.concat( # save all groups which are smaller than the group_size, as these cannot be samples
            [df[(df[group_cols[0]] == group_idx[0]) 
            & (df[group_cols[1]] == group_idx[1])] 
            for group_idx in g.size()[smaller_groups_mask].index])
        df_larger = pd.concat(
            [df[(df[group_cols[0]] == group_idx[0]) 
            & (df[group_cols[1]] == group_idx[1])] 
            for group_idx in g.size()[~smaller_groups_mask].index])
    else:
        df_larger = df
    df_sample = df_larger.groupby(group_cols).sample(group_size, random_state=42) # sample from all groups which are larger than group_size
    if sum(smaller_groups_mask) > 0:
        return pd.concat([df_smaller, df_sample]).sample(frac=1).reset_index(drop=True) # concatenate smaller groups and sampled groups
    else:
        return df_sample.reset_index(drop=True)


def read_geopandas(file_path):
    """
    Reads geopandas from csv in specified filepath
    """
    return geopandas.read_file(
        file_path,
        GEOM_POSSIBLE_NAMES="geometry",
        KEEP_GEOM_COLUMNS="NO"
    ).set_crs(epsg=4326)


def main(df=None):
    print(os.getcwd())
    group_size = 8000
    file_path = "data/raw/dataset_v10/data.csv"
    save_path = "data/raw/dataset_v2"
    os.makedirs(save_path, exist_ok=True)
    if df is None:
        df = read_geopandas(file_path=file_path)
    df = define_categories(df)
    df = balance_data(df, group_size=group_size)
    print(f"Length of dataframe: {len(df)}, approx image size: {len(df) * 150. / 1000000} GB")
    df.to_csv(f"{save_path}/data_balanced_{group_size}.csv", index = False)
    dfs = split_data(df, 4)
    for i, df_tmp in enumerate(dfs):
        df_tmp.to_csv(f"{save_path}/data_balanced_{group_size}_{i}.csv", index = False)
    return df