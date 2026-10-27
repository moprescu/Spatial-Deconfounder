import logging
import os
import re
import shutil
from typing import Literal, List, Tuple
from zipfile import ZipFile
import tempfile

import networkx as nx
import numpy as np
import pandas as pd
from omegaconf.listconfig import ListConfig
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
import plotly.express as px
import plotly.graph_objects as go
import math
from PIL import Image
from io import BytesIO

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from autogluon.core.metrics import get_metric
from autogluon.core.utils import compute_permutation_feature_importance

from collections import defaultdict

def group_by_feat(d):
    grouped = defaultdict(dict)
    for key, value in d.items():
        # split only twice from the right
        parts = key.rsplit("_", 2)
        if len(parts) != 3:
            continue
        else:
            feat, i, j = parts
            try:
                i, j = int(i), int(j)
            except ValueError:
                continue

        grouped[feat][(i, j)] = value
        if feat in d:
            grouped[feat]["grid"] = value
    return dict(grouped)

def group_by_feat_graph(d):
    grouped = defaultdict(dict)
    for key, value in d.items():
        # split only twice from the right
        parts = key.rsplit("_", 2)
        if len(parts) != 3:
            continue
        else:
            feat, i, j = parts
            try:
                j = int(j)
            except ValueError:
                continue

        grouped[feat][j] = value
        if feat in d:
            grouped[feat][0] = value
    return dict(grouped)

def delete_grid(featimp):
    featimp = dict(featimp)
    keys = list(featimp.keys())
    to_delete = []
    
    for key in keys:
        if key.endswith('_grid'):
            base_key = key[:-5]
            val_grid = featimp[key]
            val_base = featimp.get(base_key, float('nan'))

            # Define max_ignore_nan inline:
            if math.isnan(val_base):
                new_val = val_grid
            elif math.isnan(val_grid):
                new_val = val_base
            else:
                new_val = max(val_base, val_grid)

            featimp[base_key] = new_val
            to_delete.append(key)

    for key in to_delete:
        del featimp[key]
    return featimp

def scale_variable(
    x: np.ndarray, scaling: Literal["unit", "standard"] = None
) -> np.ndarray:
    """Scales a variable according to the specified scale."""
    if scaling is None:
        return x

    match scaling:
        case "unit":
            return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
        case "standard":
            return (x - np.nanmean(x)) / np.nanstd(x)
        case _:
            raise ValueError(f"Unknown scale: {scaling}")


def transform_variable(
    x: np.ndarray,
    transform: list[str] | Literal["log", "symlog", "logit"] | None = None,
) -> np.ndarray:
    """Transforms a variable according to the specified transform."""
    if transform is None:
        return x
    elif isinstance(transform, (list, ListConfig)):
        # call recursively
        for t in transform:
            x = transform_variable(x, t)
        return x
    elif transform == "log":
        return np.log(x)
    elif transform == "symlog":
        return np.sign(x) * np.log(np.abs(x))
    elif transform == "logit":
        return np.log(x / (1 - x))
    elif transform.startswith("binary"):
        # regex to extract what's inside the parentheses, e.g., binary(10) -> 10
        cut_value = float(re.search(r"\((.*?)\)", transform).group(1))
        return np.where(x < cut_value, 0.0, 1.0)
    elif transform.startswith("gaussian_noise"):
        # regex to extract what's inside the parentheses, e.g., gaussian_noise(0.1) -> 0.1
        scaler = float(re.search(r"\((.*?)\)", transform).group(1))
        sig = np.nanstd(x)
        return x + np.random.normal(0, sig * scaler, x.shape)
    elif transform.startswith("qbinary"):
        value = float(re.search(r"\((.*?)\)", transform).group(1))
        quantile = np.quantile(x, value)
        return np.where(x < quantile, 0.0, 1.0)
    elif transform.startswith("affine"):
        args = re.search(r"\((.*?)\)", transform).group(1)
        b, m = [float(x) for x in args.split(",")]
        return b + m * x
    else:
        raise ValueError(f"Unknown transform: {transform}")

def make_hpars(checkpoint_names):
    base_config = {
        "model.names": ["timm_image"],
        "model.timm_image.train_transforms": ["resize_shorter_side", "center_crop"],
        "data.categorical.convert_to_text": False,
        "data.numerical.convert_to_text": False,
        "optim.cross_modal_align": "null",
        "data.modality_dropout": 0,
        "model.timm_image.use_learnable_image": False,
        "optim.lemda.turn_on": False,
        "env.strategy": "ddp",
        "env.num_workers": 0,
        # "optim.max_epochs": 1,
        "env.per_gpu_batch_size": 4,
        # "env.inference_batch_size_ratio": 2,
    }
    
    hpars = {}
    learner_names = []

    for checkpoint in checkpoint_names:
        # learner_name = checkpoint.split('_')[0]  # e.g., "vit" from "vit_base_patch16_224"
        learner_name = checkpoint
        learner_names.append(learner_name)

        hpars[learner_name] = {
            **base_config,
            "model.timm_image.checkpoint_name": checkpoint,
        }
    
    hpars["learner_names"] = learner_names
    return hpars
    
def __find_best_gmrf_params(x: np.ndarray, graph: nx.Graph) -> np.ndarray:
    """Select the best param using the penalized likelihood loss of a
    spatial GMLRF smoothing model."""
    lams = 10 ** np.linspace(-3, 1, 20)
    nodelist = np.array(graph.nodes)
    node2ix = {n: i for i, n in enumerate(nodelist)}
    e1 = np.array([node2ix[e[0]] for e in graph.edges])
    e2 = np.array([node2ix[e[1]] for e in graph.edges])
    L = nx.laplacian_matrix(graph).toarray()

    # solves the optiization problem argmin ||beta - x||^2 + lam * beta^T L beta
    def solve(x, lam, L):
        Q = lam * L.copy()
        Q[np.diag_indices_from(Q)] += 1
        L = cholesky(Q, lower=True)
        z = solve_triangular(L, x, lower=True)
        beta = solve_triangular(L.T, z)
        return beta

    losses = {}
    for lam in reversed(lams):
        # TODO: use sparse matrix/ugly dependencies
        beta = solve(x, lam, L)
        sig = np.std(x - beta)

        # compute loss assuming x ~ N(beta, sig**2)
        y_loss = 0.5 * ((x.values - beta) / sig) ** 2 + np.log(sig)

        # diffs ~ N(0, sig**2 / lam)
        l = lam / sig**2
        diff_loss = 0.5 * l * (beta[e1] - beta[e2]) ** 2 - 0.5 * np.log(l)

        penalty_loss = len(e1) * l + (1 / sig**2)

        # total_loss
        losses[lam] = y_loss.sum() + diff_loss.sum() + penalty_loss

    best_lam = min(lams, key=lambda l: losses[l])

    logging.info(f"Best lambda: {best_lam:.4f}")
    losses_ = {np.round(k, 4): np.round(v, 4) for k, v in losses.items()}
    logging.info(f"Losses: {losses_}")

    return best_lam


def generate_noise_like_by_penalty(x: pd.Series, graph: nx.Graph) -> np.ndarray:
    """Injects noise into residuals using a Gaussian Markov Random Field."""
    # find best smoothness param from penalized likelihood
    res_sig = np.nanstd(x)
    res_standard = x / res_sig
    res_graph = nx.subgraph(graph, x.index)
    best_lam = __find_best_gmrf_params(res_standard, res_graph)

    # make spatial noise from GMRF
    Q = best_lam * nx.laplacian_matrix(graph).toarray()
    Q[np.diag_indices_from(Q)] += 1
    Z = np.random.randn(Q.shape[0])
    L = cholesky(Q, lower=True)
    noise = solve_triangular(L, Z, lower=True).T
    noise = noise / noise.std() * res_sig

    return noise


def generate_noise_like(
    x: np.ndarray, edge_list: np.ndarray, attempts: int = 10
) -> np.ndarray:
    """Injects noise into residuals using a Gaussian Markov Random Field."""
    n = len(x)
    nbrs_means = get_nbrs_means(x, edge_list)
    rho = get_nbrs_corr(x, edge_list, nbrs_means=nbrs_means)

    # 1. Build precision matrix
    # Arrays to hold the data, row indices, and column indices for Q
    data = []
    rows = []
    cols = []

    # Off-diagonal entries and compute degree for diagonal
    degree = np.zeros(n)
    for i, j in edge_list:
        data.extend([-rho, -rho])
        rows.extend([i, j])
        cols.extend([j, i])
        degree[i] += 1
        degree[j] += 1

    # Add diagonal entries
    data.extend(np.maximum(degree, 0.1))
    rows.extend(range(n))
    cols.extend(range(n))

    # build precision matrix
    Q = csc_matrix((data, (rows, cols)), shape=(n, n))
    factorization = splu(Q)

    best_result = np.inf
    best_corr = None
    best_attempt = None
    for _ in range(attempts):
        noise = factorization.solve(np.random.normal(size=n))
        noise_nbrs_means = get_nbrs_means(noise, edge_list)
        corr = np.corrcoef(noise, noise_nbrs_means)[0, 1]
        if np.abs(rho - corr) < best_result:
            best_result = np.abs(rho - corr)
            best_corr = corr
            best_attempt = noise

    # scale noise to have same variance as residuals
    noise = best_attempt / best_attempt.std() * np.nanstd(x)

    return noise


def get_nbrs_means(x: np.ndarray, edge_list: np.ndarray) -> np.ndarray:
    """Computes the mean of each node's neighbors."""
    nbrs = [[] for _ in range(x.shape[0])]
    for i, j in edge_list:
        nbrs[i].append(j)
        nbrs[j].append(i)

    xbar = np.nanmean(x)
    nbrs_means = np.zeros(len(x))
    for i in range(len(x)):
        if not nbrs[i]:
            nbrs_means[i] = xbar if np.isnan(x[i]) else x[i]
        else:
            valid = [x_j for x_j in x[nbrs[i]] if not np.isnan(x_j)]
            if valid:
                nbrs_means[i] = np.mean(valid)
            else:
                nbrs_means[i] = xbar if np.isnan(x[i]) else x[i]

    return nbrs_means


def get_nbrs_corr(
    x: np.ndarray, edge_list: np.ndarray, nbrs_means: np.ndarray | None = None
) -> float:
    """Computes the correlation between each node and its neighbors."""
    if nbrs_means is None:
        nbrs_means = get_nbrs_means(x, edge_list)
    x_ = x.copy()
    x_[np.isnan(x_)] = nbrs_means[np.isnan(x_)]
    rho = np.corrcoef(x_, nbrs_means)[0, 1]
    return float(rho)


def moran_I(x: np.ndarray, edge_list: np.ndarray) -> float:
    x = x.copy()

    xbar = np.nanmean(x)
    nbrs_means = get_nbrs_means(x, edge_list)

    x_ = x.copy()
    x_[np.isnan(x_)] = nbrs_means[np.isnan(x_)]

    # Subtract mean from attribute values
    x_diff = x_ - xbar

    # Compute numerator: sum of product of weight and pair differences from mean
    src_diff = x_diff[edge_list[:, 0]]
    dst_diff = x_diff[edge_list[:, 1]]
    numerator = np.sum(src_diff * dst_diff) * len(x_diff)

    # Compute denominator: sum of squared differences from mean
    denominator = np.sum(x_diff**2) * len(edge_list)

    return float(numerator / denominator)


def double_zip_folder(folder_path, output_path):
    # Create a temporary zip file
    shutil.make_archive(output_path, "zip", folder_path)

    # Zip the temporary zip file
    zipzip_path = output_path + ".zip.zip"
    with ZipFile(zipzip_path, "w") as f:
        f.write(output_path + ".zip")

    # Remove the temporary zip file
    os.remove(output_path + ".zip")

    return zipzip_path


def sort_dict(d: dict) -> dict[str, float]:
    return {
        str(k): float(v) for k, v in sorted(d.items(), key=lambda x: x[1], reverse=True)
    }


def spatial_train_test_split(
    graph: nx.Graph, init_frac: float, levels: int, buffer: int
):
    logging.info(f"Selecting tunning split removing {levels} nbrs from val. pts.")

    # make dict of neighbors from graph
    node_list = np.array(graph.nodes())
    n = len(node_list)
    nbrs = {node: set(graph.neighbors(node)) for node in node_list}

    # first find the centroid of the tuning subgraph
    num_tuning_centroids = int(init_frac * n)
    tuning_nodes = np.random.choice(n, size=num_tuning_centroids, replace=False)
    tuning_nodes = set(node_list[tuning_nodes])

    # not remove all neighbors of the tuning centroids from the training data
    for _ in range(levels):
        tmp = tuning_nodes.copy()
        for node in tmp:
            for nbr in nbrs[node]:
                tuning_nodes.add(nbr)
    tuning_nodes = list(tuning_nodes)

    # buffer
    buffer_nodes = set(tuning_nodes.copy())
    for _ in range(buffer):
        tmp = buffer_nodes.copy()
        for node in tmp:
            for nbr in nbrs[node]:
                buffer_nodes.add(nbr)
    buffer_nodes = list(set(buffer_nodes))

    return tuning_nodes, buffer_nodes

def spatial_train_test_split_radius(
    graph: nx.Graph,
    init_frac: float,
    levels: int,
    buffer: int = 0,
    seed: int | None = None,
    radius: int = 1,
) -> tuple[list, list, list]:
    """Split restricted to nodes with max neighbors."""
    logging.info(
        f"Selecting tuning split with {levels} levels and {buffer} buffer, restricted to max-degree nodes."
    )
    node_list = list(graph.nodes())
    n = len(node_list)

    # dict of neighbors
    nbrs = {node: get_k_hop_neighbors(graph, node, radius) for node in node_list}
    
    # find nodes with max neighbors
    nbr_counts = {node: len(neigh) for node, neigh in nbrs.items()}
    max_count = max(nbr_counts.values())
    max_nodes = [node for node, cnt in nbr_counts.items() if cnt == max_count]
    
    # restrict neighbor sets to only max-degree nodes
    nbrs = {node: set(graph.neighbors(node)) for node in node_list}
    nbrs = {node: neigh & set(max_nodes) for node, neigh in nbrs.items()}

    # restrict everything to only these nodes
    node_list = max_nodes
    n = len(node_list)

    # pick tuning centroids from max nodes
    num_tuning_centroids = int(init_frac * n)
    rng = np.random.default_rng(seed)
    tuning_nodes = rng.choice(node_list, size=num_tuning_centroids, replace=False)
    tuning_nodes = set(tuning_nodes)

    # expand tuning set
    for _ in range(levels):
        tmp = tuning_nodes.copy()
        for node in tmp:
            tuning_nodes.update(nbrs[node])
    tuning_nodes = list(tuning_nodes)

    # buffer set
    buffer_nodes = set(tuning_nodes)
    for _ in range(buffer):
        tmp = buffer_nodes.copy()
        for node in tmp:
            buffer_nodes.update(nbrs[node])
    buffer_nodes = list(buffer_nodes - set(tuning_nodes))

    # training = remaining max-degree nodes not in tuning or buffer
    training_nodes = list(set(node_list) - set(tuning_nodes) - set(buffer_nodes))
    logging.info(
        f"Length of training, tuning and buffer: {len(training_nodes)} and {len(tuning_nodes)} and {len(buffer_nodes)}"
    )
    return training_nodes, tuning_nodes, buffer_nodes

def unpack_covariates(groups: dict) -> list[str]:
    covariates = []
    for c in groups:
        if isinstance(c, dict) and len(c) == 1:
            covariates.extend(next(iter(c.values())))
        elif isinstance(c, str):
            covariates.append(c)
        else:
            msg = "covar group must me dict with a single element or str"
            logging.error(msg)
            raise ValueError(msg)

    return covariates

def get_k_hop_neighbors(graph, node, k):
    """
    Find all neighbors within k hops of a given node.
    
    Args:
        graph: NetworkX graph object
        node: The starting node
        k: Maximum number of hops to consider
    
    Returns:
        set: All nodes within k hops of the given node (excluding the node itself)
    """
    
    # Get all nodes within k hops (including the starting node)
    ego_subgraph = nx.ego_graph(graph, node, radius=k)
    
    # Return all nodes except the starting node itself
    return set(ego_subgraph.nodes()) - {node}


def create_map(all_cols : List[str], map_df: pd.DataFrame, output_dir: str, edge_list):
    
    # Create initial figure with first variable
    fig = go.Figure()
    
    # Convert geometries to GeoJSON format
    import json
    from shapely.geometry import mapping
    
    # Create GeoJSON from the geometry column
    geojson_features = []
    for idx, row in map_df.iterrows():
        feature = {
            "type": "Feature",
            "properties": {"id": str(idx)},
            "geometry": mapping(row['geometry'])  # Use shapely.geometry.mapping
        }
        geojson_features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": geojson_features
    }
    
    # Pre-convert index to strings once
    locations = map_df.index.astype(str)
    
    # Add traces for each variable (initially all invisible except first)
    for i, col in enumerate(all_cols):
        fig.add_trace(
            go.Choroplethmapbox(
                geojson=geojson,
                locations=locations,  # Use pre-converted locations
                z=map_df[col],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=col),
                hovertemplate=f'<b>%{{location}}</b><br>{col}: %{{z}}<extra></extra>',
                name=col,
                visible=True if i == 0 else False
            )
        )
    
    # Create dropdown buttons
    buttons = []
    for i, col in enumerate(all_cols):
        visibility = [False] * len(all_cols)
        visibility[i] = True
        buttons.append(
            dict(
                label=col,
                method="update",
                args=[{"visible": visibility},
                      {"title": f"World Map: {col}"}]
            )
        )
    
    # Calculate bounds from geometries for auto-zoom
    # Get bounds from each geometry and find overall min/max
    all_bounds = [geom.bounds for geom in map_df['geometry']]  # [(minx, miny, maxx, maxy), ...]
    
    # Calculate overall bounds
    minx = min(bounds[0] for bounds in all_bounds)
    miny = min(bounds[1] for bounds in all_bounds)
    maxx = max(bounds[2] for bounds in all_bounds)
    maxy = max(bounds[3] for bounds in all_bounds)
    
    bounds = [minx, miny, maxx, maxy]
    
    # Calculate center point
    center_lat = (bounds[1] + bounds[3]) / 2  # (miny + maxy) / 2
    center_lon = (bounds[0] + bounds[2]) / 2  # (minx + maxx) / 2
    
    # Calculate zoom level based on bounds span
    lat_span = bounds[3] - bounds[1]
    lon_span = bounds[2] - bounds[0]
    
    # Estimate zoom level (this is approximate)
    max_span = max(lat_span, lon_span)
    if max_span > 100:
        zoom = 1
    elif max_span > 50:
        zoom = 2
    elif max_span > 20:
        zoom = 3
    elif max_span > 10:
        zoom = 4
    elif max_span > 5:
        zoom = 5
    elif max_span > 2:
        zoom = 6
    elif max_span > 1:
        zoom = 7
    else:
        zoom = 8
    
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.1,
                y=1.15,
            )
        ],
        mapbox=dict(
            style="carto-positron",
            zoom=zoom,
            center=dict(lat=center_lat, lon=center_lon)
        ),
        height=600,
        title=f"World Map: {all_cols[0]}"
    )
    
    fig.write_html(f"{output_dir}/map.html", include_plotlyjs=True)

    
def haversine(lat1, lon1, lat2, lon2):
    """Accurate great-circle distance in km."""
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def add_neighbor_columns(df, nbrs, nbr_cols, max_nbrs, coords, a=None, change=None, treatment=None,
                                 is_binary_treatment=None, spaceenv=None, get_t_pct=None, spline_basis=None,
                                 extra_colnames=None, covariates=None, nbr_idx=None, nbr_half_seed=42):
    # Initialize neighbor columns with NaN instead of 0
    dftrain = df.copy()
    for feature in nbr_cols:
        for i in range(1, max_nbrs + 1):
            dftrain[f"{feature}_nbr_{i}"] = float('nan')

    nbr_half_rng = np.random.default_rng(nbr_half_seed) if change == "nbr_half" else None

    for node in dftrain.index:
        lat1, lon1 = coords.loc[node, ['latitude', 'longitude']]
        neighbors = list(nbrs.get(node, []))
        
        # Sort neighbors by haversine distance to the node
        neighbors_sorted = sorted(
            neighbors,
            key=lambda nbr: haversine(lat1, lon1, coords.loc[nbr, 'latitude'], coords.loc[nbr, 'longitude'])
        )
        
        # Filter neighbors that exist in dftrain
        valid_neighbors = [nbr for nbr in neighbors_sorted if nbr in dftrain.index]

        # For nbr_half: pre-compute per-node random 0/1 mask across max_nbrs slots (half 1s, half 0s)
        if change == "nbr_half":
            half_mask = np.zeros(max_nbrs, dtype=int)
            perm = nbr_half_rng.permutation(max_nbrs)
            half_mask[perm[: max_nbrs // 2]] = 1
        else:
            half_mask = None

        for feature in nbr_cols:
            # Get feature values for all valid neighbors
            neighbor_values = [dftrain.at[nbr_node, feature] for nbr_node in valid_neighbors]
            
            # Calculate mean of neighbor values (if any exist)
            if neighbor_values:
                mean_value = sum(neighbor_values) / len(neighbor_values)
            else:
                mean_value = float('nan')  # or 0 if you prefer
            
            # Fill neighbor columns
            for i in range(max_nbrs):
                colname = f"{feature}_nbr_{i+1}"
                if i < len(valid_neighbors):
                    # Use actual neighbor value
                    nbr_node = valid_neighbors[i]
                    dftrain.at[node, colname] = dftrain.at[nbr_node, feature]
                else:
                    # Use mean of all neighbors for missing positions
                    dftrain.at[node, colname] = mean_value
                    
                if change == "nbr" or (change == "nbr_k" and nbr_idx is not None and i == nbr_idx - 1) or change == "nbr_half":
                    effective_a = half_mask[i] if change == "nbr_half" else a
                    if treatment == feature:
                        extra_value = effective_a
                    elif feature in extra_colnames:
                        if not is_binary_treatment and spaceenv.bsplines:
                            t_a_pct = get_t_pct(effective_a)
                            # Find which spline basis this column corresponds to
                            col_idx = extra_colnames.index(feature)
                            extra_value = spline_basis[col_idx](t_a_pct)
                        elif is_binary_treatment and spaceenv.binary_treatment_iteractions:
                            col_idx = extra_colnames.index(feature)
                            covariate_val = dftrain.at[nbr_node, covariates[col_idx]] if i < len(valid_neighbors) else dftrain.at[node, covariates[col_idx]]
                            extra_value = covariate_val * effective_a

                    dftrain.at[node, colname] = extra_value
                    
    if change == "center":
        dftrain[treatment] = a
    return dftrain


def add_neighbor_columns_grid(dftrain, nbrs, nbr_cols, max_nbrs, coords):
    """
    For each row in dftrain, add up to 8 neighboring values for each feature in `nbr_cols`,
    ordered clockwise using (row, col) positions in `coords`.

    Assumes dftrain and coords share the same index and that coords has 'row' and 'col' columns.
    """
    # Clockwise neighbor offsets: NW, N, NE, E, SE, S, SW, W
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, 1),  ( 1, 1), ( 1, 0),
        ( 1, -1), ( 0, -1)
    ]

    # Prepare fast lookup from (row, col) → index
    coord_to_index = {
        (r, c): idx for idx, (r, c) in coords[["row", "col"]].iterrows()
    }

    for feature in nbr_cols:
        for i in range(1, max_nbrs + 1):
            dftrain[f"{feature}_nbr_{i}"] = 0  # default to 0

    for node in dftrain.index:
        row, col = coords.loc[node, ["row", "col"]]

        for i, (dr, dc) in enumerate(neighbor_offsets[:max_nbrs]):
            neighbor_pos = (row + dr, col + dc)
            if neighbor_pos in coord_to_index:
                nbr_node = coord_to_index[neighbor_pos]
                if nbr_node in dftrain.index:
                    for feature in nbr_cols:
                        colname = f"{feature}_nbr_{i+1}"
                        dftrain.at[node, colname] = dftrain.at[nbr_node, feature]

    return dftrain


def get_top_common_alternating(dict1, dict2, top_k=10):
    # Sort both dicts by importance
    sorted1 = sorted(dict1.items(), key=lambda x: x[1], reverse=True)
    sorted2 = sorted(dict2.items(), key=lambda x: x[1], reverse=True)
    
    selected = []
    seen = set()
    i = j = 0
    
    while len(selected) < top_k and (i < len(sorted1) or j < len(sorted2)):
        # Try from first dict
        if i < len(sorted1):
            feature, score = sorted1[i]
            if feature in dict2 and feature not in seen:
                selected.append((feature, score, dict2[feature]))
                seen.add(feature)
            i += 1
        
        if len(selected) >= top_k:
            break
            
        # Try from second dict
        if j < len(sorted2):
            feature, score = sorted2[j]
            if feature in dict1 and feature not in seen:
                selected.append((feature, dict1[feature], score))
                seen.add(feature)
            j += 1
    
    return selected



def save_maps_pdf(all_cols: List[str], map_df: gpd.GeoDataFrame, output_dir: str, edge_list=None, new_dir="maps_pdf"):
    """
    Create separate PDF files for each feature's map visualization using matplotlib.
    """
    map_df = map_df.copy()
    tolerance = 0.01
    map_df["geometry"] = map_df["geometry"].apply(lambda geom: geom.simplify(tolerance, preserve_topology=True))
    
    # Create PDF directory if it doesn't exist
    pdf_dir = os.path.join(output_dir, new_dir)
    os.makedirs(pdf_dir, exist_ok=True)
    
    for col in all_cols:
        values = pd.to_numeric(map_df[col], errors="coerce")
        assert values.notna().any(), f"All values for {col} are NaN"
        
        # Standardize and clip to ±3 std
        mean = values.mean()
        std = values.std()
        z_vals = ((values - mean) / std).clip(lower=-3, upper=3)
        
        # Normalize for color mapping
        norm = Normalize(vmin=-3, vmax=3)
        cmap = plt.get_cmap('RdBu_r')
        colors = cmap(norm(z_vals.fillna(0)))  # fallback to 0 for missing
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        map_df.plot(ax=ax, color=colors, edgecolor='black', linewidth=0.0)
        
        # Force matplotlib to draw the figure to get accurate measurements
        fig.canvas.draw()
        
        # Get the actual bounds of the plotted map and axis limits
        map_bounds = map_df.total_bounds  # [minx, miny, maxx, maxy]
        ax_xlim = ax.get_xlim()
        ax_ylim = ax.get_ylim()
        
        # Calculate the data range vs axis range ratios
        map_height = map_bounds[3] - map_bounds[1]
        axis_height = ax_ylim[1] - ax_ylim[0]
        
        map_width = map_bounds[2] - map_bounds[0]  
        axis_width = ax_xlim[1] - ax_xlim[0]
        
        # Use the smaller ratio to account for padding/margins
        height_ratio = map_height / axis_height
        width_ratio = map_width / axis_width
        shrink = min(height_ratio, width_ratio) * 0.65  # Further reduce
        
        # Add colorbar with height matching the map
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []  # dummy array for colorbar
        cbar = fig.colorbar(sm, ax=ax, shrink=shrink)
        
        ax.axis('off')
        
        # Save to PDF
        pdf_filename = f"{col.replace(' ', '_').replace('/', '_')}_map.pdf"
        pdf_path = os.path.join(pdf_dir, pdf_filename)
        try:
            plt.savefig(pdf_path, bbox_inches='tight')
            # print(f"Saved: {pdf_path}")
        except Exception as e:
            print(f"Error saving {pdf_path}: {e}")
        finally:
            plt.close(fig)
    
    # print(f"All maps saved to: {pdf_dir}")
    
def save_maps_pdf_grid(all_cols: List[str], map_df: pd.DataFrame, output_dir: str, new_dir="maps_pdf"):
    """
    Create separate PDF files for each feature's map visualization using matplotlib
    using 2D grid layout (based on 'row' and 'col' columns).
    Places each value directly at its specified row, col position.
    """
    map_df = map_df.copy()
    
    # Create PDF directory if it doesn't exist
    pdf_dir = os.path.join(output_dir, new_dir)
    os.makedirs(pdf_dir, exist_ok=True)
    
    if "row" not in map_df.columns or "col" not in map_df.columns:
        raise ValueError("map_df must contain 'row' and 'col' columns")
    
    # Infer grid shape
    max_row = map_df["row"].max()
    max_col = map_df["col"].max()
    grid_shape = (max_row + 1, max_col + 1)
    
    for col in all_cols:
        values = pd.to_numeric(map_df[col], errors="coerce")
        assert values.notna().any(), f"All values for {col} are NaN"
        
        # Standardize and clip to ±3 std
        mean = values.mean()
        std = values.std()
        z_vals = ((values - mean) / std).clip(lower=-3, upper=3)
        
        # Create empty grid filled with NaN
        grid = np.full(grid_shape, np.nan)
        
        # Place each value at its specified (row, col) position
        for idx, (_, row_data) in enumerate(map_df.iterrows()):
            row_pos = int(row_data["row"])
            col_pos = int(row_data["col"])
            if not pd.isna(z_vals.iloc[idx]):
                grid[row_pos, col_pos] = z_vals.iloc[idx]
        
        # Normalize for color mapping
        norm = Normalize(vmin=-3, vmax=3)
        cmap = plt.get_cmap('RdBu_r')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(grid, cmap=cmap, norm=norm)
        
        # Add colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        fig.colorbar(sm, ax=ax, shrink=0.7)
        
        ax.axis('off')
        
        # Save to PDF
        pdf_filename = f"{col.replace(' ', '_').replace('/', '_')}_map.pdf"
        pdf_path = os.path.join(pdf_dir, pdf_filename)
        try:
            plt.savefig(pdf_path, bbox_inches='tight')
            # print(f"Saved: {pdf_path}")
        except Exception as e:
            print(f"Error saving {pdf_path}: {e}")
        finally:
            plt.close(fig)
    
    # print(f"All maps saved to: {pdf_dir}")

def feature_importance(
    data=None,
    label=None,
    predict_func=None,
    features: list = None,
    subsample_size: int = 5000,
    time_limit: float = None,
    num_shuffle_sets: int = None,
    include_confidence_band: bool = True,
    confidence_level: float = 0.99,
    silent: bool = False,
    eval_metric: str = "rmse",
):
    """
    Calculates feature importance scores for the given model via permutation importance. Refer to https://explained.ai/rf-importance/ for an explanation of permutation importance.
    A feature's importance score represents the performance drop that results when the model makes predictions on a perturbed copy of the data where this feature's values have been randomly shuffled across rows.
    A feature score of 0.01 would indicate that the predictive performance dropped by 0.01 when the feature was randomly shuffled.
    The higher the score a feature has, the more important it is to the model's performance.
    If a feature has a negative score, this means that the feature is likely harmful to the final model, and a model trained with the feature removed would be expected to achieve a better predictive performance.
    Note that calculating feature importance can be a very computationally expensive process, particularly if the model uses hundreds or thousands of features. In many cases, this can take longer than the original model training.
    To estimate how long `feature_importance(model, data, features)` will take, it is roughly the time taken by `predict_proba(data, model)` multiplied by the number of features.

    Note: For highly accurate importance and p_value estimates, it is recommended to set `subsample_size` to at least 5000 if possible and `num_shuffle_sets` to at least 10.

    Parameters
    ----------
    data : :class:`pd.DataFrame` or numpy array
        The feature data (X) used to calculate feature importance scores. This should contain only the feature columns 
        and should NOT include the label column. The data will be used for permutation testing where individual 
        features are shuffled to measure their impact on model performance.
        Do not pass the training data through this argument, as the feature importance scores calculated will be biased due to overfitting.
            More accurate feature importances will be obtained from new data that was held-out during training.
    label : :class:`pd.Series`, :class:`pd.DataFrame`, or numpy array
        The true labels (y) corresponding to the data. These are the ground truth values that will be compared 
        against the model's predictions to calculate the evaluation metric during permutation importance testing.
    predict_func : Callable[..., np.ndarray]
        Function that computes model predictions or prediction probabilities on input data.
        Output must be in the form of a numpy ndarray or pandas Series or DataFrame.
        Output `y_pred` must be in a form acceptable as input to `eval_metric(y, y_pred)`.
        If using a fit model object, this is typically `model.predict` or `model.predict_proba`, depending on the `eval_metric` being used.
        If `eval_metric.needs_pred==True`, use `model.predict`, otherwise use `model.predict_proba`.
    features : list, default = None
        List of str feature names that feature importances are calculated for and returned, specify None to get all feature importances.
        If you only want to compute feature importances for some of the features, you can pass their names in as a list of str.
        Valid feature names change depending on the `feature_stage`.
            To get the list of feature names for `feature_stage='original'`, call `predictor.feature_metadata_in.get_features()`.
            To get the list of feature names for `feature_stage='transformed'`, call `list(predictor.transform_features().columns)`.
            To get the list of feature names for `feature_stage=`transformed_model`, call `list(predictor.transform_features(model={model_name}).columns)`.
        [Advanced] Can also contain tuples as elements of (feature_name, feature_list) form.
            feature_name can be any string so long as it is unique with all other feature names / features in the list.
            feature_list can be any list of valid features in the data.
            This will compute importance of the combination of features in feature_list, naming the set of features in the returned DataFrame feature_name.
            This importance will differ from adding the individual importances of each feature in feature_list, and will be more accurate to the overall group importance.
            Example: ['featA', 'featB', 'featC', ('featBC', ['featB', 'featC'])]
            In this example, the importance of 'featBC' will be calculated by jointly permuting 'featB' and 'featC' together as if they were a single two-dimensional feature.
    subsample_size : int, default = 5000
        The number of rows to sample from `data` when computing feature importance.
        If `subsample_size=None` or `data` contains fewer than `subsample_size` rows, all rows will be used during computation.
        Larger values increase the accuracy of the feature importance scores.
        Runtime linearly scales with `subsample_size`.
    time_limit : float, default = None
        Time in seconds to limit the calculation of feature importance.
        If None, feature importance will calculate without early stopping.
        A minimum of 1 full shuffle set will always be evaluated. If a shuffle set evaluation takes longer than `time_limit`, the method will take the length of a shuffle set evaluation to return regardless of the `time_limit`.
    num_shuffle_sets : int, default = None
        The number of different permutation shuffles of the data that are evaluated.
        Larger values will increase the quality of the importance evaluation.
        It is generally recommended to increase `subsample_size` before increasing `num_shuffle_sets`.
        Defaults to 5 if `time_limit` is None or 10 if `time_limit` is specified.
        Runtime linearly scales with `num_shuffle_sets`.
    include_confidence_band: bool, default = True
        If True, returned DataFrame will include two additional columns specifying confidence interval for the true underlying importance value of each feature.
        Increasing `subsample_size` and `num_shuffle_sets` will tighten the confidence interval.
    confidence_level: float, default = 0.99
        This argument is only considered when `include_confidence_band` is True, and can be used to specify the confidence level used for constructing confidence intervals.
        For example, if `confidence_level` is set to 0.99, then the returned DataFrame will include columns 'p99_high' and 'p99_low' which indicates that the true feature importance will be between 'p99_high' and 'p99_low' 99% of the time (99% confidence interval).
        More generally, if `confidence_level` = 0.XX, then the columns containing the XX% confidence interval will be named 'pXX_high' and 'pXX_low'.
    silent : bool, default = False
        Whether to suppress logging output.
    eval_metric: str, default = "rmse"
        Evaluation metric name. Uses autogluon.core.metrics to get Scorer object.

    Returns
    -------
    :class:`pd.DataFrame` of feature importance scores with 6 columns:
        index: The feature name.
        'importance': The estimated feature importance score.
        'stddev': The standard deviation of the feature importance score. If NaN, then not enough num_shuffle_sets were used to calculate a variance.
        'p_value': P-value for a statistical t-test of the null hypothesis: importance = 0, vs the (one-sided) alternative: importance > 0.
            Features with low p-value appear confidently useful to the predictor, while the other features may be useless to the predictor (or even harmful to include in its training data).
            A p-value of 0.01 indicates that there is a 1% chance that the feature is useless or harmful, and a 99% chance that the feature is useful.
            A p-value of 0.99 indicates that there is a 99% chance that the feature is useless or harmful, and a 1% chance that the feature is useful.
        'n': The number of shuffles performed to estimate importance score (corresponds to sample-size used to determine confidence interval for true score).
        'pXX_high': Upper end of XX% confidence interval for true feature importance score (where XX=99 by default).
        'pXX_low': Lower end of XX% confidence interval for true feature importance score.
    """

    if num_shuffle_sets is None:
        num_shuffle_sets = 10 if time_limit else 5

    fi_df = compute_permutation_feature_importance(
        X=data,
        y=label,
        predict_func=predict_func,
        eval_metric=get_metric(eval_metric),
        features=features,
        subsample_size=subsample_size,
        num_shuffle_sets=num_shuffle_sets,
        time_limit=time_limit,
        silent=silent,
    )


    if include_confidence_band:
        if confidence_level <= 0.5 or confidence_level >= 1.0:
            raise ValueError("confidence_level must lie between 0.5 and 1.0")
        ci_str = "{:0.0f}".format(confidence_level * 100)
        import scipy.stats

        num_features = len(fi_df)
        ci_low_dict = dict()
        ci_high_dict = dict()
        for i in range(num_features):
            fi = fi_df.iloc[i]
            mean = fi["importance"]
            stddev = fi["stddev"]
            n = fi["n"]
            if stddev == np.nan or n == np.nan or mean == np.nan or n == 1:
                ci_high = np.nan
                ci_low = np.nan
            else:
                t_val = scipy.stats.t.ppf(1 - (1 - confidence_level) / 2, n - 1)
                ci_high = mean + t_val * stddev / math.sqrt(n)
                ci_low = mean - t_val * stddev / math.sqrt(n)
            ci_high_dict[fi.name] = ci_high
            ci_low_dict[fi.name] = ci_low
        high_str = "p" + ci_str + "_high"
        low_str = "p" + ci_str + "_low"
        fi_df[high_str] = pd.Series(ci_high_dict)
        fi_df[low_str] = pd.Series(ci_low_dict)
    return fi_df

# def create_grid_features_compact(dftrain, radius=2, fill_missing='neighbors_mean', a=None, change=None, treatment=None):
#     """
#     Efficiently create grid neighborhoods for each point as 2D arrays (grids),
#     storing them in a new column per feature. Also creates individual columns
#     for each cell position in the neighborhood.
    
#     Parameters
#     ----------
#     dftrain : pd.DataFrame
#         Input DataFrame with index of format "{row}_{col}".
#     radius : int
#         Neighborhood radius (grid size = 2*radius + 1).
#     fill_missing : str
#         How to fill missing values:
#         'mean'            -> fill NaN with column mean
#         'median'          -> fill NaN with column median
#         'zero'            -> fill NaN with 0
#         'neighbors_mean'  -> fill NaN with mean of adjacent neighbors
#         'neighbors_median'-> fill NaN with median of adjacent neighbors
        
#     Returns
#     -------
#     tuple
#         (pd.DataFrame, str): DataFrame with new columns for each feature containing 
#         2D grids (tmp file paths) and individual cell columns, plus temp directory path.
#     """

#     # Parse coordinates from index
#     index_parts = dftrain.index.to_series().str.split('_', expand=True)
#     rows = index_parts[0].astype(int).values
#     cols = index_parts[1].astype(int).values

#     # Determine grid bounds
#     min_row, max_row = rows.min(), rows.max()
#     min_col, max_col = cols.min(), cols.max()
#     n_rows = max_row - min_row + 1
#     n_cols = max_col - min_col + 1

#     feature_cols = dftrain.columns

#     # Precompute global fill values for mean/median/zero
#     fill_values = {}
#     for col in feature_cols:
#         if fill_missing == 'mean':
#             val = dftrain[col].mean()
#         elif fill_missing == 'median':
#             val = dftrain[col].median()
#         elif fill_missing == 'zero':
#             val = 0
#         else:
#             val = np.nan  # for neighbor-based filling
#         fill_values[col] = 0 if (pd.isna(val) and fill_missing not in ('neighbors_mean', 'neighbors_median')) else val

#     temp_dir = tempfile.mkdtemp(prefix='grid_features_')
#     result_df = dftrain.copy()

#     # For each feature
#     for col in feature_cols:
#         # Initialize full grid
#         if fill_missing in ('neighbors_mean', 'neighbors_median'):
#             full_grid = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
#             full_grid[rows - min_row, cols - min_col] = dftrain[col].values
#         else:
#             full_grid = np.full((n_rows, n_cols), fill_values[col], dtype=np.float64)
#             full_grid[rows - min_row, cols - min_col] = dftrain[col].fillna(fill_values[col]).values

#         grids = []
#         cell_data = {f"{col}_{dr}_{dc}": []
#                      for dr in range(-radius, radius + 1)
#                      for dc in range(-radius, radius + 1)}

#         # Process each point
#         for i, (r, c) in enumerate(zip(rows - min_row, cols - min_col)):
#             r_start, r_end = r - radius, r + radius + 1
#             c_start, c_end = c - radius, c + radius + 1

#             pad_top = max(0, -r_start)
#             pad_left = max(0, -c_start)
#             pad_bottom = max(0, r_end - n_rows)
#             pad_right = max(0, c_end - n_cols)

#             r_start_clamped = max(0, r_start)
#             r_end_clamped = min(n_rows, r_end)
#             c_start_clamped = max(0, c_start)
#             c_end_clamped = min(n_cols, c_end)

#             neighborhood = full_grid[r_start_clamped:r_end_clamped,
#                                      c_start_clamped:c_end_clamped]

#             # Pad with NaN so edges don't get false fill values
#             if pad_top or pad_left or pad_bottom or pad_right:
#                 neighborhood = np.pad(
#                     neighborhood,
#                     ((pad_top, pad_bottom), (pad_left, pad_right)),
#                     mode='constant',
#                     constant_values=np.nan
#                 )

#             # Handle missing values
#             if fill_missing in ('neighbors_mean', 'neighbors_median'):
#                 mask = np.isnan(neighborhood)
#                 if mask.any():
#                     for rr, cc in zip(*np.where(mask)):
#                         r0, r1 = max(0, rr - 1), min(neighborhood.shape[0], rr + 2)
#                         c0, c1 = max(0, cc - 1), min(neighborhood.shape[1], cc + 2)

#                         neighbor_vals = neighborhood[r0:r1, c0:c1].flatten()
#                         neighbor_vals = neighbor_vals[~np.isnan(neighbor_vals)]

#                         if len(neighbor_vals) > 0:
#                             if fill_missing == 'neighbors_mean':
#                                 neighborhood[rr, cc] = neighbor_vals.mean()
#                             else:  # neighbors_median
#                                 neighborhood[rr, cc] = np.median(neighbor_vals)
#                         else:
#                             neighborhood[rr, cc] = 0  # fallback
#             elif fill_missing == 'zero':
#                 neighborhood = np.nan_to_num(neighborhood, nan=0.0)
#             # mean/median handled globally

#             # Store cell values
#             for dr in range(-radius, radius + 1):
#                 for dc in range(-radius, radius + 1):
#                     cell_value = neighborhood[dr + radius, dc + radius]
#                     cell_data[f"{col}_{dr}_{dc}"].append(cell_value)

#             # Save normalized grid image
#             norm_grid = ((neighborhood - neighborhood.min()) /
#                          (neighborhood.ptp() + 1e-8) * 255).astype(np.uint8)
#             img = Image.fromarray(norm_grid, mode='L')
#             file_path = os.path.join(temp_dir, f'grid_{col}_{dftrain.index[i]}.png')
#             img.save(file_path, format='PNG')
#             grids.append(file_path)

#         # Add results
#         result_df[f"{col}_grid"] = grids
#         for cell_col_name, cell_values in cell_data.items():
#             result_df[cell_col_name] = cell_values

#     return result_df, temp_dir

def create_grid_features_compact(dftrain, radius=2, fill_missing='mean', a=None, change=None, treatment=None,
                                 is_binary_treatment=None, spaceenv=None, get_t_pct=None, spline_basis=None,
                                 extra_colnames=None, covariates=None, nbr_idx=None,
                                 asym_treatment=False, nbr_half_seed=42):
    """
    Efficiently create grid neighborhoods for each point as 2D arrays (grids),
    storing them in a new column per feature. Also creates individual columns
    for each cell position in the neighborhood.
    
    Parameters
    ----------
    dftrain : pd.DataFrame
        Input DataFrame with index of format "{row}_{col}".
    radius : int
        Neighborhood radius (grid size = 2*radius + 1).
    fill_missing : str
        How to fill missing values:
        'mean'            -> fill NaN with column mean
        'median'          -> fill NaN with column median
        'zero'            -> fill NaN with 0
        'neighbors_mean'  -> fill NaN with mean of adjacent neighbors
        'neighbors_median'-> fill NaN with median of adjacent neighbors
    a : float, optional
        Treatment value to assign when treatment is not None
    change : str, optional
        Type of change to make when treatment is not None:
        'center' -> change center cell (0, 0) to value a
        'nbr' -> change all neighbor cells (not center) to value a
    treatment : str, optional
        Column name to apply treatment changes to
    is_binary_treatment : bool, optional
        Whether treatment is binary
    spaceenv : object, optional
        Space environment object with bsplines and binary_treatment_interactions attributes
    get_t_pct : function, optional
        Function to get treatment percentage
    spline_basis : list, optional
        List of spline basis functions
    extra_colnames : list, optional
        Names of extra columns for splines/interactions
    covariates : list, optional
        List of covariate column names
        
    Returns
    -------
    tuple
        (pd.DataFrame, str): DataFrame with new columns for each feature containing 
        2D grids (tmp file paths) and individual cell columns, plus temp directory path.
    """

    # Parse coordinates from index
    index_parts = dftrain.index.to_series().str.split('_', expand=True)
    rows = index_parts[0].astype(int).values
    cols = index_parts[1].astype(int).values

    # Determine grid bounds
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    n_rows = max_row - min_row + 1
    n_cols = max_col - min_col + 1

    feature_cols = dftrain.columns

    # Precompute global fill values for mean/median/zero
    fill_values = {}
    for col in feature_cols:
        if fill_missing == 'mean':
            val = dftrain[col].mean()
        elif fill_missing == 'median':
            val = dftrain[col].median()
        elif fill_missing == 'zero':
            val = 0
        else:
            val = np.nan  # for neighbor-based filling
        fill_values[col] = 0 if (pd.isna(val) and fill_missing not in ('neighbors_mean', 'neighbors_median')) else val

    temp_dir = tempfile.mkdtemp(prefix='grid_features_')
    result_df = dftrain.copy()

    # Canonical ordered list of neighbor (dr, dc) positions, excluding center
    nbr_positions = [(dr, dc) for dr in range(-radius, radius + 1)
                     for dc in range(-radius, radius + 1)
                     if not (dr == 0 and dc == 0)]

    nbr_half_rng = np.random.default_rng(nbr_half_seed) if change == "nbr_half" else None
    # Pre-compute per-point random half-mask over nbr_positions (half 1s, half 0s)
    if change == "nbr_half":
        n_nbr = len(nbr_positions)
        n_points = len(rows)
        nbr_half_masks = np.zeros((n_points, n_nbr), dtype=int)
        for pi in range(n_points):
            perm = nbr_half_rng.permutation(n_nbr)
            nbr_half_masks[pi, perm[: n_nbr // 2]] = 1
    else:
        nbr_half_masks = None

    # For each feature
    for col in feature_cols:
        # Initialize full grid
        if fill_missing in ('neighbors_mean', 'neighbors_median'):
            full_grid = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
            full_grid[rows - min_row, cols - min_col] = result_df[col].values  # Use result_df instead of dftrain
        else:
            full_grid = np.full((n_rows, n_cols), fill_values[col], dtype=np.float64)
            full_grid[rows - min_row, cols - min_col] = result_df[col].fillna(fill_values[col]).values  # Use result_df

        grids = []
        cell_data = {f"{col}_{dr}_{dc}": []
                     for dr in range(-radius, radius + 1)
                     for dc in range(-radius, radius + 1)}

        # Process each point
        for i, (r, c) in enumerate(zip(rows - min_row, cols - min_col)):
            r_start, r_end = r - radius, r + radius + 1
            c_start, c_end = c - radius, c + radius + 1

            pad_top = max(0, -r_start)
            pad_left = max(0, -c_start)
            pad_bottom = max(0, r_end - n_rows)
            pad_right = max(0, c_end - n_cols)

            r_start_clamped = max(0, r_start)
            r_end_clamped = min(n_rows, r_end)
            c_start_clamped = max(0, c_start)
            c_end_clamped = min(n_cols, c_end)

            # copy: the treatment changes below must not leak into full_grid,
            # i.e. into the neighborhoods of points processed later
            neighborhood = full_grid[r_start_clamped:r_end_clamped,
                                     c_start_clamped:c_end_clamped].copy()

            # Pad with NaN so edges don't get false fill values
            if pad_top or pad_left or pad_bottom or pad_right:
                neighborhood = np.pad(
                    neighborhood,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant',
                    constant_values=np.nan
                )

            # Handle missing values
            if fill_missing in ('neighbors_mean', 'neighbors_median'):
                mask = np.isnan(neighborhood)
                if mask.any():
                    for rr, cc in zip(*np.where(mask)):
                        r0, r1 = max(0, rr - 1), min(neighborhood.shape[0], rr + 2)
                        c0, c1 = max(0, cc - 1), min(neighborhood.shape[1], cc + 2)

                        neighbor_vals = neighborhood[r0:r1, c0:c1].flatten()
                        neighbor_vals = neighbor_vals[~np.isnan(neighbor_vals)]

                        if len(neighbor_vals) > 0:
                            if fill_missing == 'neighbors_mean':
                                neighborhood[rr, cc] = neighbor_vals.mean()
                            else:  # neighbors_median
                                neighborhood[rr, cc] = np.median(neighbor_vals)
                        else:
                            neighborhood[rr, cc] = 0  # fallback
            elif fill_missing == 'zero':
                neighborhood = np.nan_to_num(neighborhood, nan=0.0)
            # mean/median handled globally

            # Apply spatial treatment changes if specified (only for treatment column)
            if treatment is not None and col == treatment:
                center_r, center_c = radius, radius  # Center position in neighborhood
                
                if change == "center":
                    # Change center cell (0, 0) to value a
                    neighborhood[center_r, center_c] = a
                elif change == "nbr":
                    # Change all neighbor cells (not center) to value a
                    for nr in range(neighborhood.shape[0]):
                        for nc in range(neighborhood.shape[1]):
                            if nr != center_r or nc != center_c:  # Not the center
                                neighborhood[nr, nc] = a
                elif change == "nbr_k" and nbr_idx is not None:
                    # Change only the single neighbor at position nbr_idx (1-indexed) to value a
                    nbr_dr, nbr_dc = nbr_positions[nbr_idx - 1]
                    neighborhood[center_r + nbr_dr, center_c + nbr_dc] = a
                elif change == "nbr_half":
                    # Randomly set half of neighbors to 1, rest to 0
                    mask_i = nbr_half_masks[i]
                    for k_idx, (nbr_dr, nbr_dc) in enumerate(nbr_positions):
                        neighborhood[center_r + nbr_dr, center_c + nbr_dc] = mask_i[k_idx]

            # Apply spatial changes to extra columns (spline/interaction features)
            if extra_colnames is not None and col in extra_colnames:
                center_r, center_c = radius, radius  # Center position in neighborhood

                # Calculate the value for this extra column based on treatment value a
                # (skipped for nbr_half, which computes per-neighbor values below)
                if change != "nbr_half" and a is not None:
                    if not is_binary_treatment and spaceenv.bsplines:
                        t_a_pct = get_t_pct(a)
                        # Find which spline basis this column corresponds to
                        col_idx = extra_colnames.index(col)
                        extra_value = spline_basis[col_idx](t_a_pct)
                    elif is_binary_treatment and spaceenv.binary_treatment_iteractions:
                        # Find which covariate this extra column corresponds to
                        # Assuming extra_colnames follow same order as covariates
                        col_idx = extra_colnames.index(col)
                        covariate_val = result_df[covariates[col_idx]].iloc[i]
                        extra_value = covariate_val * a
                
                if change == "center":
                    # Change center cell to the calculated extra value
                    neighborhood[center_r, center_c] = extra_value
                elif change == "nbr":
                    # Change all neighbor cells (not center) to the calculated extra value
                    for nr in range(neighborhood.shape[0]):
                        for nc in range(neighborhood.shape[1]):
                            if nr != center_r or nc != center_c:  # Not the center
                                neighborhood[nr, nc] = extra_value
                elif change == "nbr_k" and nbr_idx is not None:
                    nbr_dr, nbr_dc = nbr_positions[nbr_idx - 1]
                    neighborhood[center_r + nbr_dr, center_c + nbr_dc] = extra_value
                elif change == "nbr_half":
                    mask_i = nbr_half_masks[i]
                    for k_idx, (nbr_dr, nbr_dc) in enumerate(nbr_positions):
                        m_a = mask_i[k_idx]
                        if not is_binary_treatment and spaceenv.bsplines:
                            t_m_pct = get_t_pct(m_a)
                            col_idx = extra_colnames.index(col)
                            ev = spline_basis[col_idx](t_m_pct)
                        elif is_binary_treatment and spaceenv.binary_treatment_iteractions:
                            col_idx = extra_colnames.index(col)
                            covariate_val = result_df[covariates[col_idx]].iloc[i]
                            ev = covariate_val * m_a
                        else:
                            ev = extra_value
                        neighborhood[center_r + nbr_dr, center_c + nbr_dc] = ev

            # ASYM mask: for the treatment column only, blank out cells with
            # dc < 0 so the image patch (and tabular cells) only carries the
            # eastward + same-column treatment values.
            if asym_treatment and treatment is not None and col == treatment:
                fill_val = fill_values.get(col, 0.0)
                if pd.isna(fill_val):
                    fill_val = 0.0
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        if dc < 0:
                            neighborhood[dr + radius, dc + radius] = fill_val

            # Store cell values (with treatment changes applied)
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    cell_value = neighborhood[dr + radius, dc + radius]
                    cell_data[f"{col}_{dr}_{dc}"].append(cell_value)

            # Save normalized grid image
            norm_grid = ((neighborhood - neighborhood.min()) /
                         (neighborhood.ptp() + 1e-8) * 255).astype(np.uint8)
            img = Image.fromarray(norm_grid, mode='L')
            file_path = os.path.join(temp_dir, f'grid_{col}_{result_df.index[i]}.png')
            img.save(file_path, format='PNG')
            grids.append(file_path)

        # Apply treatment changes to result_df individual cell columns if needed
        if treatment is not None and col == treatment:
            fill_val_treat = fill_values.get(col, 0.0)
            if pd.isna(fill_val_treat):
                fill_val_treat = 0.0
            for cell_col_name in cell_data.keys():
                # Parse the relative position from column name
                # Format: "{col}_{dr}_{dc}"
                parts = cell_col_name.split('_')
                dr, dc = int(parts[-2]), int(parts[-1])

                if change == "center" and dr == 0 and dc == 0:
                    # Change center cell values to a
                    cell_data[cell_col_name] = [a] * len(cell_data[cell_col_name])
                elif change == "nbr" and (dr != 0 or dc != 0):
                    # ASYM: only flip eastward (dc >= 0) neighbors when masking
                    if asym_treatment and dc < 0:
                        pass  # leave masked-to-fill from per-point loop
                    else:
                        cell_data[cell_col_name] = [a] * len(cell_data[cell_col_name])
                elif change == "nbr_k" and nbr_idx is not None:
                    nbr_dr, nbr_dc = nbr_positions[nbr_idx - 1]
                    if dr == nbr_dr and dc == nbr_dc:
                        cell_data[cell_col_name] = [a] * len(cell_data[cell_col_name])
                elif change == "nbr_half" and (dr != 0 or dc != 0):
                    k_idx = nbr_positions.index((dr, dc))
                    cell_data[cell_col_name] = [int(nbr_half_masks[pi, k_idx]) for pi in range(len(cell_data[cell_col_name]))]

                # ASYM: enforce mask on dc < 0 cells regardless of change
                if asym_treatment and dc < 0:
                    cell_data[cell_col_name] = [fill_val_treat] * len(cell_data[cell_col_name])
                    

        # Apply extra column changes to result_df individual cell columns if needed
        if extra_colnames is not None and col in extra_colnames:
            for cell_col_name in cell_data.keys():
                # Parse the relative position from column name
                # Format: "{col}_{dr}_{dc}"
                parts = cell_col_name.split('_')
                dr, dc = int(parts[-2]), int(parts[-1])

                # Calculate the appropriate value for this extra column
                # (skipped for nbr_half, which computes per-neighbor values below)
                extra_value = None
                covariate_vals = None
                col_idx = extra_colnames.index(col)
                if change != "nbr_half" and a is not None:
                    if not is_binary_treatment and spaceenv.bsplines:
                        t_a_pct = get_t_pct(a)
                        extra_value = spline_basis[col_idx](t_a_pct)

                    elif is_binary_treatment and spaceenv.binary_treatment_iteractions:
                        # Use per-point covariate values
                        covariate_vals = []
                        for point_i in range(len(result_df)):
                            covariate_val = result_df[covariates[col_idx]].iloc[point_i]
                            covariate_vals.append(covariate_val * a)

                if change == "center" and dr == 0 and dc == 0:
                    # Change center cell values to calculated extra value
                    if extra_value is None:
                        cell_data[cell_col_name] = covariate_vals
                    else:
                        cell_data[cell_col_name] = [extra_value] * len(cell_data[cell_col_name])
                elif change == "nbr" and (dr != 0 or dc != 0):
                    # Change neighbor cell values to calculated extra value
                    if extra_value is None:
                        cell_data[cell_col_name] = covariate_vals
                    else:
                        cell_data[cell_col_name] = [extra_value] * len(cell_data[cell_col_name])
                elif change == "nbr_k" and nbr_idx is not None:
                    nbr_dr, nbr_dc = nbr_positions[nbr_idx - 1]
                    if dr == nbr_dr and dc == nbr_dc:
                        if extra_value is None:
                            cell_data[cell_col_name] = covariate_vals
                        else:
                            cell_data[cell_col_name] = [extra_value] * len(cell_data[cell_col_name])
                elif change == "nbr_half" and (dr != 0 or dc != 0):
                    k_idx = nbr_positions.index((dr, dc))
                    per_point_vals = []
                    for pi in range(len(result_df)):
                        m_a = int(nbr_half_masks[pi, k_idx])
                        if not is_binary_treatment and spaceenv.bsplines:
                            t_m_pct = get_t_pct(m_a)
                            per_point_vals.append(spline_basis[col_idx](t_m_pct))
                        elif is_binary_treatment and spaceenv.binary_treatment_iteractions:
                            cov_val = result_df[covariates[col_idx]].iloc[pi]
                            per_point_vals.append(cov_val * m_a)
                        else:
                            per_point_vals.append(m_a)
                    cell_data[cell_col_name] = per_point_vals

        # Add results
        result_df[f"{col}_grid"] = grids
        for cell_col_name, cell_values in cell_data.items():
            result_df[cell_col_name] = cell_values

    return result_df, temp_dir


