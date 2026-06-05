import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from spacebench.algorithms import SpaceAlgo
from spacebench.log import LOGGER
from sci import SpaceDataset
from .spatialplus import SpatialPlus, Spatial
from .spatialplus_median import SpatialPlusMedian
from .utils import get_k_hop_neighbors
from copy import deepcopy
import networkx as nx


total_batch_size = 64


def pad_center(tensor, target_size):
    """Pad 4D tensor (B,H,W,C) to target_size x target_size, centered."""
    b, h, w, c = tensor.shape
    pad_h = target_size - h
    pad_w = target_size - w
    assert pad_h >= 0 and pad_w >= 0
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(tensor, (0, 0, pad_left, pad_right, pad_top, pad_bottom))


class NeighborTreatmentDataset(Dataset):
    """
    Builds patch tensors of treatments and covariates.
    Only used to extract flat_wo_center (neighbor treatments).
    """
    def __init__(
        self,
        dataset,
        nodes,
        coords2id,
        radius,
        treat_scaler=None,
        feat_scaler=None,
        output_scaler=None,
        a=None,
        change=None,
        datatype="grid",
        dataset_radius=None,
        nbr_off=None,
    ):
        if datatype != "grid":
            raise ValueError(f"Unsupported dataset type: {datatype}")
        if dataset_radius is None:
            dataset_radius = radius

        treat = dataset.full_treatment.reshape(-1, 1)
        cov = dataset.full_covariates
        out = dataset.full_outcome.reshape(-1, 1)

        nonbinary_treat_cols = [i for i in range(treat.shape[1]) if not np.all(np.isin(treat[:, i], [0, 1]))]
        nonbinary_cov_cols   = [i for i in range(cov.shape[1])   if not np.all(np.isin(cov[:, i],   [0, 1]))]
        nonbinary_out_cols   = [i for i in range(out.shape[1])   if not np.all(np.isin(out[:, i],   [0, 1]))]

        if treat_scaler is None and nonbinary_treat_cols:
            treat_scaler = StandardScaler().fit(treat[:, nonbinary_treat_cols])
        if feat_scaler is None and nonbinary_cov_cols:
            feat_scaler = StandardScaler().fit(cov[:, nonbinary_cov_cols])
        if output_scaler is None and nonbinary_out_cols:
            output_scaler = StandardScaler().fit(out[:, nonbinary_out_cols])

        self.treat_scaler  = treat_scaler
        self.feat_scaler   = feat_scaler
        self.output_scaler = output_scaler

        treat_scaled = treat.copy()
        if nonbinary_treat_cols and treat_scaler:
            treat_scaled[:, nonbinary_treat_cols] = treat_scaler.transform(treat[:, nonbinary_treat_cols])

        cov_scaled = cov.copy()
        if nonbinary_cov_cols and feat_scaler:
            cov_scaled[:, nonbinary_cov_cols] = feat_scaler.transform(cov[:, nonbinary_cov_cols])

        out_scaled = out.copy()
        if nonbinary_out_cols and output_scaler:
            out_scaled[:, nonbinary_out_cols] = output_scaler.transform(out[:, nonbinary_out_cols])

        true_treatment = torch.from_numpy(treat_scaled).float()
        outcomes       = torch.from_numpy(out_scaled).float()
        id2coords      = {v: k for k, v in coords2id.items()}

        # Build treatment patches
        self.treatments = torch.zeros(len(nodes), 2 * radius + 1, 2 * radius + 1, 1)
        for ii, n in enumerate(nodes):
            center_coords = id2coords[n]
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    cur_coords = (center_coords[0] + i, center_coords[1] + j)
                    if change == "center" and (i, j) == (0, 0):
                        self.treatments[ii, i + radius, j + radius, 0] = a
                    elif change == "nbr" and (i, j) != (0, 0):
                        self.treatments[ii, i + radius, j + radius, 0] = a
                    elif change == "nbr_one" and nbr_off is not None and (i, j) == tuple(nbr_off):
                        self.treatments[ii, i + radius, j + radius, 0] = a
                    else:
                        self.treatments[ii, i + radius, j + radius, 0] = true_treatment[coords2id[cur_coords]]

        treat_size  = 2 * radius + 1
        cov_size    = 2 * dataset_radius + 1
        target_size = max(treat_size, cov_size)
        self.treatments = pad_center(self.treatments, target_size)

        self.true_treatment = true_treatment[nodes].squeeze()
        self.outcomes       = outcomes[nodes]

    def __len__(self):
        return len(self.outcomes)

    def __getitem__(self, idx):
        return self.treatments[idx], self.outcomes[idx]


class EMAP(SpaceAlgo):
    """
    Exposure Mapping Model.

    Appends mean of neighbor treatments to the covariates
    and fits a single outcome model.
    """
    supports_binary     = True
    supports_continuous = True

    def __init__(
        self,
        radius: float,
        k: int = 100,
        model: str = "spatialplus",
        max_iter: int = 20_000,
        lam: float = 0.001,
        lam_t: float = 0.001,
        lam_y: float = 0.001,
        spatial_split_kwargs: dict | None = None,
        batch_size=None,
        device: str = None,
    ):
        self.cvae_radius = radius
        self.modelname = model
        
        self.spatial_kwargs = {
            "k": k,
            "max_iter": max_iter,
            "lam": lam,
            "spatial_split_kwargs": spatial_split_kwargs,
        }
        
        self.spatialplus_kwargs = {
            "k": k,
            "max_iter": max_iter,
            "lam_t": lam_t,
            "lam_y": lam_y,
            "spatial_split_kwargs": spatial_split_kwargs,
        }

        if batch_size is not None:
            global total_batch_size
            total_batch_size = batch_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        LOGGER.debug(f"Exposure Mapping using device: {self.device}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_max_nodes(self, dataset):
        """Return nodes that have the maximum k-hop neighbor count."""
        graph      = nx.from_edgelist(dataset.full_edge_list)
        node_list  = list(graph.nodes())
        nbrs       = {n: get_k_hop_neighbors(graph, n, max(dataset.radius, self.cvae_radius)) for n in node_list}
        nbr_counts = {n: len(v) for n, v in nbrs.items()}
        max_count  = max(nbr_counts.values())
        return [n for n, cnt in nbr_counts.items() if cnt == max_count]

    def _flat_wo_center(self, data: NeighborTreatmentDataset) -> np.ndarray:
        """
        Compute the mean neighbor treatment from the treatment patch, excluding
        the center pixel. Returns shape (N, 1).
        """
        if self.cvae_radius == 0:
            return None

        B, H, W, C = data.treatments.shape
        flat        = data.treatments.view(B, -1, C)          # [B, H*W, 1]
        center_idx  = (H * W) // 2
        flat_wo     = torch.cat(
            [flat[:, :center_idx], flat[:, center_idx + 1:]], dim=1
        )                                                      # [B, H*W-1, 1]
        return flat_wo.mean(dim=1).cpu().numpy()               # [B, H*W-1]

    def _build_augmented_dataset(self, dataset, nodes, a=None, change=None, nbr_off=None):
        """
        Build NeighborTreatmentDataset for *nodes*, extract flat_wo_center,
        and return (new_dataset, flat_wo_center_array).
        """
        data = NeighborTreatmentDataset(
            dataset,
            nodes,
            self.max_coords2id,
            self.cvae_radius,
            treat_scaler=self.train_data.treat_scaler,
            feat_scaler=self.train_data.feat_scaler,
            output_scaler=self.train_data.output_scaler,
            a=a,
            change=change,
            datatype=dataset.datatype,
            dataset_radius=getattr(dataset, "conf_radius", self.cvae_radius),
            nbr_off=nbr_off,
        )

        flat_wo = self._flat_wo_center(data)

        new_dataset = deepcopy(dataset)
        if change == "center" and a is not None:
            new_dataset.treatment = np.full_like(new_dataset.treatment, a)
        if flat_wo is not None:
            new_dataset.covariates = np.concatenate(
                [new_dataset.covariates, flat_wo], axis=1
            )

        return new_dataset, data

    # ------------------------------------------------------------------
    # SpaceAlgo interface
    # ------------------------------------------------------------------

    def fit(self, dataset: SpaceDataset):
        LOGGER.debug("Building max-neighbor node set...")
        self.max_nodes     = self._build_max_nodes(dataset)
        self.max_coords2id = {
            tuple(coord): i
            for i, coord in enumerate(dataset.full_coordinates)
        }

        conf_radius = getattr(dataset, "conf_radius", self.cvae_radius)

        # Build training dataset (no scaler yet — let it fit)
        LOGGER.debug("Building treatment-patch dataset for training nodes...")
        self.train_data = NeighborTreatmentDataset(
            dataset,
            self.max_nodes,
            self.max_coords2id,
            self.cvae_radius,
            datatype=dataset.datatype,
            dataset_radius=conf_radius,
        )

        # Augment covariates with neighbor treatments
        flat_wo = self._flat_wo_center(self.train_data)
        new_dataset = deepcopy(dataset)
        if flat_wo is not None:
            new_dataset.covariates = np.concatenate(
                [new_dataset.covariates, flat_wo], axis=1
            )

        LOGGER.debug("Fitting model...")
        if self.modelname == "spatialplus":
            self.head_model = SpatialPlus(**self.spatialplus_kwargs)
        elif self.modelname == "spatial":
            self.head_model = Spatial(**self.spatial_kwargs)
        elif self.modelname == "spatialplus_median":
            self.head_model = SpatialPlusMedian(**self.spatialplus_kwargs)
            self.head_model.N_SEEDS = 4
        self.head_model.fit(new_dataset)
        LOGGER.debug("fitting complete.")

    def eval(self, dataset: SpaceDataset) -> dict:
        LOGGER.debug("Computing counterfactuals...")
        ite = []
        for a in dataset.treatment_values:
            preds_a = self.predict(dataset, self.max_nodes, a=a, change="center")
            ite.append(preds_a)
        ite = np.concatenate(ite, axis=1)

        effects = {"erf": ite.mean(0), "ite": ite}

        if dataset.has_binary_treatment():
            effects["ate"] = self.head_model.t_coef

            from sci.env import PER_NEIGHBOR_SPILLOVER
            if PER_NEIGHBOR_SPILLOVER:
                # Flip ONE neighbor — the mean-neighbor covariate shifts
                # by 1/N, giving the per-neighbor spillover effect.
                nbr_off = (0, 1)
                spill = []
                for a in dataset.treatment_values:
                    preds_a = self.predict(
                        dataset, self.max_nodes, a=a,
                        change="nbr_one", nbr_off=nbr_off,
                    )
                    spill.append(preds_a)
                spill = np.concatenate(spill, axis=1)
                s = spill.mean(0)
                effects["spill"] = s[1] - s[0]
            else:
                spill = []
                for a in dataset.treatment_values:
                    preds_a = self.predict(dataset, self.max_nodes, a=a, change="nbr")
                    spill.append(preds_a)
                spill = np.concatenate(spill, axis=1)
                s = spill.mean(0)
                effects["spill"] = s[1] - s[0]

        return effects

    def tune_metric(self, dataset: SpaceDataset) -> float:
        new_dataset, _ = self._build_augmented_dataset(dataset, self.max_nodes)
        m = self.head_model.tune_metric(new_dataset)
        return m if not np.isnan(m) else 10_000.0

    def predict(self, dataset: SpaceDataset, nodes, a, change, nbr_off=None) -> np.ndarray:
        new_dataset, _ = self._build_augmented_dataset(dataset, nodes, a=a, change=change, nbr_off=nbr_off)
        return self.head_model.predict(new_dataset)