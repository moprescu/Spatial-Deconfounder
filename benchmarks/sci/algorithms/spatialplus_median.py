import numpy as np
import torch
import torch.nn.functional as F

from spacebench.algorithms import SpaceAlgo
from spacebench.log import LOGGER
from sci.env import SpaceDataset
from sci.algorithms.spatialplus import (
    SpatialPlus,
    tps_pred,
    tps_loss,
    safe_standardize_stats,
    SAFE_STANDARDIZE,
)
from sci.algorithms.datautils import spatial_train_test_split
import networkx as nx


def _tps_loss_l2(y, coords, cp_coords, covars, params, lam,
                 binary_loss=False, mask=None, lam_covar=0.0):
    k = cp_coords.shape[0]
    d = coords.shape[1]
    p = covars.shape[1]
    pred = tps_pred(coords, cp_coords, covars, params)
    if not binary_loss:
        pred_loss = F.mse_loss(pred, y, reduction="none")
    else:
        pred_loss = F.binary_cross_entropy_with_logits(pred, y, reduction="none")
    pred_loss = (pred_loss * mask).mean() if mask is not None else pred_loss.mean()
    reg_loss = params[-k:, None].pow(2).sum()
    total = pred_loss + lam * reg_loss
    if lam_covar > 0.0 and p > 0:
        b = params[1 + d : 1 + d + p]
        total = total + lam_covar * b.pow(2).sum()
    return total


def _tps_opt_l2(y, coords, cp_coords, covars, lam, lr=0.003, max_iter=20_000,
                binary_loss=False, mask=None, lam_covar=0.0):
    d = coords.shape[1]
    p = covars.shape[1]
    k = cp_coords.shape[0]
    params = torch.zeros(1 + d + p + k, requires_grad=True, device=y.device)
    opt = torch.optim.Adam([params], lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.9, patience=10, verbose=False, min_lr=1e-8,
    )
    for _ in range(max_iter):
        opt.zero_grad()
        loss = _tps_loss_l2(y, coords, cp_coords, covars, params, lam,
                            binary_loss, mask, lam_covar)
        loss.backward()
        opt.step()
        sched.step(loss)
    return params.detach()


class _SingleBestSpatialPlus(SpatialPlus):
    """SpatialPlus variant with ZTOnly head split and covariate L2 ridge.

    Drops the trailing ``n_z_covars`` columns from the y-model so the CVAE
    latent block cannot absorb T-correlated outcome variance. Also applies
    ``lam_t_covar`` ridge on the t-model's linear-covariate coefficients
    (auto-set to 10 when ``dataset.radius == 2``).
    """

    def __init__(
        self,
        k: int = 100,
        max_iter: int = 20_000,
        lam_t: float = 0.001,
        lam_y: float = 0.001,
        spatial_split_kwargs: dict | None = None,
    ):
        super().__init__(
            k=k, max_iter=max_iter, lam_t=lam_t, lam_y=lam_y,
            spatial_split_kwargs=spatial_split_kwargs,
        )
        self.n_z_covars = 0
        self.lam_t_covar = 0.0
        self.lam_y_covar = 0.0

    def _split(self, covars):
        if self.n_z_covars == 0:
            return covars, covars
        p = covars.shape[1]
        if self.n_z_covars > p:
            raise ValueError(
                f"n_z_covars={self.n_z_covars} exceeds covariate count {p}"
            )
        return covars, covars[:, : p - self.n_z_covars]

    def fit(self, dataset):
        if dataset.radius == 2:
            self.lam_t_covar = 0.0

        if self.spatial_split_kwargs is not None:
            train_ix, _, _ = spatial_train_test_split(
                nx.from_edgelist(dataset.edges),
                **self.spatial_split_kwargs,
                buffer=1 + dataset.radius,
            )
            self.mask = torch.zeros(dataset.size()).to(self.device)
            self.mask[train_ix] = torch.tensor(1.0)
        else:
            self.mask = None

        coords = torch.FloatTensor(dataset.coordinates).to(self.device)
        covars = torch.FloatTensor(dataset.covariates).to(self.device)
        t = torch.FloatTensor(dataset.treatment).to(self.device)
        self.t_mu, self.t_std = t.mean(), t.std()
        if not dataset.has_binary_treatment():
            t = (t - self.t_mu) / self.t_std

        k = min(self.k, dataset.size())
        self.cp_idx = torch.LongTensor(
            sorted(np.random.choice(dataset.size(), k, replace=False))
        ).to(self.device)
        self.cp_coords = coords[self.cp_idx]

        t_covars, y_covars = self._split(covars)

        if SAFE_STANDARDIZE:
            self.coords_mu, self.coords_std = safe_standardize_stats(coords)
            self.t_covars_mu, self.t_covars_std = safe_standardize_stats(t_covars)
            self.covars_mu, self.covars_std = safe_standardize_stats(y_covars)
        else:
            self.coords_mu, self.coords_std = coords.mean(0), coords.std(0)
            self.t_covars_mu, self.t_covars_std = t_covars.mean(0), t_covars.std(0)
            self.covars_mu, self.covars_std = y_covars.mean(0), y_covars.std(0)

        coords_s = (coords - self.coords_mu) / self.coords_std
        t_covars_s = (t_covars - self.t_covars_mu) / self.t_covars_std

        self.t_params = _tps_opt_l2(
            t, coords_s, self.cp_coords, t_covars_s,
            lam=self.lam_t, max_iter=self.max_iter,
            binary_loss=dataset.has_binary_treatment(),
            mask=self.mask, lam_covar=self.lam_t_covar,
        )

        t_pred = tps_pred(coords_s, self.cp_coords, t_covars_s, self.t_params)
        if dataset.has_binary_treatment():
            t_pred = torch.sigmoid(t_pred)
        with torch.no_grad():
            t_resid = t - t_pred

        y = torch.FloatTensor(dataset.outcome).to(self.device)
        self.y_mu, self.y_std = y.mean(), y.std()
        if SAFE_STANDARDIZE and self.y_std < 1e-8:
            self.y_std = torch.ones_like(self.y_std)
        y = (y - self.y_mu) / self.y_std
        _, y_covars_raw = self._split(
            torch.FloatTensor(dataset.covariates).to(self.device)
        )
        inputs = torch.cat([t_resid[:, None], y_covars_raw], dim=1)
        if SAFE_STANDARDIZE:
            self.inputs_mu, self.inputs_std = safe_standardize_stats(inputs)
        else:
            self.inputs_mu, self.inputs_std = inputs.mean(0), inputs.std(0)
        inputs = (inputs - self.inputs_mu) / self.inputs_std

        self.y_params = _tps_opt_l2(
            y, coords_s, self.cp_coords, inputs,
            lam=self.lam_y, max_iter=self.max_iter,
            mask=self.mask, lam_covar=self.lam_y_covar,
        )

        self.t_coef = (self.y_params[3] * self.y_std / self.inputs_std[0]).cpu().item()

    def tune_metric(self, dataset):
        if self.mask is None:
            LOGGER.warning("No mask specified for the tune metric. Using full dataset.")
            tune_mask = None
        else:
            tune_mask = 1.0 - self.mask

        coords = torch.FloatTensor(dataset.coordinates).to(self.device)
        covars = torch.FloatTensor(dataset.covariates).to(self.device)
        y = torch.FloatTensor(dataset.outcome).to(self.device)
        t = torch.FloatTensor(dataset.treatment).to(self.device)

        t_covars, y_covars = self._split(covars)
        t_covars = (t_covars - self.t_covars_mu) / self.t_covars_std
        coords = (coords - self.coords_mu) / self.coords_std
        if not dataset.has_binary_treatment():
            t = (t - self.t_mu) / self.t_std

        t_loss = tps_loss(
            t, coords, self.cp_coords, t_covars, self.t_params,
            lam=0.0, binary_loss=dataset.has_binary_treatment(), mask=tune_mask,
        )
        t_pred = tps_pred(coords, self.cp_coords, t_covars, self.t_params)
        if dataset.has_binary_treatment():
            t_pred = torch.sigmoid(t_pred)
        t_resid = t - t_pred

        inputs = torch.cat([t_resid[:, None], y_covars], dim=1)
        inputs = (inputs - self.inputs_mu) / self.inputs_std
        y = (y - self.y_mu) / self.y_std
        y_loss = tps_loss(
            y, coords, self.cp_coords, inputs, self.y_params,
            lam=0.0, mask=tune_mask,
        )
        return t_loss.cpu().item() + y_loss.cpu().item()

    def predict(self, dataset):
        coords = torch.FloatTensor(dataset.coordinates).to(self.device)
        covars = torch.FloatTensor(dataset.covariates).to(self.device)
        t = torch.FloatTensor(dataset.treatment).to(self.device)

        t_covars, y_covars = self._split(covars)
        t_covars = (t_covars - self.t_covars_mu) / self.t_covars_std
        coords = (coords - self.coords_mu) / self.coords_std
        if not dataset.has_binary_treatment():
            t = (t - self.t_mu) / self.t_std

        with torch.no_grad():
            t_pred = tps_pred(coords, self.cp_coords, t_covars, self.t_params)
            if dataset.has_binary_treatment():
                t_pred = torch.sigmoid(t_pred)
            t_resid = t - t_pred

            inputs = torch.cat([t_resid[:, None], y_covars], dim=1)
            inputs = (inputs - self.inputs_mu) / self.inputs_std
            y_pred = tps_pred(coords, self.cp_coords, inputs, self.y_params)
            y_pred = y_pred * self.y_std + self.y_mu
        return y_pred.cpu().numpy().reshape(-1, 1)


class SpatialPlusMedian(SpaceAlgo):
    """N-seed median ensemble of _SingleBestSpatialPlus.

    Trains N_SEEDS sub-heads with seeds 0..N_SEEDS-1, then takes the median
    t_coef across all heads. Both modes use _SingleBestSpatialPlus; ``use_z``
    controls whether the CVAE latent block is present at all:

    Args:
        use_z: If True, sub-heads receive Z covariates and apply the ZTOnly
               split (Z goes to t-model only, dropped from y-model).
               If False, ``n_z_covars`` is forced to 0 -- no ZTOnly split
               and no Z columns in either model.

    predict() returns predictions consistent with the median t_coef:
      - Odd N_SEEDS: predictions of the head whose t_coef is the median.
      - Even N_SEEDS: average of the two median-bracketing heads, so
        mean(pred(T=1) - pred(T=0)) == np.median(t_coefs).
    """

    N_SEEDS = 3

    supports_binary = True
    supports_continuous = True

    def __init__(
        self,
        k: int = 100,
        max_iter: int = 20_000,
        lam_t: float = 0.001,
        lam_y: float = 0.001,
        spatial_split_kwargs: dict | None = None,
        use_z: bool = False,
        n_z_covars: int = 0,
        lam_t_covar: float = 0.0,
        lam_y_covar: float = 0.0,
    ):
        super().__init__()
        self.k = k
        self.max_iter = max_iter
        self.lam_t = lam_t
        self.lam_y = lam_y
        self.spatial_split_kwargs = spatial_split_kwargs
        self.use_z = use_z
        self.n_z_covars = n_z_covars
        self.lam_t_covar = lam_t_covar
        self.lam_y_covar = lam_y_covar
        self._heads: list = []
        self._t_coefs = None
        self._predict_idx: list[int] = []

    def _make_head(self):
        sub = _SingleBestSpatialPlus(
            k=self.k,
            max_iter=self.max_iter,
            lam_t=self.lam_t,
            lam_y=self.lam_y,
            spatial_split_kwargs=self.spatial_split_kwargs,
        )
        sub.n_z_covars = self.n_z_covars if self.use_z else 0
        sub.lam_t_covar = self.lam_t_covar
        sub.lam_y_covar = self.lam_y_covar
        return sub

    def fit(self, dataset: SpaceDataset):
        head_cls = "with_z" if self.use_z else "no_z"
        self._heads = []
        for seed in range(self.N_SEEDS):
            LOGGER.info(f"SpatialPlusMedian ({head_cls}) seed {seed}/{self.N_SEEDS - 1}")
            # np.random.seed(seed)
            # torch.manual_seed(seed)
            sub = self._make_head()
            sub.fit(dataset)
            self._heads.append(sub)

        self._t_coefs = np.array([float(h.t_coef) for h in self._heads])
        self.t_coef = float(np.median(self._t_coefs))
        LOGGER.info(f"t_coefs across seeds: {self._t_coefs}, median: {self.t_coef}")

        sorted_idx = np.argsort(self._t_coefs)
        n = self.N_SEEDS
        if n % 2 == 1:
            self._predict_idx = [int(sorted_idx[n // 2])]
        else:
            self._predict_idx = [int(sorted_idx[n // 2 - 1]),
                                 int(sorted_idx[n // 2])]

        proxy = self._heads[self._predict_idx[-1]]
        for attr in (
            "mask", "cp_idx", "cp_coords",
            "coords_mu", "coords_std",
            "t_covars_mu", "t_covars_std",
            "covars_mu", "covars_std",
            "inputs_mu", "inputs_std",
            "y_mu", "y_std", "t_mu", "t_std",
            "t_params", "y_params",
        ):
            if hasattr(proxy, attr):
                setattr(self, attr, getattr(proxy, attr))

    def predict(self, dataset: SpaceDataset):
        preds = [self._heads[i].predict(dataset) for i in self._predict_idx]
        if len(preds) == 1:
            return preds[0]
        return sum(preds) / len(preds)

    def tune_metric(self, dataset: SpaceDataset) -> float:
        return float(np.median([h.tune_metric(dataset) for h in self._heads]))

    def eval(self, dataset: SpaceDataset):
        return self._heads[self._predict_idx[-1]].eval(dataset)

    def available_estimands(self):
        return self._heads[self._predict_idx[-1]].available_estimands()
