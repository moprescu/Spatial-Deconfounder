import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor,
    # RichProgressBar
)
from pytorch_lightning.loggers import WandbLogger

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from .utils import UNet, get_k_hop_neighbors, DoubleConvMultiChannel
from .unet import TemporalUNetHead, TemporalUNetDataset
import networkx as nx

from sci import SpaceDataset
from spacebench.algorithms import SpaceAlgo
from spacebench.log import LOGGER
import os
from .spatialplus import SpatialPlus
from .spatialplus_median import SpatialPlusMedian
from sci.algorithms.datautils import spatial_train_test_split
# from .pysal_spreg import GMLag
from copy import deepcopy

from sklearn.decomposition import PCA
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt

total_batch_size = 64
latent_idx = "mu" # "z_s" for sample and "mu" for expected value for latents


class CVAE_Grid(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        radius: float,
        encoder_conv1: int = 32,
        encoder_conv2: int = 64,
        kernel_size: int = 3,
        encoder_pool: str = "avg",
        connectivity: int = 4,
        tau: float = 10.0,
        eps: float = 1e-5,
        binary_treatment: bool = True,
    ):
        """
        CVAE for interference-aware deconfounding on gridded datasets

        Args:
            latent_dim: Dimension of latent space Z_s
            radius: Radius for patch-based convolution
            tau: Precision parameter for GMRF prior
            eps: Small constant for numerical stability
            binary_treatment: If True, use sigmoid + BCE (binary). If False, use linear + Gaussian NLL (continuous).
        """
        super(CVAE_Grid, self).__init__()

        self.feature_dim = feature_dim
        self.radius = radius
        self.tau = tau
        self.eps = eps
        self.binary_treatment = binary_treatment
        self.patch_size = 2 * radius + 1
        self.kernel_size = kernel_size
        # self.per_location_latent_dim = latent_dim
        self.per_location_latent_dim = encoder_conv2 // 2
        self.latent_dim = self.patch_size * self.patch_size * self.per_location_latent_dim
        self.connectivity = connectivity
        
        # Calculate input channels: A_s + A_{N_s} (treatments + neighbor treatments)
        # Patch of treatments size self.patch_size x self.patch_size
        self.encoder_input_dim = 1 + self.feature_dim
        
        # Encoder: Two 3x3 convolutions over radius-r patch
        self.enc_mid_chan = encoder_conv1
        self.enc_out_chan = encoder_conv2
        self.encoder_conv = DoubleConvMultiChannel(self.encoder_input_dim, self.enc_mid_chan, self.enc_out_chan, radius=self.radius, k_size=self.kernel_size)
        self.encoder_pool = encoder_pool
                
        # Encoder output processing including covariates
        self.encoder_flatten_dim = self.enc_out_chan
        # self.fc_mu = nn.Linear(self.encoder_flatten_dim, self.latent_dim)
        # self.fc_logvar = nn.Linear(self.encoder_flatten_dim, self.latent_dim)
        
        # Decoder: One-layer MLP
        # Binary:     p_ψ(A_s=1|x_s,Z_s) = σ(f_ψ(x_s,Z_s))  → 1 output
        # Continuous:  p_ψ(A_s|x_s,Z_s) = N(μ_ψ, σ²_ψ)      → 2 outputs (mean, log_var)
        decoder_input_dim = self.patch_size * self.patch_size * self.feature_dim + self.latent_dim
        if self.binary_treatment:
            self.decoder = nn.Linear(decoder_input_dim, 1)
        else:
            self.decoder_mu = nn.Linear(decoder_input_dim, 1)
            self.decoder_logvar = nn.Linear(decoder_input_dim, 1)
        
        # Precompute grid Laplacian matrix L for GMRF prior
        self.register_buffer('laplacian', self._build_patch_laplacian())
        
        # init_temperature = 1.0
        # self.log_temperature = nn.Parameter(torch.tensor(init_temperature).log())
    
    
    def _build_patch_laplacian(self):
        """
        Build patch Laplacian matrix L for 2D grid
        """
        n = self.patch_size ** 2
        L = torch.zeros(n, n)
        
        def coord_to_idx(i, j):
            return i * self.patch_size + j
        
        for i in range(self.patch_size):
            for j in range(self.patch_size):
                idx = coord_to_idx(i, j)
                degree = 0
                
                # Add edges to neighbors
                if self.connectivity == 4:
                    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                else:
                    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
                for di, dj in nbrs:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.patch_size and 0 <= nj < self.patch_size:
                        neighbor_idx = coord_to_idx(ni, nj)
                        L[idx, neighbor_idx] = -1
                        degree += 1
                
                L[idx, idx] = degree
        
        return L    
    
    def encode(self, A, x):
        """
        Encoder: q_φ(Z_s|A_s, A_{N_s}, x_s)
        Two 3×3 convolutions over radius-r patch
        
        Args:
            A: Current and Neighbor treatments [batch_size, patch_size, patch_size, 1]
            x: Current and Neighbor features [batch_size, patch_size, patch_size, feature_dim]
        
        Returns:
            mu, logvar: Parameters of latent distribution
        """
        # Rearrange for convolution: [batch, channels, height, width]
        encoder_input = torch.cat([A, x], dim=3).permute(0, 3, 1, 2)
                
        # Two 3×3 convolutions
        h = self.encoder_conv(encoder_input)
        
        # if self.encoder_pool == "max":
        #     h = F.adaptive_max_pool2d(h, (1, 1)).squeeze(-1).squeeze(-1)
        # elif self.encoder_pool == "avg":
        #     # Global average pooling to get fixed-size representation
        #     h = F.adaptive_avg_pool2d(h, (1, 1)).squeeze(-1).squeeze(-1)
                
        # # Output latent parameters
        # mu = self.fc_mu(h)
        # logvar = self.fc_logvar(h)
        mu = h[:, 0:self.enc_out_chan//2].flatten(start_dim=1)
        logvar =h[:, self.enc_out_chan//2:].flatten(start_dim=1)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        # temperature = torch.exp(self.log_temperature)
        return mu + eps * std
    
    def decode(self, x_s, z_s):
        """
        Decoder: predicts treatment given features and latent variables.
        Binary:     p_ψ(A_s=1|x_s, Z_s) = σ(f_ψ(x_s, Z_s))
        Continuous: p_ψ(A_s|x_s, Z_s) = N(μ_ψ(x_s,Z_s), σ²_ψ(x_s,Z_s))

        Args:
            x_s: Features at location [batch_size, patch_size *  patch_size *  feature_dim]
            z_s: Latent variables [batch_size, patch_size *  patch_size *  latent_dim]

        Returns:
            Binary: Treatment probability at location s
            Continuous: (mean, log_var) tuple
        """
        combined = torch.cat([x_s, z_s], dim=-1)

        if self.binary_treatment:
            logits = self.decoder(combined)
            probs = torch.sigmoid(logits)
            return probs.squeeze(-1)
        else:
            mu = self.decoder_mu(combined).squeeze(-1)
            logvar = self.decoder_logvar(combined).squeeze(-1)
            return mu, logvar
    
    def forward(self, treatments, covariates):
        """
        Forward pass through interference-aware CVAE
        
        Args:
            treatments: Current and neighboring treatments [batch_size, patch_size, patch_size, 1]
            covariates: Features at location [batch_size, patch_size, patch_size, feature_dim]
        
        Returns:
            treatment_probs: Treatment probabilities
            z_s: Latent distribution
            out: Outcome prediction
        """
        # Encode
        mu, logvar = self.encode(treatments, covariates)
        
        # Reparameterize
        z_s = self.reparameterize(mu, logvar)
        
        # Decode
        decode_out = self.decode(covariates.reshape(covariates.size(0), -1), z_s)

        if self.binary_treatment:
            return decode_out, mu, logvar
        else:
            dec_mu, dec_logvar = decode_out
            return (dec_mu, dec_logvar), mu, logvar
    
    def get_latent(self, treatments, covariates):
        """
        Return latent values interference-aware CVAE
        
        Args:
            treatments: Current and neighboring treatments [batch_size, patch_size, patch_size, 1]
            covariates: Features at location [batch_size, patch_size, patch_size, feature_dim]
        
        Returns:
            treatment_probs: Treatment probabilities
            z_s: Latent distribution
            out: Outcome prediction
        """
        # Encode
        mu, logvar = self.encode(treatments, covariates)
        
        # Reparameterize
        z_s = self.reparameterize(mu, logvar)
        # print(str(mu))
        
        return mu.view(-1, self.patch_size, self.patch_size, self.per_location_latent_dim), logvar.view(-1, self.patch_size, self.patch_size, self.per_location_latent_dim), z_s.view(-1, self.patch_size, self.patch_size, self.per_location_latent_dim)
    

class CVAE(pl.LightningModule):
    def __init__(
        self,
        feature_dim: int,
        radius: float,
        encoder_conv1: int = 32,
        encoder_conv2: int = 64,
        kernel_size: int = 3,
        encoder_pool: str = "avg",
        connectivity: int = 4,
        tau: float = 10.0,
        eps: float = 1e-5,
        weight_decay: float = 1e-5,
        lr: float = 1e-3,
        beta_max: float = 1e-1,
        beta_epoch_max: int = 10,
        epochs: int = 10,
        datatype: str = "grid",
        binary_treatment: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.radius = radius
        self.encoder_conv1 = encoder_conv1
        self.encoder_conv2 = encoder_conv2
        self.kernel_size = kernel_size
        self.encoder_pool = encoder_pool
        self.connectivity = connectivity
        self.tau = tau
        self.eps = eps
        self.datatype = datatype
        self.binary_treatment = binary_treatment

        if self.datatype == "grid":
            self.model = CVAE_Grid(
                feature_dim=self.feature_dim,
                radius=self.radius,
                encoder_conv1=self.encoder_conv1,
                encoder_conv2=self.encoder_conv2,
                kernel_size=self.kernel_size,
                encoder_pool=self.encoder_pool,
                connectivity=self.connectivity,
                tau=self.tau,
                eps=self.eps,
                binary_treatment=self.binary_treatment,
            )
        else:
            raise ValueError(f"Unsupported dataset type: {datatype}")
        
        # self.model = torch.compile(self.model, mode="reduce-overhead")
        
        self.weight_decay = weight_decay
        self.lr = lr
        self.beta_max = beta_max
        self.beta_epoch_max = beta_epoch_max
        self.epochs = epochs
        self.num_samples = 100
        self.n_pixels = self.model.patch_size * self.model.patch_size
        self.current_val_p_value = None
        
    def predict_step(self, batch, batch_idx):
        treatments, covariates, true_treatments = batch
        mu, logvar, z_s = self.model.get_latent(treatments, covariates)
        return {"mu": mu, "logvar": logvar, "z_s": z_s}
        
    def training_step(self, batch, batch_idx):
        treatments, covariates, true_treatments = batch
        decode_out, mu, logvar = self.model(treatments, covariates)

        treatment_loss, kldiv_loss = self.loss(decode_out, mu, logvar, true_treatments)
        # Combined objective
        if self.beta_epoch_max == 0:
            beta = self.beta_max
        else:
            beta = min(self.beta_max, self.current_epoch * (self.beta_max / self.beta_epoch_max))
        loss = treatment_loss + beta * kldiv_loss
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_treatment_loss", treatment_loss, on_epoch=True, prog_bar=True)
        self.log("train_kldiv_loss", kldiv_loss, on_epoch=True, prog_bar=True)
        self.log("beta", beta, on_epoch=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        treatments, covariates, true_treatments = batch
        decode_out, mu, logvar = self.model(treatments, covariates)

        treatment_loss, kldiv_loss = self.loss(decode_out, mu, logvar, true_treatments)
        # Combined objective
        if self.beta_epoch_max == 0:
            beta = self.beta_max
        else:
            beta = min(self.beta_max, self.current_epoch * (self.beta_max / self.beta_epoch_max))
        loss = treatment_loss + beta * kldiv_loss
        max_loss = treatment_loss + self.beta_max * kldiv_loss

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_max_loss", max_loss, on_epoch=True, prog_bar=True)
        self.log("val_treatment_loss", treatment_loss, on_epoch=True, prog_bar=True)
        self.log("val_kldiv_loss", kldiv_loss, on_epoch=True, prog_bar=True)

        # Predictive checks implementation - vectorized
        batch_size = covariates.size(0)

        # Compute test statistic for true treatments: T(a) = E_Z[log p(a|X,Z)]
        # Sample all z_s at once: [num_samples, batch_size, z_dim]
        z_s = self.model.reparameterize(
            mu.unsqueeze(0).expand(self.num_samples, -1, -1),
            logvar.unsqueeze(0).expand(self.num_samples, -1, -1)
        ).view(-1, mu.size(-1))  # [num_samples * batch_size, z_dim]

        # Expand covariates to match: [num_samples * batch_size, cov_dim]
        covariates_flat = covariates.reshape(batch_size, -1)
        cov_dim = covariates_flat.size(-1)
        covariates_expanded = covariates_flat.unsqueeze(0).expand(
            self.num_samples, -1, -1
        ).reshape(self.num_samples * batch_size, cov_dim)

        if self.binary_treatment:
            # --- Binary predictive checks (Bernoulli) ---
            a_probs = self.model.decode(covariates_expanded, z_s)
            a_probs = a_probs.view(self.num_samples, batch_size, -1)
            true_treatments_expanded = true_treatments.unsqueeze(-1).unsqueeze(0).expand(self.num_samples, -1, -1)

            log_probs = -F.binary_cross_entropy(
                a_probs, true_treatments_expanded, reduction='none'
            ).sum(dim=-1)

            test_stat_true = log_probs.mean(dim=0)

            z_s_mc = self.model.reparameterize(
                mu.unsqueeze(0).expand(self.num_samples, -1, -1),
                logvar.unsqueeze(0).expand(self.num_samples, -1, -1)
            ).view(-1, mu.size(-1))

            a_probs_mc = self.model.decode(covariates_expanded, z_s_mc)
            a_probs_mc = a_probs_mc.view(self.num_samples, batch_size, -1)
            a_mc = torch.bernoulli(a_probs_mc)

            z_s_inner = self.model.reparameterize(
                mu.unsqueeze(0).unsqueeze(0).expand(self.num_samples, self.num_samples, -1, -1),
                logvar.unsqueeze(0).unsqueeze(0).expand(self.num_samples, self.num_samples, -1, -1)
            ).view(-1, mu.size(-1))

            covariates_inner = covariates_flat.unsqueeze(0).unsqueeze(0).expand(
                self.num_samples, self.num_samples, -1, -1
            ).reshape(self.num_samples * self.num_samples * batch_size, cov_dim)

            a_probs_inner = self.model.decode(covariates_inner, z_s_inner)
            a_probs_inner = a_probs_inner.view(self.num_samples, self.num_samples, batch_size, -1)

            a_mc_expanded = a_mc.unsqueeze(1).expand(-1, self.num_samples, -1, -1)

            mc_log_probs = -F.binary_cross_entropy(
                a_probs_inner, a_mc_expanded, reduction='none'
            ).sum(dim=-1)
        else:
            # --- Continuous predictive checks (Gaussian) ---
            dec_out = self.model.decode(covariates_expanded, z_s)
            dec_mu_all, dec_logvar_all = dec_out
            dec_mu_all = dec_mu_all.view(self.num_samples, batch_size, -1)
            dec_logvar_all = dec_logvar_all.view(self.num_samples, batch_size, -1)
            true_treatments_expanded = true_treatments.unsqueeze(-1).unsqueeze(0).expand(self.num_samples, -1, -1)

            # Gaussian log-likelihood: -0.5 * (logvar + (x - mu)^2 / var)
            log_probs = -0.5 * (
                dec_logvar_all + (true_treatments_expanded - dec_mu_all) ** 2 / torch.exp(dec_logvar_all)
            ).sum(dim=-1)

            test_stat_true = log_probs.mean(dim=0)

            # Sample treatments from the learned Gaussian for MC check
            z_s_mc = self.model.reparameterize(
                mu.unsqueeze(0).expand(self.num_samples, -1, -1),
                logvar.unsqueeze(0).expand(self.num_samples, -1, -1)
            ).view(-1, mu.size(-1))

            dec_out_mc = self.model.decode(covariates_expanded, z_s_mc)
            dec_mu_mc, dec_logvar_mc = dec_out_mc
            dec_mu_mc = dec_mu_mc.view(self.num_samples, batch_size, -1)
            dec_logvar_mc = dec_logvar_mc.view(self.num_samples, batch_size, -1)
            # Sample from Gaussian: a ~ N(mu, sigma^2)
            a_mc = dec_mu_mc + torch.exp(0.5 * dec_logvar_mc) * torch.randn_like(dec_mu_mc)

            z_s_inner = self.model.reparameterize(
                mu.unsqueeze(0).unsqueeze(0).expand(self.num_samples, self.num_samples, -1, -1),
                logvar.unsqueeze(0).unsqueeze(0).expand(self.num_samples, self.num_samples, -1, -1)
            ).view(-1, mu.size(-1))

            covariates_inner = covariates_flat.unsqueeze(0).unsqueeze(0).expand(
                self.num_samples, self.num_samples, -1, -1
            ).reshape(self.num_samples * self.num_samples * batch_size, cov_dim)

            dec_out_inner = self.model.decode(covariates_inner, z_s_inner)
            dec_mu_inner, dec_logvar_inner = dec_out_inner
            dec_mu_inner = dec_mu_inner.view(self.num_samples, self.num_samples, batch_size, -1)
            dec_logvar_inner = dec_logvar_inner.view(self.num_samples, self.num_samples, batch_size, -1)

            a_mc_expanded = a_mc.unsqueeze(1).expand(-1, self.num_samples, -1, -1)

            mc_log_probs = -0.5 * (
                dec_logvar_inner + (a_mc_expanded - dec_mu_inner) ** 2 / torch.exp(dec_logvar_inner)
            ).sum(dim=-1)

        # Average over inner samples to get test statistics
        mc_test_stats = mc_log_probs.mean(dim=1)  # [num_samples, batch_size]

        # Compute predictive p-values
        comparisons = (mc_test_stats < test_stat_true.unsqueeze(0)).float()
        p_values = comparisons.mean(dim=0)  # Average over M samples

        # Log the mean p-value across the batch
        self.log("val_p_value", p_values.mean(), on_epoch=True, prog_bar=True)
        self.current_val_p_value = p_values.mean()
        self.current_val_treatment_loss = treatment_loss.detach().cpu().item()
        self.current_val_loss = loss.detach().cpu().item()

        torch.cuda.empty_cache()

        return loss
    
    def on_validation_epoch_end(self):
        # average over logged values from the epoch
        val_loss = self.trainer.callback_metrics.get("val_loss")
        val_treatment_loss = self.trainer.callback_metrics.get("val_treatment_loss")
        val_p_value = getattr(self, "current_val_p_value", None)

        if val_loss is not None:
            self.current_val_loss = float(val_loss.cpu().item())
        if val_treatment_loss is not None:
            self.current_val_treatment_loss = float(val_treatment_loss.cpu().item())
        if val_p_value is not None:
            self.current_val_p_value = float(val_p_value.cpu().item())
    
    def on_save_checkpoint(self, checkpoint):
        if hasattr(self, "current_val_loss"):
            checkpoint["val_loss"] = self.current_val_loss
        if hasattr(self, "current_val_treatment_loss"):
            checkpoint["val_treatment_loss"] = self.current_val_treatment_loss
        if hasattr(self, "current_val_p_value"):
            checkpoint["val_p_value"] = self.current_val_p_value
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), weight_decay=self.weight_decay, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
            
    def kldiv(self, mu, logvar):
        mu_ = mu.view(-1, self.model.patch_size * self.model.patch_size, self.model.per_location_latent_dim)
        logvar_ = logvar.view(-1, self.model.patch_size * self.model.patch_size, self.model.per_location_latent_dim)

        # Fix (b): Add epsilon regularization to avoid singular Laplacian
        # Prior precision should be τ(L + εI), not just τL
        laplacian = self.model.laplacian.to(mu.device)
        n_pixels = laplacian.shape[0]
        laplacian_reg = laplacian + self.model.eps * torch.eye(n_pixels, device=mu.device)

        # Quadratic term: μ^T (L + εI) μ
        mu_L = torch.matmul(mu_.transpose(1, 2), laplacian_reg)  # [batch, latent_dim, n_pixels]
        quadratic_term = torch.sum(mu_L * mu_.transpose(1, 2), dim=2)  # [batch, latent_dim]

        # Trace term: tr((L + εI) · diag(exp(logvar))) = sum(diag(L + εI) * exp(logvar))
        L_diag = torch.diag(laplacian_reg)  # [n_pixels]
        trace_term = torch.sum(L_diag.unsqueeze(0).unsqueeze(-1) * torch.exp(logvar_), dim=1)  # [batch, latent_dim]

        # Fix (a): Add posterior entropy term: -(1/2) * Σ logvar_i
        # Without this, there's no incentive for the model to produce tight posteriors
        entropy_term = -0.5 * torch.sum(logvar_, dim=(1, 2))  # [batch]

        # Full KL: (τ/2) * [μ^T(L+εI)μ + tr((L+εI)·Σ)] - (1/2) * Σ logvar_i
        kl_prior = self.model.tau / 2 * torch.sum(quadratic_term + trace_term, dim=1)  # [batch]
        kldiv_loss = kl_prior + entropy_term  # [batch]

        return kldiv_loss.mean() / (n_pixels * self.model.per_location_latent_dim)
    
    def loss(self, decode_out, mu, logvar, true_treatments):
        # Term 1: -log p_ψ(A_s | x_s, Z_s) - Treatment reconstruction
        if self.binary_treatment:
            treatment_loss = F.binary_cross_entropy(decode_out, true_treatments.float())
        else:
            # Gaussian negative log-likelihood for continuous treatments
            dec_mu, dec_logvar = decode_out
            treatment_loss = F.gaussian_nll_loss(
                dec_mu, true_treatments.float(), torch.exp(dec_logvar)
            )

        # Term 2: β * KL(q_φ || p_θ) - Spatial KL divergence
        kldiv_loss = self.kldiv(mu, logvar)

        return treatment_loss, kldiv_loss     

    
class UNetHead(pl.LightningModule):
    def __init__(
        self,
        feature_dim: int,
        radius: float,
        bilinear: bool = False,
        unet_base_chan: int = 16,
        weight_decay: float = 1e-5,
        lr: float = 1e-3,
        epochs: int = 10,
        datatype: str = "grid",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.radius = radius
        self.bilinear = bilinear
        self.unet_base_chan = unet_base_chan
        self.datatype = datatype
        
        if self.datatype != "grid":
            raise ValueError(f"Unsupported dataset type: {datatype}")
        
        self.model = UNet(n_channels=1 + self.feature_dim, n_classes=1, bilinear=self.bilinear, radius=self.radius, base_channels=self.unet_base_chan)
        self.model = torch.compile(self.model, mode="reduce-overhead")
        
        self.weight_decay = weight_decay
        self.lr = lr
        self.epochs = epochs
        
    def predict_step(self, batch, batch_idx):
        treatments, features, true_outcomes = batch
        
        treat = treatments.permute(0, 3, 1, 2)
        feat = features.permute(0, 3, 1, 2)
        head_input = torch.cat([treat, feat], dim=1)
        
        pred = self.model(head_input)
        return pred
        
    def training_step(self, batch, batch_idx):
        treatments, features, true_outcomes = batch
        treat = treatments.permute(0, 3, 1, 2)
        feat = features.permute(0, 3, 1, 2)
        head_input = torch.cat([treat, feat], dim=1)
        
        pred = self.model(head_input)
        
        loss = self.loss(pred, true_outcomes)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)        
        return loss
        
    def validation_step(self, batch, batch_idx):
        treatments, features, true_outcomes = batch
        treat = treatments.permute(0, 3, 1, 2)
        feat = features.permute(0, 3, 1, 2)
        head_input = torch.cat([treat, feat], dim=1)
        
        pred = self.model(head_input)
        
        loss = self.loss(pred, true_outcomes)        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), weight_decay=self.weight_decay, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
            
    
    def loss(self, pred, true_outcomes):        
        # Term 3: γ * E_q[(Y_s - h_θ(A_s,A_{N_s},x_s,Z_s))²] - Outcome prediction
        outcome_loss = F.mse_loss(pred, true_outcomes)
        return outcome_loss
     
        
def pad_center(tensor, target_size):
    """Pad 4D tensor (B,H,W,C) to target_size x target_size, centered."""
    b, h, w, c = tensor.shape
    if h == target_size and w == target_size:
        return tensor
    if c == 0:
        return torch.zeros(b, target_size, target_size, 0, dtype=tensor.dtype, device=tensor.device)
    pad_h = target_size - h
    pad_w = target_size - w
    assert pad_h >= 0 and pad_w >= 0
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    # for shape (B,H,W,C), need (left,right,top,bottom) padding order
    # F.pad applies padding from the last dimension backwards
    return F.pad(tensor, (0, 0, pad_left, pad_right, pad_top, pad_bottom))


class CVAEDataset(Dataset):
    def __init__(self, dataset, nodes, coords2id, radius, treat_scaler=None, feat_scaler=None, output_scaler=None, a=None, change=None, datatype="grid", dataset_radius=None, nbr_off=None):
        
        if datatype != "grid":
            raise ValueError(f"Unsupported dataset type: {datatype}")
        if dataset_radius is None:
            dataset_radius = radius
        
        treat = dataset.full_treatment.reshape(-1, 1)  # shape (N,), np.array
        cov = dataset.full_covariates   # shape (N, covariate_dim)
        out = dataset.full_outcome.reshape(-1, 1)      # shape (N,)

        # Identify non-binary columns
        nonbinary_treat_cols = [i for i in range(treat.shape[1]) if not np.all(np.isin(treat[:, i], [0, 1]))]
        nonbinary_cov_cols = [i for i in range(cov.shape[1]) if not np.all(np.isin(cov[:, i], [0, 1]))]
        nonbinary_out_cols = [i for i in range(out.shape[1]) if not np.all(np.isin(out[:, i], [0, 1]))]

        # Initialize or fit scalers
        if treat_scaler is None and nonbinary_treat_cols:
            treat_scaler = StandardScaler().fit(treat[:, nonbinary_treat_cols])
        if cov.shape[1] > 0 and feat_scaler is None and nonbinary_cov_cols:
            feat_scaler = StandardScaler().fit(cov[:, nonbinary_cov_cols])
        if output_scaler is None and nonbinary_out_cols:
            output_scaler = StandardScaler().fit(out[:, nonbinary_out_cols])

        self.treat_scaler = treat_scaler
        self.feat_scaler = feat_scaler
        self.output_scaler = output_scaler

        # Transform non-binary columns
        self.treat_scaled = treat.copy()
        if nonbinary_treat_cols and treat_scaler:
            self.treat_scaled[:, nonbinary_treat_cols] = treat_scaler.transform(treat[:, nonbinary_treat_cols])

        self.cov_scaled = cov.copy()
        if cov.shape[1] > 0 and nonbinary_cov_cols and feat_scaler:
            self.cov_scaled[:, nonbinary_cov_cols] = feat_scaler.transform(cov[:, nonbinary_cov_cols])

        self.out_scaled = out.copy()
        if nonbinary_out_cols and output_scaler:
            self.out_scaled[:, nonbinary_out_cols] = output_scaler.transform(out[:, nonbinary_out_cols])        
        
        true_treatment = torch.from_numpy(self.treat_scaled).float()
        covariates = torch.from_numpy(self.cov_scaled).float()
        outcomes = torch.from_numpy(self.out_scaled).float()
        ids = nodes
        id2coords = {v: k for k, v in coords2id.items()}

        self.treatment_vector = true_treatment[ids]
        self.covariate_vector = covariates[ids]
        self.outcome_vector = outcomes

        # Scale the counterfactual value `a` to match scaled treatment space
        a_scaled = a
        if a is not None and treat_scaler is not None and nonbinary_treat_cols:
            a_scaled = treat_scaler.transform(np.array([[a]]))[0, 0]

        self.treatments = torch.zeros(len(ids), 2*radius+1, 2*radius+1, 1)

        for ii, n in enumerate(nodes):
            center_coords = id2coords[n]
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    cur_coords = (center_coords[0] + i, center_coords[1] + j)
                    if change == "center" and (i, j) == (0, 0):
                        self.treatments[ii, i + radius, j + radius, 0] = float(a_scaled)
                    elif change == "nbr" and (i, j) != (0, 0):
                        self.treatments[ii, i + radius, j + radius, 0] = float(a_scaled)
                    elif change == "nbr_one" and nbr_off is not None and (i, j) == tuple(nbr_off):
                        self.treatments[ii, i + radius, j + radius, 0] = float(a_scaled)
                    else:
                        self.treatments[ii, i + radius, j + radius, 0] = true_treatment[coords2id[cur_coords]]

        effective_cov_radius = dataset_radius if cov.shape[1] > 0 else radius
        self.covariates = torch.zeros(len(ids), 2*effective_cov_radius+1, 2*effective_cov_radius+1, cov.shape[1])

        
        if cov.shape[1] > 0:
            for ii, n in enumerate(nodes):
                center_coords = id2coords[n]
                for i in range(-effective_cov_radius, effective_cov_radius + 1):
                    for j in range(-effective_cov_radius, effective_cov_radius + 1):
                        cur_coords = (center_coords[0] + i, center_coords[1] + j)
                        self.covariates[ii, i + effective_cov_radius, j + effective_cov_radius, :] = covariates[coords2id[cur_coords]]
        
        treat_size = 2*radius + 1
        cov_size = 2*dataset_radius + 1
        target_size = max(treat_size, cov_size)

        self.treatments = pad_center(self.treatments, target_size)
        self.covariates = pad_center(self.covariates, target_size)

        self.true_treatment = true_treatment[ids].squeeze()
        self.outcomes = outcomes[ids]
        
    def __len__(self):
        return len(self.outcomes)

    def __getitem__(self, idx):
        treatments = self.treatments[idx]
        covariates = self.covariates[idx]
        true_treatments = self.true_treatment[idx]
        # true_outcomes = self.outcomes[idx]
        
        return treatments, covariates, true_treatments


class UNetDataset(Dataset):
    def __init__(self, dataset, nodes, coords2id, radius, treat_scaler=None, feat_scaler=None, output_scaler=None, a=None, change=None, datatype="grid", latent=None, dataset_radius=None, nbr_off=None):
        
        if datatype != "grid":
            raise ValueError(f"Unsupported dataset type: {datatype}")
        if dataset_radius is None:
            dataset_radius = radius
        
        treat = dataset.full_treatment.reshape(-1, 1)  # shape (N,), np.array
        cov = dataset.full_covariates   # shape (N, covariate_dim)
        out = dataset.full_outcome.reshape(-1, 1)      # shape (N,)

        # Identify non-binary columns
        nonbinary_treat_cols = [i for i in range(treat.shape[1]) if not np.all(np.isin(treat[:, i], [0, 1]))]
        nonbinary_cov_cols = [i for i in range(cov.shape[1]) if not np.all(np.isin(cov[:, i], [0, 1]))]
        nonbinary_out_cols = [i for i in range(out.shape[1]) if not np.all(np.isin(out[:, i], [0, 1]))]

        # Initialize or fit scalers
        if treat_scaler is None and nonbinary_treat_cols:
            treat_scaler = StandardScaler().fit(treat[:, nonbinary_treat_cols])
        if feat_scaler is None and nonbinary_cov_cols:
            feat_scaler = StandardScaler().fit(cov[:, nonbinary_cov_cols])
        if output_scaler is None and nonbinary_out_cols:
            output_scaler = StandardScaler().fit(out[:, nonbinary_out_cols])

        self.treat_scaler = treat_scaler
        self.feat_scaler = feat_scaler
        self.output_scaler = output_scaler

        # Transform non-binary columns
        self.treat_scaled = treat.copy()
        if nonbinary_treat_cols and treat_scaler:
            self.treat_scaled[:, nonbinary_treat_cols] = treat_scaler.transform(treat[:, nonbinary_treat_cols])

        self.cov_scaled = cov.copy()
        if nonbinary_cov_cols and feat_scaler:
            self.cov_scaled[:, nonbinary_cov_cols] = feat_scaler.transform(cov[:, nonbinary_cov_cols])

        self.out_scaled = out.copy()
        if nonbinary_out_cols and output_scaler:
            self.out_scaled[:, nonbinary_out_cols] = output_scaler.transform(out[:, nonbinary_out_cols])        
        
        true_treatment = torch.from_numpy(self.treat_scaled).float()
        covariates = torch.from_numpy(self.cov_scaled).float()
        outcomes = torch.from_numpy(self.out_scaled).float()
        ids = nodes
        id2coords = {v: k for k, v in coords2id.items()}

        # Scale the counterfactual value `a` to match scaled treatment space
        a_scaled = a
        if a is not None and treat_scaler is not None and nonbinary_treat_cols:
            a_scaled = treat_scaler.transform(np.array([[a]]))[0, 0]

        self.treatments = torch.zeros(len(ids), 2*radius+1, 2*radius+1, 1)

        for ii, n in enumerate(nodes):
            center_coords = id2coords[n]
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    cur_coords = (center_coords[0] + i, center_coords[1] + j)
                    if change == "center" and (i, j) == (0, 0):
                        self.treatments[ii, i + radius, j + radius, 0] = float(a_scaled)
                    elif change == "nbr" and (i, j) != (0, 0):
                        self.treatments[ii, i + radius, j + radius, 0] = float(a_scaled)
                    elif change == "nbr_one" and nbr_off is not None and (i, j) == tuple(nbr_off):
                        self.treatments[ii, i + radius, j + radius, 0] = float(a_scaled)
                    else:
                        self.treatments[ii, i + radius, j + radius, 0] = true_treatment[coords2id[cur_coords]]

        self.covariates = torch.zeros(len(ids), 2*dataset_radius+1, 2*dataset_radius+1, cov.shape[1])
        
        if cov.shape[1] > 0:
            for ii, n in enumerate(nodes):
                center_coords = id2coords[n]
                for i in range(-dataset_radius, dataset_radius + 1):
                    for j in range(-dataset_radius, dataset_radius + 1):
                        cur_coords = (center_coords[0] + i, center_coords[1] + j)
                        self.covariates[ii, i + dataset_radius, j + dataset_radius, :] = covariates[coords2id[cur_coords]]

        treat_size = 2*radius + 1
        cov_size = 2*dataset_radius + 1
        
        self.latents = torch.from_numpy(latent)
        latent_size = self.latents.shape[1]  # spatial dim: 2*cvae_radius+1
        target_size = max(treat_size, cov_size, latent_size)

        self.treatments = pad_center(self.treatments, target_size)
        self.covariates = pad_center(self.covariates, target_size)
        self.latents = pad_center(self.latents, target_size)
        self.covariates = torch.cat([self.covariates, self.latents], dim=-1)
        
        self.true_treatment = true_treatment[ids].squeeze()
        self.outcomes = outcomes[ids]
        
    def __len__(self):
        return len(self.outcomes)

    def __getitem__(self, idx):
        treatments = self.treatments[idx]
        covariates = self.covariates[idx]
        # true_treatments = self.true_treatment[idx]
        true_outcomes = self.outcomes[idx]
        
        return treatments, covariates, true_outcomes


class Deconfounder(SpaceAlgo):
    """
    Wrapper of Interference-Aware Deconfounder with GPU acceleration support.
    """
    supports_binary = True
    supports_continuous = True
    
    def __init__(self,
        radius: float,
        head: str = "unet",
        encoder_conv1: int = 32,
        encoder_conv2: int = 64,
        kernel_size: int = 3,
        encoder_pool: str = "avg",
        connectivity: int = 4,
        tau: float = 10.0,
        eps: float = 1e-5,
        bilinear: bool = False,
        unet_base_chan: int = 16,
        weight_decay_cvae: float = 1e-5,
        lr_cvae: float = 1e-3,
        weight_decay_head: float = 1e-5,
        lr_head: float = 1e-3,
        beta_max: float = 1e-1,
        beta_epoch_max: int = 10,
        device: str = None,
        epochs_cvae: int = 50,
        epochs_head: int = 50,
        k: int = 100,
        max_iter: int = 20_000,
        lam_t: float = 0.001,
        lam_y: float = 0.001,
        spatial_split_kwargs: dict | None = None,
        batch_size = None,
        w_lags: int = 1,
        nbr_treatment_radius: int = None,
    ):
        """
        Initialize Interference-Aware Deconfounder with optional device specification.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.cvae_kwargs = {
            "radius": radius,
            "encoder_conv1": encoder_conv1,
            "encoder_conv2": encoder_conv2,
            "kernel_size": kernel_size,
            "encoder_pool": encoder_pool,
            "connectivity": connectivity,
            "tau": tau,
            "eps": eps,
            "weight_decay": weight_decay_cvae,
            "lr": lr_cvae,
            "beta_max": beta_max,
            "beta_epoch_max": beta_epoch_max,
            "epochs": epochs_cvae,
        }
        
        self.unet_kwargs = {
            "bilinear": bilinear,
            "weight_decay": weight_decay_head,
            "lr": lr_head,
            "epochs": epochs_head,
            "unet_base_chan": unet_base_chan,
        }
        
        self.spatialplus_kwargs = {
            "k": k,
            "max_iter": max_iter,
            "lam_t": lam_t,
            "lam_y": lam_y,
            "spatial_split_kwargs": spatial_split_kwargs,
        }
        
        self.s2sls_kwargs = {
            "w_lags": w_lags,
        }
        
        
        self.epochs_cvae = epochs_cvae
        self.epochs_head = epochs_head
        self.head = head
        self.cvae_radius = self.cvae_kwargs["radius"]
        self.nbr_treatment_radius = nbr_treatment_radius if nbr_treatment_radius is not None else self.cvae_radius
        
        if batch_size is not None:
            global total_batch_size
            total_batch_size = batch_size
        
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Store whether we're using GPU
        self.use_gpu = self.device.type == 'cuda'
        
        if self.use_gpu:
            LOGGER.debug(f"Using GPU acceleration on device: {self.device}")
        else:
            LOGGER.debug("Using CPU computation")
            
    
    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor on appropriate device."""
        return torch.from_numpy(array).float().to(self.device)
    
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor back to numpy array."""
        return tensor.detach().cpu().numpy()

    @staticmethod
    def _extract_neighbor_treatments(treatments):
        """Extract flattened neighbor treatments (center removed) from a treatment patch.

        Args:
            treatments: Tensor of shape (B, 2r+1, 2r+1, C)

        Returns:
            numpy array of shape (B, (2r+1)^2 - 1)
        """
        B, H, W, C = treatments.shape
        flat = treatments.view(B, -1, C)
        center_idx = (H * W) // 2
        flat_wo_center = torch.cat(
            [flat[:, :center_idx], flat[:, center_idx+1:]], dim=1
        )
        return flat_wo_center.squeeze(-1).cpu().numpy()

    def _get_neighbor_treatment_data(self, dataset, nodes, treat_scaler=None, feat_scaler=None, output_scaler=None, a=None, change=None):
        """Build a CVAEDataset at nbr_treatment_radius for extracting neighbor treatments."""
        return CVAEDataset(
            dataset, nodes, self.max_coords2id, self.nbr_treatment_radius,
            treat_scaler, feat_scaler, output_scaler,
            a=a, change=change, datatype=dataset.datatype,
            dataset_radius=dataset.conf_radius,
        )

    @staticmethod
    def _orthogonalize_latents_wrt_treatment(latents: np.ndarray, treatment: np.ndarray) -> np.ndarray:
        """Project out the treatment direction from latents.

        Z_ortho = Z - T_c * (T_c·Z) / (T_c·T_c),  T_c = T - mean(T).
        Prevents Z from over-explaining T in the SpatialPlus t-model and from
        absorbing the T->Y causal effect in the y-model.
        
        # This is only a problem in dataset with interference radius 2
        """
        T_c = treatment - treatment.mean()
        denom = float(T_c @ T_c)
        if denom < 1e-12:
            return latents
        proj = (T_c @ latents) / denom  # (d_z,)
        return latents - np.outer(T_c, proj)

    def fit(self, dataset: SpaceDataset, tune=False):
        self.tune = tune
        import wandb
        os.environ["WANDB_START_METHOD"] = "thread"
        os.environ["PYTORCH_LIGHTNING_DEBUG"] = "1"
        
        LOGGER.debug("Building cvae model...")

        self.binary_treatment = dataset.has_binary_treatment()
        self.cvae_kwargs["feature_dim"] = dataset.covariates.shape[1]
        self.cvae_kwargs["datatype"] = dataset.datatype
        self.cvae_kwargs["radius"] = max(self.cvae_radius, dataset.conf_radius)
        self.cvae_kwargs["binary_treatment"] = self.binary_treatment

        self.cvae_model = CVAE(**self.cvae_kwargs)
        
        LOGGER.debug("Building dataset and dataloader...")
        graph = nx.from_edgelist(dataset.full_edge_list)
        train_ix, test_ix, _ = spatial_train_test_split_radius(
            graph, 
            init_frac= 0.02, 
            levels = 1, 
            buffer = 1 + max(self.cvae_radius, self.nbr_treatment_radius, dataset.radius),
            radius = max(self.cvae_radius, self.nbr_treatment_radius, dataset.radius),
        )
        self.train_ix = train_ix
        self.test_ix = test_ix
        
        graph = nx.from_edgelist(dataset.full_edge_list)        
        node_list = list(graph.nodes())
        nbrs = {node: get_k_hop_neighbors(graph, node,  max(dataset.radius, self.cvae_radius, self.nbr_treatment_radius)) for node in node_list}
        nbr_counts = {node: len(neigh) for node, neigh in nbrs.items()}
        max_count = max(nbr_counts.values())
        
        self.max_nodes = [node for node, cnt in nbr_counts.items() if cnt == max_count]
        self.max_coords2id = {tuple(coord): i for i, coord in enumerate(dataset.full_coordinates)}
                
        self.cvae_traindata = CVAEDataset(dataset, self.train_ix, self.max_coords2id, self.cvae_radius, datatype=dataset.datatype, dataset_radius=dataset.conf_radius)
        self.cvae_valdata = CVAEDataset(dataset, self.test_ix, self.max_coords2id, self.cvae_radius, self.cvae_traindata.treat_scaler, self.cvae_traindata.feat_scaler, self.cvae_traindata.output_scaler, datatype=dataset.datatype, dataset_radius=dataset.conf_radius)
        
        loader = DataLoader(self.cvae_traindata, batch_size=total_batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(self.cvae_valdata, batch_size=total_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        self.cvae_model.steps_per_epoch = len(loader)
        
        
        LOGGER.debug("Preparing trainer...")
        callbacks = [
            ModelCheckpoint(
                dirpath="new_checkpoints/",
                filename="{epoch}-{val_max_loss:.2f}",
                monitor="val_max_loss",
                mode="min",
                save_top_k=3
            ),
            EarlyStopping(
                monitor="val_max_loss",
                patience=5,
                mode="min"
            ),
            LearningRateMonitor(logging_interval="step"),
            # RichProgressBar()
        ]
        
        # wandb_logger = WandbLogger(
        #     project="cvae-treatment-model",  # Your project name
        #     name="cvae-experiment",          # Run name
        #     log_model=False,                  # Log model checkpoints
        #     save_dir="./wandb_logs",
        #     config=self.cvae_kwargs,
        #     offline=True,
        # )
        self.cvae_trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            enable_checkpointing=True,
            logger=None,
            gradient_clip_val=1.0,
            enable_progress_bar=False,
            callbacks=callbacks,
            max_epochs=self.epochs_cvae,
            deterministic=True,
            enable_model_summary=True,
        )

        LOGGER.debug("Training cvae model...")
        self.cvae_trainer.fit(self.cvae_model, train_dataloaders=loader, val_dataloaders=val_loader)
        
        # Load the best checkpoint state dict
        best_checkpoint_path = self.cvae_trainer.checkpoint_callback.best_model_path
        if best_checkpoint_path:
            LOGGER.debug(f"Loading best model from: {best_checkpoint_path}")
            checkpoint = torch.load(best_checkpoint_path, map_location=self.cvae_model.device)
            self.cvae_model.load_state_dict(checkpoint['state_dict'])
            
            self.cur_val_p_value = checkpoint["val_p_value"]
            self.val_treatment_loss = checkpoint["val_treatment_loss"]
            
            del checkpoint
            torch.cuda.empty_cache()
            
            self.p_range = [0.25, 0.75]
            
            if self.cur_val_p_value < self.p_range[0] or self.cur_val_p_value > self.p_range[1]:
                LOGGER.debug(f"Validation p_value too low: {self.cur_val_p_value:.3f}")
                # return 100000
                # raise ValueError(f"Validation p_value too low: {self.cur_val_p_value:.3f}")
        else:
            LOGGER.warning("No best checkpoint found, using final epoch model")

        LOGGER.debug("Finished training cvae model.")
        
        
        LOGGER.debug("Getting latent dims...")
        
        latent_loader = DataLoader(self.cvae_traindata, batch_size=total_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        latent_val_loader = DataLoader(self.cvae_valdata, batch_size=total_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        train_latents = self.cvae_trainer.predict(self.cvae_model, latent_loader)
        train_latents = torch.cat([o[latent_idx] for o in train_latents]).cpu().numpy()
        
        val_latents = self.cvae_trainer.predict(self.cvae_model, latent_val_loader)
        val_latents = torch.cat([o[latent_idx] for o in val_latents]).cpu().numpy()
        
        del latent_loader, latent_val_loader
        torch.cuda.empty_cache()
        
        
        LOGGER.debug(f"Building outcome head {self.head}...")
        
        if self.head == "spatialplus" or self.head == "s2sls-lag1":

            spatialplusdata = CVAEDataset(dataset, self.max_nodes, self.max_coords2id, self.cvae_radius, datatype=dataset.datatype, dataset_radius=dataset.conf_radius)
            latent_loader = DataLoader(spatialplusdata, batch_size=total_batch_size, shuffle=False, num_workers=4, pin_memory=True)

            B = spatialplusdata.treatments.shape[0]
            if self.nbr_treatment_radius != 0:
                nbr_treat_data = self._get_neighbor_treatment_data(dataset, self.max_nodes)
                flat_wo_center = self._extract_neighbor_treatments(nbr_treat_data.treatments)

            val_latents = self.cvae_trainer.predict(self.cvae_model, latent_loader)
            val_latents = torch.cat([o[latent_idx] for o in val_latents]).view(B, -1).cpu().numpy()
            if "health" in dataset.parent_env and dataset.radius == 2:
                val_latents = self._orthogonalize_latents_wrt_treatment(val_latents, dataset.treatment)

            new_dataset = deepcopy(dataset)
            if self.nbr_treatment_radius != 0:
                new_dataset.covariates = np.concatenate([new_dataset.covariates, flat_wo_center, val_latents], axis=1)
            else:
                new_dataset.covariates = np.concatenate([new_dataset.covariates, val_latents], axis=1)

            if self.head == "spatialplus":
                if dataset.radius == 1:
                    self.head_model = SpatialPlus(**self.spatialplus_kwargs)
                elif dataset.radius == 2:
                    if "health" in dataset.parent_env:
                        self.head_model = SpatialPlus(**self.spatialplus_kwargs)
                    else:
                        self.head_model = SpatialPlusMedian(**self.spatialplus_kwargs, use_z=True)
                        self.head_model.n_z_covars = int(val_latents.shape[1])
                self.head_model.fit(new_dataset)
            elif self.head == "s2sls-lag1":
                from .pysal_spreg import GMLag
                self.head_model = GMLag(**self.s2sls_kwargs)
                self.s2sls_fit = False
            else:
                raise ValueError(f"Unsupported model type: {self.head}")
        
            
        if self.head == "unet":

            self.unet_kwargs["feature_dim"] = dataset.covariates.shape[1] + (self.cvae_kwargs["encoder_conv2"] // 2)
            self.unet_kwargs["datatype"] = dataset.datatype
            self.unet_kwargs["radius"] = max(self.cvae_radius, self.nbr_treatment_radius, dataset.conf_radius)

            self.head_model = UNetHead(**self.unet_kwargs)

            self.head_traindata = UNetDataset(dataset, self.train_ix, self.max_coords2id, self.nbr_treatment_radius, datatype=dataset.datatype, latent=train_latents, dataset_radius=dataset.conf_radius)
            self.head_valdata = UNetDataset(dataset, self.test_ix, self.max_coords2id, self.nbr_treatment_radius, self.head_traindata.treat_scaler, self.head_traindata.feat_scaler, self.head_traindata.output_scaler, datatype=dataset.datatype, latent=val_latents, dataset_radius=dataset.conf_radius)

            loader = DataLoader(self.head_traindata, batch_size=total_batch_size, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(self.head_valdata, batch_size=total_batch_size, shuffle=False, num_workers=4, pin_memory=True)


            LOGGER.debug("Preparing trainer...")
            callbacks = [
                ModelCheckpoint(
                    dirpath="new_unet_checkpoints/",
                    filename="{epoch}-{val_loss:.2f}",
                    monitor="val_loss",
                    mode="min",
                    save_top_k=3
                ),
                EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    mode="min"
                ),
                LearningRateMonitor(logging_interval="step"),
                # RichProgressBar()
            ]

            self.head_trainer = pl.Trainer(
                accelerator="auto",
                devices=1,
                enable_checkpointing=True,
                logger=None,
                gradient_clip_val=1.0,
                enable_progress_bar=False,
                callbacks=callbacks,
                max_epochs=self.epochs_head,
                deterministic=True,
                enable_model_summary=True,
            )

            LOGGER.debug("Training outcome model...")
            self.head_trainer.fit(self.head_model, train_dataloaders=loader, val_dataloaders=val_loader)

            # Load the best checkpoint state dict
            best_checkpoint_path = self.head_trainer.checkpoint_callback.best_model_path
            if best_checkpoint_path:
                LOGGER.debug(f"Loading best model from: {best_checkpoint_path}")
                checkpoint = torch.load(best_checkpoint_path, map_location=self.head_model.device)
                self.head_model.load_state_dict(checkpoint['state_dict'])

                del checkpoint
                torch.cuda.empty_cache()
            else:
                LOGGER.warning("No best checkpoint found, using final epoch model")

            LOGGER.debug("Finished training outcome model.")

        if self.head == "temporal_unet":
            # ---- Temporal UNet head: CVAE latents + full spatial maps ----
            # Extract CVAE latents for ALL max_nodes (center pixel only)
            LOGGER.debug("Extracting latents for full spatial grid...")
            all_data = CVAEDataset(
                dataset, self.max_nodes, self.max_coords2id, self.cvae_radius,
                self.cvae_traindata.treat_scaler, self.cvae_traindata.feat_scaler,
                self.cvae_traindata.output_scaler, datatype=dataset.datatype,
                dataset_radius=dataset.conf_radius,
            )
            all_loader = DataLoader(all_data, batch_size=total_batch_size, shuffle=False, num_workers=4, pin_memory=True)
            all_latents = self.cvae_trainer.predict(self.cvae_model, all_loader)
            # Center pixel latent: (N_nodes, latent_dim)
            all_latents = torch.cat([o[latent_idx] for o in all_latents]).cpu().numpy()
            per_loc_dim = self.cvae_kwargs["encoder_conv2"] // 2
            patch_size = 2 * max(self.cvae_radius, dataset.conf_radius) + 1
            center = patch_size // 2
            all_latents = all_latents.reshape(-1, patch_size, patch_size, per_loc_dim)
            all_latents = all_latents[:, center, center, :]  # (N_nodes, latent_dim)

            # Map latents to (H, W, latent_dim) grid
            grid_H, grid_W = dataset.grid_H, dataset.grid_W
            latent_map = np.zeros((grid_H, grid_W, per_loc_dim), dtype=np.float32)
            coords = dataset.full_coordinates
            for i, node_idx in enumerate(self.max_nodes):
                r, c = coords[node_idx]
                latent_map[r, c, :] = all_latents[i]

            # Broadcast to (latent_dim, H, W) and tile across timesteps
            latent_channels = latent_map.transpose(2, 0, 1)  # (latent_dim, H, W)
            n_time = dataset.X.shape[0]
            latent_tiled = np.broadcast_to(
                latent_channels[None, :, :, :], (n_time, per_loc_dim, grid_H, grid_W)
            ).copy()

            # Augmented input: X[t] + latent channels → (T-1, 15+latent_dim, H, W)
            X_aug = np.concatenate([dataset.X, latent_tiled], axis=1).astype(np.float32)
            self._latent_channels = latent_channels  # save for predict
            self._per_loc_dim = per_loc_dim

            n_channels_aug = X_aug.shape[1]
            train_ds = TemporalUNetDataset(X_aug, dataset.Y, dataset.valid_mask, dataset.train_idx)
            val_ds   = TemporalUNetDataset(X_aug, dataset.Y, dataset.valid_mask, dataset.val_idx)

            t_batch = min(total_batch_size, 16)
            loader = DataLoader(train_ds, batch_size=t_batch, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=t_batch, shuffle=False, num_workers=4, pin_memory=True)

            self.head_model = TemporalUNetHead(
                n_channels=n_channels_aug,
                base_channels=self.unet_kwargs.get("unet_base_chan", 32),
                n_downs=3,
                bilinear=self.unet_kwargs.get("bilinear", True),
                weight_decay=self.unet_kwargs.get("weight_decay", 1e-5),
                lr=self.unet_kwargs.get("lr", 1e-3),
                epochs=self.epochs_head,
            )

            callbacks = [
                ModelCheckpoint(dirpath="temporal_cvae_unet_ckpt/", monitor="val_loss", mode="min", save_top_k=3),
                EarlyStopping(monitor="val_loss", patience=10, mode="min"),
                LearningRateMonitor(logging_interval="step"),
            ]

            self.head_trainer = pl.Trainer(
                accelerator="auto", devices=1,
                enable_checkpointing=True, logger=None,
                gradient_clip_val=1.0, enable_progress_bar=False,
                callbacks=callbacks, max_epochs=self.epochs_head,
                deterministic=True,
            )

            LOGGER.debug("Training temporal UNet outcome head...")
            self.head_trainer.fit(self.head_model, train_dataloaders=loader, val_dataloaders=val_loader)

            best = self.head_trainer.checkpoint_callback.best_model_path
            if best:
                ckpt = torch.load(best, map_location=self.head_model.device)
                self.head_model.load_state_dict(ckpt['state_dict'])
                del ckpt
                torch.cuda.empty_cache()
            LOGGER.debug("Finished training temporal UNet outcome head.")

        
        
        
    # ---- Temporal UNet helpers ----------------------------------------

    def _augment_X(self, X_input):
        """Concatenate CVAE latent channels to temporal input maps."""
        n_time = X_input.shape[0]
        latent_tiled = np.broadcast_to(
            self._latent_channels[None, :, :, :],
            (n_time, self._per_loc_dim, X_input.shape[2], X_input.shape[3]),
        ).copy()
        return np.concatenate([X_input, latent_tiled], axis=1).astype(np.float32)

    def _temporal_predict_maps(self, X_input, valid_mask):
        """Run temporal UNet on (N, C, H, W).  Returns (N, H, W)."""
        self.head_model.eval()
        X_aug = self._augment_X(X_input)
        dummy_Y = np.zeros((X_aug.shape[0], X_aug.shape[2], X_aug.shape[3]), dtype=np.float32)
        ds = TemporalUNetDataset.__new__(TemporalUNetDataset)
        ds.X = torch.from_numpy(X_aug).float()
        ds.Y = torch.from_numpy(dummy_Y).float()
        ds.mask = torch.from_numpy(valid_mask).float().unsqueeze(0).expand(len(X_aug), -1, -1)
        t_batch = min(total_batch_size, 16)
        loader = DataLoader(ds, batch_size=t_batch, shuffle=False, num_workers=4, pin_memory=True)
        preds = torch.cat(self.head_trainer.predict(self.head_model, loader)).squeeze(1).cpu().numpy()
        return preds

    # ---- eval / tune_metric -------------------------------------------

    def eval(self, dataset: SpaceDataset) -> dict:
        """Evaluate the model."""

        if self.head == "temporal_unet":
            return self._temporal_eval(dataset)

        LOGGER.debug("Computing counterfactuals...")
        ite = []
        for a in dataset.treatment_values:
            preds_a = self.predict(dataset, self.max_nodes, a=a, change="center")
            ite.append(preds_a)
        ite = np.concatenate(ite, axis=1)

        effects = {"erf": ite.mean(0), "ite": ite}

        if dataset.has_binary_treatment():
            effects["ate"] = effects["erf"][1] - effects["erf"][0]

            from sci.env import PER_NEIGHBOR_SPILLOVER
            if PER_NEIGHBOR_SPILLOVER:
                # Per-neighbor spillover: flip ONE neighbor at a time,
                # then average the effect across all neighbor positions.
                model_radius = self.nbr_treatment_radius
                nbr_positions = [
                    (dr, dc)
                    for dr in range(-model_radius, model_radius + 1)
                    for dc in range(-model_radius, model_radius + 1)
                    if not (dr == 0 and dc == 0)
                ]
                LOGGER.debug(
                    f"Per-nbr spillover: averaging over {len(nbr_positions)} "
                    f"model neighbor positions"
                )
                per_nbr_effects = []
                for (dr, dc) in nbr_positions:
                    preds_0 = self.predict(
                        dataset, self.max_nodes,
                        a=dataset.treatment_values[0],
                        change="nbr_one", nbr_off=(dr, dc),
                    )[:, 0]
                    preds_1 = self.predict(
                        dataset, self.max_nodes,
                        a=dataset.treatment_values[1],
                        change="nbr_one", nbr_off=(dr, dc),
                    )[:, 0]
                    per_nbr_effects.append(preds_1 - preds_0)
                effects["spill"] = float(np.mean(per_nbr_effects))
            else:
                # Original: set ALL neighbors to each treatment value
                spill = []
                for a in dataset.treatment_values:
                    preds_a = self.predict(dataset, self.max_nodes, a=a, change="nbr")
                    spill.append(preds_a)
                spill = np.concatenate(spill, axis=1)
                s = spill.mean(0)
                effects["spill"] = s[1] - s[0]

        # Per-pixel counterfactual: annual and summer % increase in SIC
        if hasattr(dataset, "cf_annual_treatment") and dataset.cf_annual_treatment is not None:
            LOGGER.debug("Computing annual / summer counterfactual effects...")

            preds_annual_f = self.predict(
                dataset, self.max_nodes, a=None, change=None,
            )
            preds_annual_cf = self.predict(
                dataset, self.max_nodes, a=None, change=None,
                cf_full_treatment=dataset.cf_annual_treatment,
            )
            annual_diff = preds_annual_cf - preds_annual_f
            effects["cf_annual_pct"] = float(
                annual_diff.mean() / np.abs(preds_annual_f).mean() * 100
            )

            preds_summer_f = self.predict(
                dataset, self.max_nodes, a=None, change=None,
                cf_full_treatment=dataset.summer_treatment,
            )
            preds_summer_cf = self.predict(
                dataset, self.max_nodes, a=None, change=None,
                cf_full_treatment=dataset.cf_summer_treatment,
            )
            summer_diff = preds_summer_cf - preds_summer_f
            effects["cf_summer_pct"] = float(
                summer_diff.mean() / np.abs(preds_summer_f).mean() * 100
            )

        return effects

    def _temporal_eval(self, dataset: SpaceDataset) -> dict:
        """Eval for temporal_unet head using full spatial maps."""
        effects = {}
        mask = dataset.valid_mask

        node_rows = dataset.coordinates[:, 0]
        node_cols = dataset.coordinates[:, 1]

        # ERF over treatment quantiles — per pixel (averaged over time)
        ite = []
        for a in dataset.treatment_values:
            X_a = dataset.X.copy()
            X_a[:, 1, :, :] = (a - dataset.lwdn_mu) / dataset.lwdn_sd
            X_a[:, 1, :, :][..., ~mask] = 0.0
            preds_a = self._temporal_predict_maps(X_a, mask)  # (T-1, H, W)
            avg_a = preds_a.mean(axis=0)  # (H, W)
            per_node = avg_a[node_rows, node_cols]
            ite.append(per_node.reshape(-1, 1))
        ite = np.concatenate(ite, axis=1)
        effects["erf"] = ite.mean(0)
        effects["ite"] = ite

        # Counterfactual: 5% LWDN reduction
        # Predictions are in standardized SIC space — inverse-standardize
        # before computing % change so the baseline is in real units.
        sic_mu, sic_sd = dataset.sic_mu, dataset.sic_sd
        def to_raw(p):
            return p * sic_sd + sic_mu

        if hasattr(dataset, "X_cf"):
            LOGGER.debug("Computing annual / summer counterfactual effects...")
            preds_f  = to_raw(self._temporal_predict_maps(dataset.X_factual, mask))
            preds_cf = to_raw(self._temporal_predict_maps(dataset.X_cf, mask))

            diff_annual = (preds_cf - preds_f)[:, mask]
            base_annual = np.abs(preds_f[:, mask])
            effects["cf_annual_pct"] = float(diff_annual.mean() / base_annual.mean() * 100)

            jja = dataset.jja_mask
            diff_summer = (preds_cf[jja] - preds_f[jja])[:, mask]
            base_summer = np.abs(preds_f[jja][:, mask])
            effects["cf_summer_pct"] = float(diff_summer.mean() / base_summer.mean() * 100)

        # Counterfactual: +18 W/m² LWDN increase
        if hasattr(dataset, "X_cf_plus18"):
            LOGGER.debug("Computing +18 LWDN counterfactual effects...")
            preds_f18  = to_raw(self._temporal_predict_maps(dataset.X_factual, mask))
            preds_cf18 = to_raw(self._temporal_predict_maps(dataset.X_cf_plus18, mask))

            diff18 = (preds_cf18 - preds_f18)[:, mask]
            base18 = np.abs(preds_f18[:, mask])
            effects["cf_plus18_annual_pct"] = float(diff18.mean() / base18.mean() * 100)

            jja = dataset.jja_mask
            diff18_s = (preds_cf18[jja] - preds_f18[jja])[:, mask]
            base18_s = np.abs(preds_f18[jja][:, mask])
            effects["cf_plus18_summer_pct"] = float(diff18_s.mean() / base18_s.mean() * 100)

        return effects

    def tune_metric(self, dataset: SpaceDataset) -> float:
        if not self.tune:
            if self.cur_val_p_value < self.p_range[0] or self.cur_val_p_value > self.p_range[1]:
                return 10000
        if self.head == "temporal_unet":
            preds = self._temporal_predict_maps(dataset.X[dataset.test_idx], dataset.valid_mask)
            Y_true = dataset.Y[dataset.test_idx]
            mask = dataset.valid_mask
            return float(np.mean((preds[:, mask] - Y_true[:, mask]) ** 2))

        if self.head == "unet":
            preds = self.predict(dataset, self.max_nodes, a=None, change=None)[:, 0]
            return np.mean((dataset.full_outcome[self.max_nodes] - preds) ** 2)
        elif self.head == "spatialplus" or self.head == "s2sls-lag1":
            predict_data = CVAEDataset(dataset, self.max_nodes, self.max_coords2id, self.cvae_radius, self.cvae_traindata.treat_scaler, self.cvae_traindata.feat_scaler, self.cvae_traindata.output_scaler, datatype=dataset.datatype, dataset_radius=dataset.conf_radius)

            loader = DataLoader(predict_data, batch_size=total_batch_size, shuffle=False, num_workers=4, pin_memory=True)
            latents = self.cvae_trainer.predict(self.cvae_model, loader)
            latents = torch.cat([o[latent_idx] for o in latents]).view(predict_data.treatments.shape[0], -1).cpu().numpy()

            if np.isnan(latents).any():
                return 100000
            
            if "health" in dataset.parent_env and dataset.radius == 2:
                latents = self._orthogonalize_latents_wrt_treatment(latents, dataset.treatment)
            new_dataset = deepcopy(dataset)
            if self.nbr_treatment_radius != 0:
                nbr_treat_data = self._get_neighbor_treatment_data(
                    dataset, self.max_nodes,
                    self.cvae_traindata.treat_scaler, self.cvae_traindata.feat_scaler,
                    self.cvae_traindata.output_scaler,
                )
                flat_wo_center = self._extract_neighbor_treatments(nbr_treat_data.treatments)

                new_dataset.covariates = np.concatenate([new_dataset.covariates, flat_wo_center, latents], axis=1)
            else:
                new_dataset.covariates = np.concatenate([new_dataset.covariates, latents], axis=1)
            if self.head == "spatialplus":
                m = self.head_model.tune_metric(new_dataset)
                if np.isnan(m):
                    return 1000000
                else:
                    return m
            if self.head == "s2sls-lag1":
                return self.val_treatment_loss
                
                
    
    
    def predict(self, dataset: SpaceDataset, nodes, a, change,
                cf_full_treatment=None, nbr_off=None) -> dict:
        """
        Get outcome predictions with GPU acceleration for large datasets.

        Parameters
        ----------
        cf_full_treatment : np.ndarray | None
            If provided, the outcome head uses this array as ``full_treatment``
            instead of the observed values.  The CVAE still encodes the
            *observed* treatments so the latent confounder stays fixed.
            Shape must match ``dataset.full_treatment``.
        """
        self.cvae_model.eval()
        if self.head == "unet":
            self.head_model.eval()

        # CVAE: always encode from observed treatments -> latents
        predict_data = CVAEDataset(dataset, nodes, self.max_coords2id, self.cvae_radius, self.cvae_traindata.treat_scaler, self.cvae_traindata.feat_scaler, self.cvae_traindata.output_scaler, datatype=dataset.datatype, dataset_radius=dataset.conf_radius)

        loader = DataLoader(predict_data, batch_size=total_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        latents = self.cvae_trainer.predict(self.cvae_model, loader)
        latents = torch.cat([o[latent_idx] for o in latents])

        if self.head == "unet":
            latents = latents.cpu().numpy()
            torch.cuda.empty_cache()
        elif self.head == "spatialplus" or self.head == "s2sls-lag1":
            latents = latents.view(predict_data.treatments.shape[0], -1).cpu().numpy()
            if "health" in dataset.parent_env and dataset.radius == 2:
                latents = self._orthogonalize_latents_wrt_treatment(latents, dataset.treatment)

        # Build the dataset seen by the outcome head.  When
        # cf_full_treatment is given, swap in the counterfactual treatment
        # so that patch extraction + scaling use the intervened values.
        if cf_full_treatment is not None:
            head_dataset = deepcopy(dataset)
            head_dataset.full_treatment = cf_full_treatment
            head_a, head_change = None, None
        else:
            head_dataset = dataset
            head_a, head_change = a, change

        if self.head == "unet":
            predict_head_data = UNetDataset(head_dataset, nodes, self.max_coords2id, self.nbr_treatment_radius, self.head_traindata.treat_scaler, self.head_traindata.feat_scaler, self.head_traindata.output_scaler, a=head_a, change=head_change, datatype=dataset.datatype, latent=latents, dataset_radius=dataset.conf_radius, nbr_off=nbr_off)

            loader = DataLoader(predict_head_data, batch_size=total_batch_size, shuffle=False, num_workers=4, pin_memory=True)
            preds = torch.cat(self.head_trainer.predict(self.head_model, loader))
            preds = preds.cpu().numpy()
            # scale back the preds
            preds = self.head_traindata.output_scaler.inverse_transform(preds)

        elif self.head == "spatialplus" or self.head == "s2sls-lag1":
            new_dataset = deepcopy(dataset)
            if cf_full_treatment is not None:
                new_dataset.treatment = cf_full_treatment[nodes] if len(cf_full_treatment) > len(nodes) else cf_full_treatment
            elif change == "center" and a is not None:
                new_dataset.treatment = np.full_like(new_dataset.treatment, a)
            if self.nbr_treatment_radius != 0:
                # Build counterfactual treatment patch for neighbor covariates
                counterfactual_data = CVAEDataset(
                    head_dataset, nodes, self.max_coords2id, self.nbr_treatment_radius,
                    self.cvae_traindata.treat_scaler, self.cvae_traindata.feat_scaler,
                    self.cvae_traindata.output_scaler, a=head_a, change=head_change,
                    datatype=dataset.datatype, dataset_radius=dataset.conf_radius
                )
                flat_wo_center = self._extract_neighbor_treatments(counterfactual_data.treatments)

                new_dataset.covariates = np.concatenate([new_dataset.covariates, flat_wo_center, latents], axis=1)
            else:
                new_dataset.covariates = np.concatenate([new_dataset.covariates, latents], axis=1)

            if self.head == "s2sls-lag1" and not self.s2sls_fit:
                self.head_model.fit(new_dataset)
                self.s2sls_fit = True

            return self.head_model.predict(new_dataset)


        return preds
    
    def plot_latent(self, dataset: SpaceDataset, filename):
        """
        Plot PCA of latent values with GPU acceleration for large datasets.
        """
        nodes = self.max_nodes
        a = None
        change = None
        
        
        predict_data = CVAEDataset(dataset, nodes, self.max_coords2id, self.cvae_radius, self.cvae_traindata.treat_scaler, self.cvae_traindata.feat_scaler, self.cvae_traindata.output_scaler, datatype=dataset.datatype, a=a, change=change, dataset_radius=dataset.conf_radius)
        
        loader = DataLoader(predict_data, batch_size=total_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        outs = self.cvae_trainer.predict(self.cvae_model, loader)
        mu = torch.cat([o["mu"] for o in outs]).cpu().numpy()
        logvar = torch.cat([o["logvar"] for o in outs]).cpu().numpy()
                
        new_ids = {old_id: new_id for new_id, old_id in enumerate(nodes)}
        small_coords2id = {coord: new_ids[id_] for coord, id_ in self.max_coords2id.items() if id_ in new_ids}
        
        torch.save({
            'mu': mu, 
            'logvar': logvar,
            'coords2id': small_coords2id, 
            'missing_covariates': dataset.missing_covariates[nodes],  
            'treatment': predict_data.treatment_vector,
            'covariates': predict_data.covariate_vector
        }, 'saved_data.pt')
        
        # save_maps_grid(self.max_coords2id, latents, use_center_only=False, filename=filename, standardize=True, use_pca=True)
        # save_maps_grid(small_coords2id, mu, use_center_only=True, filename=filename.replace('.pdf', '_mucenter.pdf'), standardize=True, use_pca=False, fit_on_all_pixels=False)
        # save_maps_grid(small_coords2id, logvar, use_center_only=True, filename=filename.replace('.pdf', '_varcenter.pdf'), standardize=True, use_pca=False, fit_on_all_pixels=False)
        # save_maps_grid(small_coords2id, dataset.missing_covariates[nodes], filename=filename, standardize=True, use_pca=False)
            
    

def save_maps_grid(max_coords2id, latents, n_pca_components=10, 
                   use_center_only=True, filename=None, standardize=True,
                   use_pca=True, feature_column=None, fit_on_all_pixels=False):
    """
    Display latent maps in a grid, with optional PCA reduction.
    
    Parameters:
    -----------
    max_coords2id : dict
        Dictionary mapping 2D coordinates (row, col) to indices
    latents : np.ndarray
        Array of shape (N, patch_size, patch_size, C) containing latent features
    n_pca_components : int, default 3
        Number of PCA components to compute and visualize (only used if use_pca=True)
    use_center_only : bool, default True
        If True, use only center pixel of each patch
        If False, use all pixels in the patch (flattened)
    filename : str, optional
        Base path to save figures. If None, figures are not saved.
        For each component, appends _component_{i}.png to the filename
    standardize : bool, default True
        If True, standardize components to z-scores and clip to ±3 std
        If False, use original component range for color mapping
    use_pca : bool, default True
        If True, apply PCA to features and plot components
        If False, plot a single feature column directly
    feature_column : int, optional
        Column index to plot when use_pca=False
        If None when use_pca=False, defaults to 0
    fit_on_all_pixels : bool, default False
        Only used when use_pca=True. If True, fit PCA on all pixels (flattened),
        then transform and plot only the center pixel. If False, fit and plot
        on the same features (controlled by use_center_only).
    """
    
    # Extract features from latents for PCA fitting
    if use_pca:
        if fit_on_all_pixels:
            # Fit PCA on all pixels (flattened)
            N, patch_size, _, C = latents.shape
            fit_features = latents.reshape(N*patch_size*patch_size, C)
            
            # Extract center pixel only for plotting
            center_idx = latents.shape[1] // 2
            plot_features_data = latents[:, center_idx, center_idx, :]  # Shape: (N, C)
        else:
            if use_center_only:
                # Use center pixel only for both fitting and plotting
                center_idx = latents.shape[1] // 2
                features = latents[:, center_idx, center_idx, :]  # Shape: (N, C)
            else:
                # Use all pixels, flattened for both fitting and plotting
                N, patch_size, _, C = latents.shape
                features = latents.reshape(N, -1)  # Shape: (N, patch_size*patch_size*C)
            fit_features = features
            plot_features_data = features
    else:
        plot_features_data = latents
    
    # Infer grid shape from max_coords2id
    coords = list(max_coords2id.keys())
    max_row = max(c[0] for c in coords)
    max_col = max(c[1] for c in coords)
    grid_shape = (max_row + 1, max_col + 1)
    
    if use_pca:
        # Apply PCA
        # pca = PCA(n_components=min(n_pca_components, fit_features.shape[1]))
        # pca.fit(fit_features)
        # plot_features = pca.transform(plot_features_data)  # Shape: (N, n_pca_components)
        
        # from sklearn.decomposition import FastICA
        # ica = FastICA(n_components=min(n_pca_components, fit_features.shape[1]), random_state=0)
        # ica.fit(fit_features)
        # plot_features = ica.transform(plot_features_data)
        
        from sklearn.decomposition import KernelPCA

        kpca = KernelPCA(
            n_components=min(n_pca_components, fit_features.shape[1]),
            kernel='rbf',
            gamma=None,
            fit_inverse_transform=False,
            random_state=0
        )
        kpca.fit(fit_features)
        plot_features = kpca.transform(plot_features_data)








        component_names = [f"PCA component {i}" for i in range(plot_features.shape[1])]
    else:
        # Use single feature column
        if feature_column is None:
            feature_column = 0
        plot_features = plot_features_data[:, feature_column:feature_column+1]  # Shape: (N, 1)
        component_names = [f"Feature column {feature_column}"]
    
    # Plot each component in a separate figure
    n_components = plot_features.shape[1]
    
    for comp_idx in range(n_components):
        values = plot_features[:, comp_idx]
        
        if standardize:
            # Standardize and clip to ±3 std
            mean = values.mean()
            std = values.std()
            plot_values = ((values - mean) / std).clip(min=-3, max=3)
            vmin, vmax = -3, 3
        else:
            # Use original component range
            plot_values = values
            vmin = values.min()
            vmax = values.max()
        
        # Create empty grid filled with NaN
        grid = np.full(grid_shape, np.nan)
        
        # Place each value at its specified (row, col) position
        for (row, col), idx in max_coords2id.items():
            if 0 <= idx < len(plot_values):
                grid[row, col] = plot_values[idx]
        
        # Normalize for color mapping
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('RdBu_r')
        
        width = 10
        aspect_correction = 1.2938694780251683
        height = width * aspect_correction
        fig, ax = plt.subplots(figsize=(width, height))
        im = ax.imshow(grid, cmap=cmap, norm=norm, origin='upper', aspect=aspect_correction)
        
        # Add colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cbar = fig.colorbar(sm, ax=ax, shrink=0.35)
        cbar.ax.tick_params(labelsize=14)
        
        ax.axis('off')
        plt.tight_layout()
        
        # Save if filename provided
        if filename is not None:
            base, ext = os.path.splitext(filename)
            if use_pca:
                component_filename = f"{base}_latent_component_{comp_idx}{ext}"
            else:
                component_filename = f"{base}_miss_feature_column_{feature_column}{ext}"
            plt.savefig(component_filename, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {component_filename}")
        
        plt.show()

# def save_maps_grid(max_coords2id, latents, n_pca_components=3, 
#                    use_center_only=True, filename=None, standardize=True,
#                    use_pca=True, feature_column=None):
#     """
#     Display latent maps in a grid, with optional PCA reduction.
    
#     Parameters:
#     -----------
#     max_coords2id : dict
#         Dictionary mapping 2D coordinates (row, col) to indices
#     latents : np.ndarray
#         Array of shape (N, patch_size, patch_size, C) containing latent features
#     n_pca_components : int, default 3
#         Number of PCA components to compute and visualize (only used if use_pca=True)
#     use_center_only : bool, default True
#         If True, use only center pixel of each patch
#         If False, use all pixels in the patch (flattened)
#     filename : str, optional
#         Base path to save figures. If None, figures are not saved.
#         For each component, appends _component_{i}.png to the filename
#     standardize : bool, default True
#         If True, standardize components to z-scores and clip to ±3 std
#         If False, use original component range for color mapping
#     use_pca : bool, default True
#         If True, apply PCA to features and plot components
#         If False, plot a single feature column directly
#     feature_column : int, optional
#         Column index to plot when use_pca=False
#         If None when use_pca=False, defaults to 0
#     """
    
#     # Extract features from latents
#     if use_pca:
#         if use_center_only:
#             # Use center pixel only
#             center_idx = latents.shape[1] // 2
#             features = latents[:, center_idx, center_idx, :]  # Shape: (N, C)
#         else:
#             # Use all pixels, flattened
#             N, patch_size, _, C = latents.shape
#             features = latents.reshape(N, -1)  # Shape: (N, patch_size*patch_size*C)
#     else:
#         features = latents
    
#     # Infer grid shape from max_coords2id
#     coords = list(max_coords2id.keys())
#     max_row = max(c[0] for c in coords)
#     max_col = max(c[1] for c in coords)
#     grid_shape = (max_row + 1, max_col + 1)
    
#     if use_pca:
#         # Apply PCA
#         pca = PCA(n_components=min(n_pca_components, features.shape[1]))
#         plot_features = pca.fit_transform(features)  # Shape: (N, n_pca_components)
#         component_names = [f"PCA component {i}" for i in range(plot_features.shape[1])]
#     else:
#         # Use single feature column
#         if feature_column is None:
#             feature_column = 0
#         plot_features = features[:, feature_column:feature_column+1]  # Shape: (N, 1)
#         component_names = [f"Feature column {feature_column}"]
    
#     # Plot each component in a separate figure
#     n_components = plot_features.shape[1]
    
#     for comp_idx in range(n_components):
#         values = plot_features[:, comp_idx]
        
#         if standardize:
#             # Standardize and clip to ±3 std
#             mean = values.mean()
#             std = values.std()
#             plot_values = ((values - mean) / std).clip(min=-3, max=3)
#             vmin, vmax = -3, 3
#         else:
#             # Use original component range
#             plot_values = values
#             vmin = values.min()
#             vmax = values.max()
        
#         # Create empty grid filled with NaN
#         grid = np.full(grid_shape, np.nan)
        
#         # Place each value at its specified (row, col) position
#         for (row, col), idx in max_coords2id.items():
#             if 0 <= idx < len(plot_values):
#                 grid[row, col] = plot_values[idx]
        
#         # Normalize for color mapping
#         norm = Normalize(vmin=vmin, vmax=vmax)
#         cmap = plt.get_cmap('RdBu_r')
        
#         width = 10
#         aspect_correction = 1.2938694780251683
#         height = width * aspect_correction
#         fig, ax = plt.subplots(figsize=(width, height))
#         im = ax.imshow(grid, cmap=cmap, norm=norm, origin='upper', aspect=aspect_correction)
        
#         # Add colorbar
#         sm = ScalarMappable(cmap=cmap, norm=norm)
#         sm._A = []
#         cbar = fig.colorbar(sm, ax=ax, shrink=0.35)
#         cbar.ax.tick_params(labelsize=14)
        
#         ax.axis('off')
#         plt.tight_layout()
        
#         # Save if filename provided
#         if filename is not None:
#             base, ext = os.path.splitext(filename)
#             if use_pca:
#                 component_filename = f"{base}_latent_component_{comp_idx}{ext}"
#             else:
#                 component_filename = f"{base}_miss_feature_column_{feature_column}{ext}"
#             plt.savefig(component_filename, dpi=150, bbox_inches='tight')
#             print(f"Figure saved to {component_filename}")
        
#         plt.show()      
        

def spatial_train_test_split_radius(
    graph: nx.Graph,
    init_frac: float,
    levels: int,
    buffer: int = 0,
    radius: int = 1,
) -> tuple[list, list, list]:
    """Split restricted to nodes with max neighbors."""
    LOGGER.debug(
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
    rng = np.random
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
    LOGGER.debug(
        f"Length of training, tuning and buffer: {len(training_nodes)} and {len(tuning_nodes)} and {len(buffer_nodes)}"
    )
    return training_nodes, tuning_nodes, buffer_nodes