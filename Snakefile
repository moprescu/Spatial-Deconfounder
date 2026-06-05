# This file is used to generate all the baselines for the paper.

import re

from omegaconf import OmegaConf

conda: "benchmarks/conda.yaml"


# == Load configs ==
if len(config) == 0:
    raise Exception(
        "No config file passed to snakemake."
        " Use flag --configfile benchmarks/conf/pipeline.yaml"
    )


# Parse deconfounder variant algo names into Hydra overrides that target a
# shared template in conf/algo/. Supported patterns (nbr segment optional):
#   deconfounder_r{R}[_nbr{N}]_spatialplus_{D}[-CopyX]
#   deconfounder_r{R}[_nbr{N}]_unet_{D}
#   deconfounder_r{R}[_nbr{N}]_temporal_unet_{D}
_DECONFOUNDER_SPATIALPLUS_RE = re.compile(
    r"^deconfounder_r(\d+)(?:_nbr(\d+))?_spatialplus_(\d+)(?:-Copy\d+)?$"
)
_DECONFOUNDER_UNET_RE = re.compile(
    r"^deconfounder_r(\d+)(?:_nbr(\d+))?_unet_(\d+)$"
)
_DECONFOUNDER_TEMPORAL_UNET_RE = re.compile(
    r"^deconfounder_r(\d+)(?:_nbr(\d+))?_temporal_unet_(\d+)$"
)

def algo_hydra_overrides(algo):
    m = _DECONFOUNDER_SPATIALPLUS_RE.match(algo)
    if m:
        r, n, c2 = m.groups()
        parts = [
            "algo=deconfounder_spatialplus",
            f"algo.name={algo}",
            f"algo.method.radius={r}",
            f"algo.method.encoder_conv2={c2}",
        ]
        if n is not None:
            parts.append(f"algo.method.nbr_treatment_radius={n}")
        return " ".join(parts)

    m = _DECONFOUNDER_UNET_RE.match(algo)
    if m:
        r, n, c2 = m.groups()
        # unet_base_chan widens to include 64 when encoder_conv2 == 64.
        chans = "[16,32,64]" if int(c2) == 64 else "[16,32]"
        parts = [
            "algo=deconfounder_unet",
            f"algo.name={algo}",
            f"algo.method.radius={r}",
            f"algo.method.encoder_conv2={c2}",
            f"'algo.tune.param_space.unet_base_chan._args_=[{chans}]'",
        ]
        if n is not None:
            parts.append(f"algo.method.nbr_treatment_radius={n}")
        return " ".join(parts)
    
    m = _DECONFOUNDER_TEMPORAL_UNET_RE.match(algo)
    if m:
        r, n, c2 = m.groups()
        # unet_base_chan widens to include 64 when encoder_conv2 == 64.
        chans = "[16,32,64]" if int(c2) == 64 else "[16,32]"
        parts = [
            "algo=deconfounder_temporal_unet",
            f"algo.name={algo}",
            f"algo.method.radius={r}",
            f"algo.method.encoder_conv2={c2}",
            f"'algo.tune.param_space.unet_base_chan._args_=[{chans}]'",
        ]
        if n is not None:
            parts.append(f"algo.method.nbr_treatment_radius={n}")
        return " ".join(parts)

    return f"algo={algo}"


# make target files
targets = []
for t_type in ("disc", "cont"):
    envs = config["spaceenvs"][t_type] or []
    algos = config["algorithms"].get(t_type) or []
    for env in envs:
        for algo in algos:
            logfile = f"{config['logdir']}/{env}/{algo}.jsonl"
            targets.append(logfile)


# == Define rules ==
rule all:
    input:
        targets,


rule train_spaceenv:
    output:
        config["logdir"] + "/{spaceenv}/{algo}.jsonl",
    threads: config["concurrency"] * config["cpus_per_task"]
    resources:
        mem_mb=config["mem_mb"],
    params:
        concurrency=config["concurrency"],
        overwrite=config["overwrite"],
        algo_overrides=lambda w: algo_hydra_overrides(w.algo),
    log:
        err="logs/{spaceenv}/{algo}.err",
    shell:
        """
        export LD_LIBRARY_PATH=/home/idies/workspace/Temporary/akhot2/scratch/spacedata/lib/python3.10/site-packages/nvidia/cuda/lib:$LD_LIBRARY_PATH
        HYDRA_FULL_ERROR=1 python benchmarks/run.py \
            {params.algo_overrides} \
            spaceenv={wildcards.spaceenv} \
            concurrency={params.concurrency} \
            overwrite={params.overwrite} \
            hydra.run.dir=logs/{wildcards.spaceenv}/{wildcards.algo} \
            2> {log.err}
        """
