import argparse
from distutils.util import strtobool

from flax import struct


def make_static_config_from_dict(name: str, d: dict):
    """
    << copied from claude >>
    Convert a plain Python dict to an immutable Flax struct dataclass.

    Why immutable?
    ──────────────
    jax.jit traces a function once and caches the compiled XLA computation.
    If a Python dict were used as config, JAX would re-trace every time the
    dict *object* changes (even if values stay the same), or worse, it might
    cache a stale compilation if values change silently.

    By making the config a Flax struct with pytree_node=False, JAX treats
    every field as a *static* (compile-time) constant. If a value changes,
    JAX knows to re-compile. If it doesn't change, it reuses the cache.

    Usage:
        Config = make_static_config_from_dict("Config", {"lr": 3e-4})
        cfg = Config()      # instantiate
        cfg.lr              # 3e-4
    """
    annotations = {}
    defaults = {}
    for k, v in d.items():
        annotations[k] = type(v)
        defaults[k] = struct.field(default=v, pytree_node=False)
    cls = type(name, (), {"__annotations__": annotations, **defaults})
    return struct.dataclass(cls)


def sac_args():
    """Parse command-line arguments for the SAC training script."""
    from argparse import ArgumentParser

    p = ArgumentParser(description="SAC on MuJoCo Playground (JAX/NNX)")

    # reproducibility
    p.add_argument("--seed", type=int, default=0)
    # environment
    p.add_argument("--task", type=str, default="HumanoidRun")
    p.add_argument("--episode-length", type=int, default=1000)
    p.add_argument("--num_envs", type=int, default=128)
    p.add_argument("--num-eval-envs", type=int, default=128)
    p.add_argument("--nstep", type=int, default=int(3))
    p.add_argument("--reward_scaling", type=float, default=1.0)
    p.add_argument("--rep_lr_scale", type=float, default=1.5)
    # logging
    # p.add_argument("--experiment", type=str, default="sac")
    p.add_argument("--log-dir", type=str, default="logs")
    p.add_argument("--write-terminal", type=lambda x: bool(strtobool(x)), default=True)
    # compute
    p.add_argument("--device", type=str, default="gpu", help="'gpu' or 'cpu'")
    p.add_argument("--device-id", type=int, default=0)
    # replay buffer
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-replay-size", type=int, default=int(4e6))
    p.add_argument("--warmup-samples", type=int, default=int(8192))
    # training schedule
    p.add_argument("--total-env-steps", type=int, default=int(5e6))
    p.add_argument("--log-freq", type=int, default=int(1e3))
    p.add_argument("--save-freq", type=int, default=int(1e6))
    p.add_argument("--train-per-step", type=int, default=8)
    # SAC hyperparameters
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--update-tau", type=float, default=0.005)
    p.add_argument("--init-temperature", type=float, default=1.0)
    p.add_argument("--max-grad-norm", type=float, default=0.0)
    p.add_argument("--hidden-size", type=int, default=256)
    # eval
    p.add_argument("--eval-episode-freq", type=int, default=10)
    # transfer
    p.add_argument("--target_task", type=str, default="CheetahRun")
    p.add_argument("--transfer_freq", type=int, default=int(1e1))
    p.add_argument("--transfer_steps", type=int, default=int(8))
    p.add_argument("--grad_steps", type=int, default=int(50))
    p.add_argument("--env2_warmup", type=int, default=int(2e3))
    p.add_argument(
        "--vis-freq",
        type=int,
        default=int(1e5),
        help="Steps between t-SNE saves (Task 1). Much larger than log-freq.",
    )
    p.add_argument(
        "--n-vis-frames",
        type=int,
        default=4,
        help="Number of annotated frames on the t-SNE plot (Task 1).",
    )

    return p.parse_args(), {}


def metric_vis_args():
    """Parse command-line arguments for visualize_metric_space.py.

    Separate from sac_args() because sac_args() calls parse_args() itself and
    would reject these flags. Hyper-parameter flags default to None so values
    come from the run's config.json first (written by Logger.save_config from
    sac_args), falling back to sac_args' defaults.
    """
    from argparse import ArgumentParser

    p = ArgumentParser(
        description="State / state-action metric spaces contoured with the SAC critic"
    )

    # reproducibility / compute (same as sac_args)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="gpu", help="'gpu' or 'cpu'")
    p.add_argument("--device-id", type=int, default=0)

    # where the checkpoints are
    p.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="SAC run dir holding config.json and nn_model_save/",
    )
    p.add_argument(
        "--metric-run-dir",
        type=str,
        default=None,
        help="run dir of the metric nets (default: --run-dir)",
    )
    p.add_argument(
        "--step",
        type=int,
        default=None,
        help="actor/critic checkpoint step (default: latest)",
    )
    p.add_argument(
        "--metric-step",
        type=int,
        default=None,
        help="metric checkpoint step (default: latest)",
    )
    p.add_argument("--actor-prefix", type=str, default="actor")
    p.add_argument("--critic-prefix", type=str, default="critic")
    p.add_argument("--state-metric-prefix", type=str, default="state_metric")
    p.add_argument("--sa-metric-prefix", type=str, default="sa_metric")
    p.add_argument("--normalizer-prefix", type=str, default="obs_normalizer")
    p.add_argument("--actor-path", type=str, default=None)
    p.add_argument("--critic-path", type=str, default=None)
    p.add_argument("--state-metric-path", type=str, default=None)
    p.add_argument("--sa-metric-path", type=str, default=None)
    p.add_argument(
        "--normalizer-path",
        type=str,
        default=None,
        help="normaliser the actor/critic were trained with",
    )
    p.add_argument(
        "--metric-normalizer-path",
        type=str,
        default=None,
        help="normaliser the metric nets were trained with",
    )

    # overrides; None -> config.json -> sac_args default
    p.add_argument("--task", type=str, default=None)
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--metric-hidden-size", type=int, default=None)
    p.add_argument("--episode-length", type=int, default=None)
    p.add_argument("--reward-scaling", "--reward_scaling", type=float, default=None)

    # data collection
    p.add_argument("--num-envs", "--num_envs", type=int, default=16)
    p.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="steps per env (default: episode length)",
    )
    p.add_argument(
        "--num-states",
        type=int,
        default=4000,
        help="N states used for the N x N metric matrices",
    )
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="collect with sampled actions instead of tanh(loc)",
    )
    p.add_argument(
        "--norm-passes",
        type=int,
        default=2,
        help="passes used to re-estimate a missing normaliser",
    )

    # what to compare
    p.add_argument(
        "--q-mode",
        type=str,
        choices=["mean", "expected"],
        default="mean",
        help="Q(s, tanh(loc)) or E_{a~pi} Q(s, a)",
    )
    p.add_argument("--q-samples", type=int, default=16)
    p.add_argument(
        "--q-units",
        type=str,
        choices=["reward", "scaled"],
        default="reward",
        help="'reward' divides Q by reward_scaling",
    )
    p.add_argument(
        "--metric-input", type=str, choices=["normalized", "raw"], default="normalized"
    )
    p.add_argument(
        "--symmetrize", type=str, choices=["max", "mean", "min"], default="max"
    )

    # embedding + contours
    p.add_argument(
        "--embed", type=str, choices=["mds", "classical", "tsne", "umap"], default="mds"
    )
    p.add_argument(
        "--anchor-quantiles",
        type=float,
        nargs="+",
        default=[0.5],
        help="anchor(s) = state whose Q is closest to these quantiles",
    )
    p.add_argument(
        "--anchor-index",
        type=int,
        nargs="+",
        default=None,
        help="explicit anchor index/indices into the N states "
        "(overrides --anchor-quantiles)",
    )
    p.add_argument("--levels", type=int, default=20, help="number of contour bands")
    p.add_argument(
        "--color-scale",
        type=str,
        choices=["hybrid", "quantile", "linear"],
        default="hybrid",
        help="'hybrid' = linear + quantile levels; 'quantile' = equal "
        "points per band; 'linear' = evenly spaced",
    )
    p.add_argument(
        "--mask-k",
        type=int,
        default=8,
        help="k-th neighbour distance used as local point spacing",
    )
    p.add_argument(
        "--mask-factor",
        type=float,
        default=3.0,
        help="mask triangles longer than factor x local spacing",
    )
    p.add_argument(
        "--smooth-k",
        type=int,
        default=0,
        help="colour with Q averaged over each state's k nearest "
        "neighbours under the NETWORK metric (0 = raw Q)",
    )
    p.add_argument(
        "--dq-overlay",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="dashed |dQ|-to-anchor lines on the anchor plot",
    )
    p.add_argument(
        "--block",
        type=int,
        default=32,
        help="rows per metric-evaluation batch (block x N pairs)",
    )
    p.add_argument("--out-dir", type=str, default="metric_viz")
    p.add_argument(
        "--inspect",
        action="store_true",
        help="list checkpoints and parameter shapes, then exit",
    )

    return p.parse_args(), {}
