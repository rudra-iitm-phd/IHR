#!/usr/bin/env python
"""
visualize_metric_space.py

Visualises the learned state metric and state-action metric against the SAC
critic, using N (default 4000) on-policy states.

Figures (written to --out-dir)
------------------------------
1a) state_space_Q.png
    2D MDS embedding of the STATE metric space, D_s = max(g(s,x), g(x,s)),
    contoured with Q(s, pi(s)).
1b) state_space_sa_anchor.png
    The SAME state-metric embedding, contoured with the STATE-ACTION distance
    from an anchor x:  max( d_sa((s,pi(s)),(x,pi(x))), d_sa((x,pi(x)),(s,pi(s))) ).
    Thin white lines are level sets of |Q(s,pi(s)) - Q(x,pi(x))|.
2)  sa_space_Q.png
    2D MDS embedding of the STATE-ACTION metric space on pairs (s, pi(s)),
    D_sa = max(d_sa(u,v), d_sa(v,u)), contoured with Q(s, pi(s)).
+)  embedding_diagnostics.png
    Shepard diagrams for both embeddings (how faithful the 2D layouts are).

Also summary.json and data.npz (all matrices, embeddings, Q values).

Pipeline
--------
  1. Restore actor, critic, state metric, state-action metric from the joblib
     pure-dict checkpoints written by utils.logger.nn_model_save
     (<run_dir>/nn_model_save/<prefix>_model_<step>.pt).
  2. Roll the policy out (num_envs x horizon states), subsample N.
  3. Q(s, pi(s)) = min(Q1, Q2)(s, tanh(loc))   [or E_a~pi with --q-mode expected]
  4. Pairwise metric matrices G[i, j] = net(u_i, u_j) (one direction only;
     the reverse direction is G.T), symmetrised with max, embedded with MDS.
  5. Filled contours via a Delaunay triangulation of the embedded points, with
     triangles masked by LOCAL point density, so gaps between clusters are not
     painted but sparse regions still get contours.

Usage (run from the repo root; all flags are in utils.utils.metric_vis_args)
-----
  python visualize_metric_space.py --run-dir <sac run> --inspect
  python visualize_metric_space.py \
      --run-dir <sac run> [--metric-run-dir <run with the metric nets>] \
      --state-metric-prefix <prefix> --sa-metric-prefix <prefix> \
      --out-dir metric_viz/<task>

Saving the observation normaliser (not yet done by the training scripts):
next to every logger.nn_model_save(...) call add

    joblib.dump(
        {"mean": np.asarray(obs_normalizer.mean),
         "var": np.asarray(obs_normalizer.var),
         "count": np.asarray(obs_normalizer.count)},
        osp.join(args.log_dir, "nn_model_save", f"obs_normalizer_model_{steps}.pt"),
    )
"""

import json
import os
import os.path as osp
import re
import time
from typing import Mapping

import jax
import jax.numpy as jnp
import joblib
import matplotlib
import numpy as np
from flax import nnx

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402
from mujoco_playground import registry  # noqa: E402
from scipy.linalg import eigh  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from utils.acting import wrap_env_for_training  # noqa: E402
from utils.algo_metric import (
    EnsembleStateActionMetric,
    EnsembleStateMetric,
    MinStateActiontoStateMetric,
)
from utils.algo_models import EnsembleCritic, SACGaussianActor  # noqa: E402
from utils.buffer import RunningMeanStd  # noqa: E402
from utils.utils import metric_vis_args  # noqa: E402

# sac_args() defaults, used when neither a flag nor config.json gives a value
SAC_DEFAULTS = dict(
    task="HumanoidRun", hidden_size=256, episode_length=1000, reward_scaling=1.0
)


# ===========================================================================
# Metric networks (identical to the training definitions; attribute names
# must match the checkpoints). Import them instead if they live in utils.
# ===========================================================================


# ===========================================================================
# Run directories, checkpoints, config
# ===========================================================================
CKPT_RE = re.compile(r"^(?P<prefix>.+)_model_(?P<itr>\d+)\.pt$")


def list_checkpoints(run_dir):
    """{prefix: {step: path}} for <run_dir>/nn_model_save/<prefix>_model_<step>.pt"""
    d = osp.join(run_dir, "nn_model_save")
    out = {}
    if not osp.isdir(d):
        return out
    for f in os.listdir(d):
        m = CKPT_RE.match(f)
        if m:
            out.setdefault(m["prefix"], {})[int(m["itr"])] = osp.join(d, f)
    return out


def resolve_ckpt(explicit, run_dir, prefix, step, what, required=True):
    if explicit:
        return explicit
    if run_dir is None:
        if required:
            raise ValueError(
                f"{what}: give --{what.replace('_', '-')}-path or --run-dir"
            )
        return None
    ckpts = list_checkpoints(run_dir)
    if prefix not in ckpts:
        if required:
            raise FileNotFoundError(
                f"{what}: no '{prefix}_model_*.pt' in {run_dir}/nn_model_save. "
                f"Available prefixes: {sorted(ckpts)}"
            )
        return None
    steps = ckpts[prefix]
    s = max(steps) if step is None else step
    if s not in steps:
        if required:
            raise FileNotFoundError(
                f"{what}: step {s} not found for '{prefix}'. Steps: {sorted(steps)}"
            )
        return None
    return steps[s]


def step_of(path):
    m = CKPT_RE.match(osp.basename(path)) if path else None
    return int(m["itr"]) if m else None


def load_config(run_dir):
    if run_dir is None:
        return {}
    p = osp.join(run_dir, "config.json")
    if not osp.isfile(p):
        return {}
    with open(p) as f:
        return json.load(f)


# ===========================================================================
# Loading (mirrors utils.logger.Logger.nn_model_load, plus a shape check)
# ===========================================================================
def _flatten_shapes(d, prefix=()):
    out = {}
    for k, v in d.items():
        if isinstance(v, Mapping):
            out.update(_flatten_shapes(v, prefix + (str(k),)))
        else:
            out[prefix + (str(k),)] = tuple(np.shape(v))
    return out


def load_nnx(model, path, name):
    restored = joblib.load(path)
    graphdef, state = nnx.split(model)

    want = _flatten_shapes(nnx.to_pure_dict(state))
    have = _flatten_shapes(restored)
    missing = sorted(set(want) - set(have))
    extra = sorted(set(have) - set(want))
    bad = [
        (k, want[k], have[k])
        for k in sorted(set(want) & set(have))
        if want[k] != have[k]
    ]
    if missing or extra or bad:
        lines = [f"[{name}] checkpoint {path} does not match the model:"]
        lines += [f"  missing in ckpt : {'/'.join(k)} {want[k]}" for k in missing[:10]]
        lines += [f"  unexpected      : {'/'.join(k)} {have[k]}" for k in extra[:10]]
        lines += [
            f"  shape mismatch  : {'/'.join(k)} model {w} vs ckpt {h}"
            for k, w, h in bad[:10]
        ]
        lines.append("  -> check hidden sizes and that the class definition matches.")
        raise ValueError("\n".join(lines))

    nnx.replace_by_pure_dict(state, restored)
    model = nnx.merge(graphdef, state)
    n = sum(int(np.prod(s)) for s in want.values())
    print(f"[load] {name:<14} <- {path}  ({n:,} params)")
    return model


def load_normalizer(path):
    obj = joblib.load(path)
    if isinstance(obj, RunningMeanStd):
        return jax.tree_util.tree_map(jnp.asarray, obj)
    if isinstance(obj, Mapping) and "mean" in obj and "var" in obj:
        return RunningMeanStd(
            mean=jnp.asarray(obj["mean"], jnp.float32),
            var=jnp.asarray(obj["var"], jnp.float32),
            count=jnp.asarray(obj.get("count", 1.0), jnp.float32),
        )
    raise TypeError(f"Cannot build RunningMeanStd from {type(obj)} ({path})")


def inspect(run_dirs, paths):
    for rd in run_dirs:
        if rd:
            print(f"\n=== {rd}/nn_model_save ===")
            for prefix, steps in sorted(list_checkpoints(rd).items()):
                s = sorted(steps)
                print(f"  {prefix:<22} steps: {s}")
    for name, p in paths.items():
        if not p:
            continue
        obj = joblib.load(p)
        print(f"\n--- {name}: {p} ---")
        if isinstance(obj, Mapping):
            for k, s in _flatten_shapes(obj).items():
                print(f"  {'/'.join(k)}: {s}")
        else:
            print(f"  {type(obj)}")


# ===========================================================================
# Rollouts and Q values
# ===========================================================================
def make_rollout(env, num_envs, horizon, deterministic):
    @nnx.jit
    def rollout(actor, obs_normalizer, key):
        key, reset_key = jax.random.split(key)
        state = env.reset(jax.random.split(reset_key, num_envs))

        def body(carry, _):
            state, k = carry
            k, act_key = jax.random.split(k)
            nobs = obs_normalizer.normalize(state.obs)
            if deterministic:
                action = actor.mean_action(nobs)
            else:
                action, _ = actor.sample(nobs, act_key)
            nstate = env.step(state, action)
            return (nstate, k), (state.obs, nstate.reward)

        _, (obs, rew) = jax.lax.scan(body, (state, key), None, length=horizon)
        return obs, rew  # (horizon, num_envs, obs_dim), (horizon, num_envs)

    return rollout


def make_q_fn(mode, num_samples):
    """'mean':     min(Q1,Q2)(s, tanh(loc))
       'expected': E_{a~pi}[min(Q1,Q2)(s, a)]
    Also returns pi(s) = tanh(loc), used as the action in the SA metric."""

    @nnx.jit
    def q_fn(actor, critic, nobs, key):
        mean_a = actor.mean_action(nobs)
        if mode == "mean":
            q1, q2 = critic(jnp.concatenate([nobs, mean_a], axis=-1))
            return jnp.minimum(q1, q2), mean_a
        qs = []
        for kk in jax.random.split(key, num_samples):
            a, _ = actor.sample(nobs, kk)
            q1, q2 = critic(jnp.concatenate([nobs, a], axis=-1))
            qs.append(jnp.minimum(q1, q2))
        return jnp.mean(jnp.stack(qs), axis=0), mean_a

    return q_fn


# ===========================================================================
# Pairwise metric matrices
# ===========================================================================
def pairwise_forward(net, X, block):
    """G[i, j] = net(x_i, x_j) for a ONE-directional metric net.
    The reverse direction net(x_j, x_i) is simply G.T, so the ensembles'
    second output is never needed (halves the cost)."""
    X = jnp.asarray(X, jnp.float32)
    N, dim = X.shape

    @nnx.jit
    def one_block(net, xb, X):
        b = xb.shape[0]
        a = jnp.repeat(xb, X.shape[0], axis=0)
        c = jnp.tile(X, (b, 1))
        return net(a, c).reshape(b, -1)

    G = np.zeros((N, N), np.float32)
    t0 = time.time()
    for s in range(0, N, block):
        xb = X[s : s + block]
        nb = xb.shape[0]
        if nb < block:  # pad so the jitted shape never changes
            xb = jnp.concatenate([xb, jnp.zeros((block - nb, dim), X.dtype)], 0)
        G[s : s + nb] = np.asarray(one_block(net, xb, X))[:nb]
    print(f"   {N}x{N} = {N * N:,} evaluations in {time.time() - t0:.1f}s")
    return G


def symmetrize(G, mode):
    if mode == "max":
        D = np.maximum(G, G.T)
    elif mode == "mean":
        D = 0.5 * (G + G.T)
    elif mode == "min":
        D = np.minimum(G, G.T)
    else:
        raise ValueError(mode)
    D = D.astype(np.float64)
    np.fill_diagonal(D, 0.0)
    return D


def metric_report(tag, G, D, rng):
    N = len(G)
    off = ~np.eye(N, dtype=bool)
    asym = float(np.mean(np.abs(G - G.T)[off]) / np.mean(0.5 * (G + G.T)[off]))
    i, j, k = rng.integers(0, N, (3, 200_000))
    ok = (i != j) & (j != k) & (i != k)
    tri = float(
        np.mean(D[i[ok], k[ok]] > D[i[ok], j[ok]] + D[j[ok], k[ok]] + 1e-6 * D.mean())
    )
    out = dict(
        self_distance_median=float(np.median(np.diag(G))),
        offdiag_median=float(np.median(G[off])),
        offdiag_p05=float(np.quantile(G[off], 0.05)),
        offdiag_p95=float(np.quantile(G[off], 0.95)),
        relative_asymmetry=asym,
        triangle_violation_rate=tri,
    )
    print(
        f"[{tag}] d(s,s) median {out['self_distance_median']:.4g} | off-diag "
        f"median {out['offdiag_median']:.4g} (5-95%: {out['offdiag_p05']:.4g}"
        f"-{out['offdiag_p95']:.4g}) | asymmetry {asym:.3f} | "
        f"triangle violations {tri:.2%}"
    )
    return out


def pairs_vs_dq(D, Q, rng, n=500_000):
    """Spearman between metric distance and |dQ| over random pairs."""
    N = len(Q)
    i, j = rng.integers(0, N, (2, n))
    ok = i != j
    return float(spearmanr(D[i[ok], j[ok]], np.abs(Q[i[ok]] - Q[j[ok]])).correlation)


# ===========================================================================
# 2D embedding
# ===========================================================================
def classical_mds(D):
    """Top-2 classical MDS; double-centring done in O(N^2) without forming J."""
    n = len(D)
    D2 = D**2
    r = D2.mean(axis=1)
    B = -0.5 * (D2 - r[:, None] - r[None, :] + r.mean())
    w, V = eigh(B, subset_by_index=[n - 2, n - 1])
    order = np.argsort(w)[::-1]
    return V[:, order] * np.sqrt(np.clip(w[order], 0.0, None))


def embed_2d(D, method, seed):
    t0 = time.time()
    Y = classical_mds(D)
    if method == "mds":
        try:
            from sklearn.manifold import smacof

            Y = np.asarray(
                smacof(
                    D, n_components=2, init=Y, n_init=1, max_iter=300, random_state=seed
                )[0]
            )
        except Exception as e:  # noqa: BLE001
            print(f"   SMACOF failed ({e!r}); keeping classical MDS")
    elif method == "tsne":
        from sklearn.manifold import TSNE

        Y = TSNE(
            metric="precomputed", init="random", perplexity=30, random_state=seed
        ).fit_transform(D)
    elif method == "umap":
        import umap

        Y = umap.UMAP(metric="precomputed", random_state=seed).fit_transform(D)
    elif method != "classical":
        raise ValueError(method)
    print(f"   '{method}' embedding of {len(D)} points in {time.time() - t0:.1f}s")
    return Y


def shepard_stats(D, Y):
    iu = np.triu_indices(len(D), 1)
    d_net = D[iu]
    d_2d = pdist(Y)  # same pair order as triu_indices
    stress = float(np.sqrt(np.sum((d_net - d_2d) ** 2) / np.sum(d_net**2)))
    return d_net, d_2d, stress


# ===========================================================================
# Contouring
# ===========================================================================
def masked_triangulation(Y, k, factor, rng):
    """Delaunay triangulation of the embedded points, masked by LOCAL density.

    local[i] = distance from point i to its k-th nearest neighbour. An edge
    (i, j) is too long if it exceeds factor * min(local[i], local[j]), i.e. it
    is judged by the DENSER endpoint. A triangle with any too-long edge is
    masked. Edges are also capped globally at factor x the 95th percentile of
    the local spacings, so very sparse regions show as dots, not big patches.
    Result: nothing bridges the empty gap between two clusters."""
    span = np.ptp(Y, axis=0).max()
    Yj = Y + rng.normal(scale=1e-7 * span, size=Y.shape)  # break exact duplicates
    tri = Triangulation(Yj[:, 0], Yj[:, 1])
    local = cKDTree(Yj).query(Yj, k=min(k + 1, len(Yj)))[0][:, -1]
    cap = factor * np.quantile(local, 0.95)  # global cap: no huge triangles
    t = tri.triangles
    bad = np.zeros(len(t), dtype=bool)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        ia, ib = t[:, a], t[:, b]
        edge = np.linalg.norm(Yj[ia] - Yj[ib], axis=1)
        bad |= edge > np.minimum(factor * np.minimum(local[ia], local[ib]), cap)
    tri.set_mask(bad)
    return tri


def metric_smooth(values, D, k):
    """Average each value over the point itself and its k nearest neighbours
    under the NETWORK metric D (not the 2D layout). k <= 0 returns the raw
    values."""
    if k <= 0:
        return values
    nn = np.argpartition(D, kth=min(k, len(D) - 1), axis=1)[:, : k + 1]
    return values[nn].mean(axis=1)


def make_levels(v, n, scale):
    """Contour levels.
    linear   : evenly spaced between min and max.
    quantile : every band holds the same number of points (detail where the
               data is dense, but rare values share one band).
    hybrid   : union of both, so the rare low-Q transient AND the fine
               differences inside the dense high-Q cluster get colours."""
    lin = np.linspace(v.min(), v.max(), n + 1)
    qnt = np.quantile(v, np.linspace(0.0, 1.0, n + 1))
    if scale == "quantile":
        lv = np.unique(qnt)
    elif scale == "hybrid":
        lin = np.linspace(v.min(), v.max(), n // 2 + 1)
        qnt = np.quantile(v, np.linspace(0.0, 1.0, n // 2 + 1))
        lv = np.unique(np.round(np.r_[lin, qnt], 10))
    else:
        lv = lin
    if len(lv) < 3:
        lv = np.linspace(v.min(), v.max() + 1e-9, n + 1)
    return lv


def contour_panel(fig, ax, tri, Y, values, cmap, label, args, point_size=3.0):
    levels = make_levels(values, args.levels, args.color_scale)
    cm = plt.get_cmap(cmap)
    norm = BoundaryNorm(levels, cm.N, clip=True)
    cf = ax.tricontourf(tri, values, levels=levels, cmap=cm, norm=norm)
    ax.tricontour(
        tri, values, levels=levels[::2], colors="k", linewidths=0.3, alpha=0.35
    )
    ax.scatter(
        Y[:, 0],
        Y[:, 1],
        c=values,
        cmap=cm,
        norm=norm,
        s=point_size,
        linewidths=0,
        alpha=0.9,
    )
    cb = fig.colorbar(cf, ax=ax, label=label)
    step = max(1, (len(levels) - 1) // 6)
    cb.set_ticks(levels[::step])
    cb.ax.set_yticklabels([f"{x:.4g}" for x in levels[::step]])
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("embedding dim 1")
    ax.set_ylabel("embedding dim 2")
    return levels


def pick_anchors(Q, args):
    if args.anchor_index:
        return [int(i) for i in args.anchor_index]
    return [
        int(np.argmin(np.abs(Q - np.quantile(Q, q)))) for q in args.anchor_quantiles
    ]


# ===========================================================================
# Main
# ===========================================================================
def main():
    args, _ = metric_vis_args()
    jax.config.update("jax_default_device", jax.devices(args.device)[args.device_id])

    metric_run_dir = args.metric_run_dir or args.run_dir
    cfg = load_config(args.run_dir)
    mcfg = load_config(metric_run_dir)

    # ---- resolve checkpoints -------------------------------------------------
    actor_path = resolve_ckpt(
        args.actor_path, args.run_dir, args.actor_prefix, args.step, "actor"
    )
    step = step_of(actor_path) if args.step is None else args.step
    critic_path = resolve_ckpt(
        args.critic_path, args.run_dir, args.critic_prefix, step, "critic"
    )
    sm_path = resolve_ckpt(
        args.state_metric_path,
        metric_run_dir,
        args.state_metric_prefix,
        args.metric_step,
        "state_metric",
    )
    mstep = step_of(sm_path) if args.metric_step is None else args.metric_step
    sa_path = resolve_ckpt(
        args.sa_metric_path, metric_run_dir, args.sa_metric_prefix, mstep, "sa_metric"
    )
    norm_path = resolve_ckpt(
        args.normalizer_path,
        args.run_dir,
        args.normalizer_prefix,
        step,
        "normalizer",
        required=False,
    )
    mnorm_path = resolve_ckpt(
        args.metric_normalizer_path,
        metric_run_dir,
        args.normalizer_prefix,
        mstep,
        "metric_normalizer",
        required=False,
    )

    if args.inspect:
        inspect(
            {args.run_dir, metric_run_dir},
            dict(
                actor=actor_path,
                critic=critic_path,
                state_metric=sm_path,
                sa_metric=sa_path,
            ),
        )
        print(f"\nnormaliser        : {norm_path}\nmetric normaliser : {mnorm_path}")
        return

    # ---- hyper-parameters: flag > config.json > sac_args default ----------
    def pick(flag, key, c=cfg):
        return flag if flag is not None else c.get(key, SAC_DEFAULTS.get(key))

    task = pick(args.task, "task")
    hidden = int(pick(args.hidden_size, "hidden_size"))
    m_hidden = int(
        args.metric_hidden_size
        or mcfg.get("metric_hidden_size")
        or pick(None, "hidden_size", mcfg)
    )
    episode_length = int(pick(args.episode_length, "episode_length"))
    reward_scaling = float(pick(args.reward_scaling, "reward_scaling"))
    horizon = args.horizon or episode_length
    print(
        f"[cfg] task={task} hidden={hidden} metric_hidden={m_hidden} "
        f"episode_length={episode_length} reward_scaling={reward_scaling}"
    )

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    rngs = nnx.Rngs(args.seed)

    # ---- environment -------------------------------------------------------
    env = wrap_env_for_training(
        registry.load(task, config_overrides={"impl": "jax"}),
        episode_length=episode_length,
        full_reset=False,
    )
    obs_dim, act_dim = env.observation_size, env.action_size
    if not isinstance(obs_dim, int):
        raise ValueError(f"dict observations not supported: {obs_dim}")
    print(f"[env] obs_dim={obs_dim} act_dim={act_dim}")

    # ---- 1) models ---------------------------------------------------------
    actor = load_nnx(
        SACGaussianActor(
            rngs=rngs, obs_dim=obs_dim, act_dim=act_dim, hidden_size=hidden
        ),
        actor_path,
        "actor",
    )
    critic = load_nnx(
        EnsembleCritic(rngs=rngs, obs_dim=obs_dim, act_dim=act_dim, hidden_size=hidden),
        critic_path,
        "critic",
    )
    s_metric = load_nnx(
        EnsembleStateMetric(rngs, obs_dim, m_hidden), sm_path, "state_metric"
    )
    sa_metric = load_nnx(
        EnsembleStateActionMetric(rngs, obs_dim, act_dim, m_hidden),
        sa_path,
        "sa_metric",
    )

    # ---- 2) normalisers + experience ---------------------------------------
    rollout = make_rollout(
        env, args.num_envs, horizon, deterministic=not args.stochastic
    )
    if norm_path:
        normalizer = load_normalizer(norm_path)
        print(f"[norm] policy normaliser <- {norm_path}")
    else:
        print(
            "[norm] WARNING: no saved normaliser for the actor/critic; "
            "re-estimating from policy rollouts (Q values are approximate)."
        )
        normalizer = RunningMeanStd.init((obs_dim,))
        for i in range(args.norm_passes):
            key, sub = jax.random.split(key)
            obs_pass, _ = rollout(actor, normalizer, sub)
            normalizer = RunningMeanStd.init((obs_dim,)).update(
                obs_pass.reshape(-1, obs_dim)
            )
            print(f"[norm] pass {i + 1}/{args.norm_passes} done")
    if mnorm_path:
        metric_normalizer = load_normalizer(mnorm_path)
        print(f"[norm] metric normaliser <- {mnorm_path}")
    else:
        metric_normalizer = normalizer
        if metric_run_dir != args.run_dir and args.metric_input == "normalized":
            print(
                "[norm] WARNING: metric nets come from another run but no metric "
                "normaliser was found; using the policy normaliser."
            )

    key, sub = jax.random.split(key)
    print(f"[rollout] {args.num_envs} envs x {horizon} steps (first call compiles)")
    obs_all, rew_all = rollout(actor, normalizer, sub)
    obs_all = np.asarray(obs_all).reshape(-1, obs_dim)
    ret = float(np.asarray(rew_all).sum(0).mean())
    obs_all = obs_all[np.all(np.isfinite(obs_all), axis=1)]
    N = min(args.num_states, len(obs_all))
    if N < args.num_states:
        print(
            f"[rollout] only {len(obs_all)} states collected; using N={N}. "
            "Increase --num-envs or --horizon for more."
        )
    idx = rng.choice(len(obs_all), size=N, replace=False)
    obs = obs_all[idx]
    print(
        f"[rollout] {len(obs_all)} states collected, mean return per env "
        f"{ret:.1f}; kept N = {N}"
    )

    nobs = np.asarray(normalizer.normalize(jnp.asarray(obs)))
    key, qkey = jax.random.split(key)
    Q, act = make_q_fn(args.q_mode, args.q_samples)(
        actor, critic, jnp.asarray(nobs), qkey
    )
    Q, act = np.asarray(Q, np.float64), np.asarray(act)
    if args.q_units == "reward":
        Q = Q / reward_scaling
        qlabel = r"$Q(s,\pi(s))$ / reward_scaling"
    else:
        qlabel = r"$Q(s,\pi(s))$"
    print(
        f"[critic] Q: min {Q.min():.4g}  median {np.median(Q):.4g}  max {Q.max():.4g}"
    )

    feats_s = (
        np.asarray(metric_normalizer.normalize(jnp.asarray(obs)))
        if args.metric_input == "normalized"
        else obs
    )
    feats_sa = np.concatenate([feats_s, act], axis=-1)  # (s, pi(s))

    # ---- metric matrices ------------------------------------------------------
    print("[metric] state metric g(s, x) on all pairs")
    G_s = pairwise_forward(s_metric.g, feats_s, args.block)
    D_s = symmetrize(G_s, args.symmetrize)
    rep_s = metric_report("state", G_s, D_s, rng)

    print("[metric] state-action metric d((s,pi(s)), (x,pi(x))) on all pairs")
    G_sa = pairwise_forward(sa_metric.d, feats_sa, args.block)
    D_sa = symmetrize(G_sa, args.symmetrize)
    rep_sa = metric_report("state_action", G_sa, D_sa, rng)
    del G_s, G_sa

    rho_s = pairs_vs_dq(D_s, Q, rng)
    rho_sa = pairs_vs_dq(D_sa, Q, rng)
    print(f"[pairs] Spearman(D, |dQ|): state {rho_s:.3f} | state-action {rho_sa:.3f}")

    # ---- embeddings -----------------------------------------------------------
    print("[embed] state metric space")
    Y_s = embed_2d(D_s, args.embed, args.seed)
    print("[embed] state-action metric space")
    Y_sa = embed_2d(D_sa, args.embed, args.seed)
    dn_s, d2_s, stress_s = shepard_stats(D_s, Y_s)
    dn_sa, d2_sa, stress_sa = shepard_stats(D_sa, Y_sa)
    print(f"[embed] stress-1: state {stress_s:.3f} | state-action {stress_sa:.3f}")

    tri_s = masked_triangulation(Y_s, args.mask_k, args.mask_factor, rng)
    tri_sa = masked_triangulation(Y_sa, args.mask_k, args.mask_factor, rng)
    sym = f"{args.symmetrize}-symmetrised"
    sk = args.smooth_k
    smooth_note = f"  [Q averaged over {sk} metric-NN]" if sk > 0 else ""
    Q_on_s = metric_smooth(Q, D_s, sk)  # colour field on the state space
    Q_on_sa = metric_smooth(Q, D_sa, sk)  # colour field on the SA space

    # ===== 1a) state metric space, contoured with Q =========================
    fig, ax = plt.subplots(figsize=(9, 7))
    contour_panel(fig, ax, tri_s, Y_s, Q_on_s, "viridis", qlabel, args)
    ax.set_title(
        f"State metric space ({sym} g, {args.embed.upper()}, N={N})\n"
        f"contours: {qlabel}{smooth_note}   |   "
        f"Spearman(D_s, |dQ|) = {rho_s:.3f}"
    )
    fig.tight_layout()
    fig.savefig(
        osp.join(args.out_dir, "state_space_Q.png"), dpi=170, bbox_inches="tight"
    )
    plt.close(fig)

    # ===== 1b) state metric space, contoured with SA distance to anchor ====
    anchors = pick_anchors(Q, args)
    anchor_info = []
    fig, axes = plt.subplots(
        1, len(anchors), figsize=(9 * len(anchors), 7), squeeze=False
    )
    for ax, a in zip(axes[0], anchors):
        d_anchor = D_sa[a].copy()  # max(d_sa(u_a, u_i), d_sa(u_i, u_a)); 0 at anchor
        dq_a = np.abs(Q - Q[a])
        contour_panel(
            fig,
            ax,
            tri_s,
            Y_s,
            d_anchor,
            "magma",
            r"$d_{sa}((s,\pi(s)),(x,\pi(x)))$  " + f"({args.symmetrize})",
            args,
        )
        if args.dq_overlay:
            dq_field = metric_smooth(dq_a, D_s, sk)
            others = dq_field[np.arange(N) != a]
            dl = np.unique(np.quantile(others, [0.1, 0.5]))
            if len(dl) > 1:
                cs = ax.tricontour(
                    tri_s,
                    dq_field,
                    levels=dl,
                    colors="white",
                    linewidths=0.8,
                    linestyles="--",
                )
                ax.clabel(cs, fmt=lambda v: f"|dQ|={v:.2g}", fontsize=7)
        ax.scatter(
            Y_s[a, 0],
            Y_s[a, 1],
            marker="*",
            s=380,
            c="cyan",
            edgecolors="k",
            linewidths=1.0,
            zorder=5,
            label="anchor x",
        )
        mask = np.arange(N) != a
        rho_a = float(spearmanr(d_anchor[mask], dq_a[mask]).correlation)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.7)
        ax.set_title(
            f"State metric space, contoured with state-action distance "
            f"to anchor x (idx {a})\nQ(x,π(x)) = {Q[a]:.4g}   |   "
            f"Spearman(d_sa(x,·), |dQ|) = {rho_a:.3f}"
            + ("   |   dashed: |dQ| to anchor" if args.dq_overlay else "")
        )
        anchor_info.append(dict(index=a, Q=float(Q[a]), spearman_dsa_dq=rho_a))
    fig.tight_layout()
    fig.savefig(
        osp.join(args.out_dir, "state_space_sa_anchor.png"),
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ===== 2) state-action metric space, contoured with Q ===================
    fig, ax = plt.subplots(figsize=(9, 7))
    contour_panel(fig, ax, tri_sa, Y_sa, Q_on_sa, "viridis", qlabel, args)
    for a in anchors:
        ax.scatter(
            Y_sa[a, 0],
            Y_sa[a, 1],
            marker="*",
            s=300,
            c="cyan",
            edgecolors="k",
            linewidths=1.0,
            zorder=5,
        )
    ax.set_title(
        f"State-action metric space on (s, π(s)) ({sym} d_sa, "
        f"{args.embed.upper()}, N={N})\ncontours: {qlabel}{smooth_note}"
        f"   |   Spearman(D_sa, |dQ|) = {rho_sa:.3f}   (★ = anchor)"
    )
    fig.tight_layout()
    fig.savefig(osp.join(args.out_dir, "sa_space_Q.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ===== diagnostics: Shepard diagrams ===================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (name, dn, d2, st) in zip(
        axes,
        [("state", dn_s, d2_s, stress_s), ("state-action", dn_sa, d2_sa, stress_sa)],
    ):
        sub = rng.choice(len(dn), size=min(len(dn), 400_000), replace=False)
        ax.hexbin(dn[sub], d2[sub], gridsize=70, bins="log", cmap="Blues", mincnt=1)
        m = max(dn.max(), d2.max())
        ax.plot([0, m], [0, m], "r--", lw=1, label="perfect")
        rho = spearmanr(dn[sub], d2[sub]).correlation
        ax.set_title(f"{name}: stress-1 = {st:.3f}, Spearman = {rho:.3f}")
        ax.set_xlabel("network distance (symmetrised)")
        ax.set_ylabel("2D embedding distance")
        ax.legend(loc="upper left")
    fig.suptitle("How faithful are the 2D layouts? (stress < 0.1 good, > 0.2 poor)")
    fig.tight_layout()
    fig.savefig(
        osp.join(args.out_dir, "embedding_diagnostics.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ---- save -----------------------------------------------------------------
    summary = dict(
        meta=dict(
            task=task,
            N=N,
            actor=actor_path,
            critic=critic_path,
            state_metric=sm_path,
            sa_metric=sa_path,
            normalizer=norm_path,
            metric_normalizer=mnorm_path,
            reward_scaling=reward_scaling,
            rollout_mean_return=ret,
            symmetrize=args.symmetrize,
            embed=args.embed,
            q_mode=args.q_mode,
            q_units=args.q_units,
            smooth_k=sk,
        ),
        Q=dict(min=float(Q.min()), median=float(np.median(Q)), max=float(Q.max())),
        state=dict(**rep_s, spearman_D_dq=rho_s, stress=stress_s),
        state_action=dict(**rep_sa, spearman_D_dq=rho_sa, stress=stress_sa),
        anchors=anchor_info,
        args=vars(args),
    )
    with open(osp.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        osp.join(args.out_dir, "data.npz"),
        obs=obs,
        nobs=nobs,
        Q=Q,
        act=act,
        D_s=D_s.astype(np.float32),
        D_sa=D_sa.astype(np.float32),
        Y_s=Y_s,
        Y_sa=Y_sa,
        anchors=np.array(anchors),
    )
    print(
        f"\n[done] state_space_Q.png, state_space_sa_anchor.png, sa_space_Q.png, "
        f"embedding_diagnostics.png, summary.json, data.npz -> {args.out_dir}/"
    )


if __name__ == "__main__":
    main()
