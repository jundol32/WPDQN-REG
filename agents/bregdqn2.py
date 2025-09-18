"""DQN with Lp-Bregman + log-barrier using a 3-way scaling strategy.

Principle:
- INTERNAL (geometry/contraction): limit network outputs and TD via (S, α)
  * out_scale_in = S keeps Q_internal inside (or near) the Lp-ball of radius R
  * AVI-only reward scaling α makes the Bellman map self-mapping in the INTERNAL domain
- EXTERNAL (optimization-only): apply a small symmetric scale s to (Q, Y) **only in the loss**
  * Q_loss = s * Q_internal,  Y_loss = s * Y_internal
  * Use the *constant-free* Bregman data term: h(Q) - <∇h(Y), Q>
  * This keeps magnitudes healthy without p-power explosions
"""
from typing import Sequence
from jaxrl_m.typing import *

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxrl_m.common import TrainState, target_update, nonpytree_field

import flax
import flax.linen as nn


# ===================== Network =====================
class DQNNet(nn.Module):
    hidden_dims: Sequence[int]
    num_actions: int
    # INTERNAL output scaling & head options
    out_scale_in: float = 1.0          # S (INTERNAL range scale)
    use_tanh_head: bool = True
    tanh_temp: float = 1.0
    head_mode: str = "signed"          # "signed" → [-S, S], "unsigned" → [0, S]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = observations
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        logits = nn.Dense(self.num_actions)(x)
        if self.use_tanh_head:
            t = jnp.tanh(logits / self.tanh_temp)
            if self.head_mode == "unsigned":
                q_values_internal = self.out_scale_in * (t + 1.0) * 0.5  # [0, S]
            else:
                q_values_internal = self.out_scale_in * t               # [-S, S]
        else:
            q_values_internal = self.out_scale_in * logits
        return q_values_internal


# ===================== Agent =====================
class WPDQNAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    q_network: TrainState
    target_q_network: TrainState
    config: dict = nonpytree_field()

    @jax.jit
    def update(agent, batch: Batch):

        def loss_fn(q_params):
            # --- 0) shortcuts & params ---
            p = agent.config.get('loss_p', 4.0)
            R = agent.config.get('radius_R', 0.5)
            tau = agent.config.get('barrier_tau', 1e-3)
            eps = 1e-6

            # INTERNAL (geometry) scales
            avi_reward_scale = agent.config.get('avi_reward_scale', 1.0)  # α (TD 전용)
            # EXTERNAL (optimization-only) symmetric loss scale
            loss_scale_s = agent.config.get('loss_scale_s', 10.0)         # s (작고 고정 권장)

            # --- 1) TD target (INTERNAL) -----------------------------------------
            next_q_vals_internal = agent.target_q_network(batch['next_observations'])  # (B, A) INTERNAL
            next_q_internal = jnp.max(next_q_vals_internal, axis=-1)                   # (B,)
            rewards_internal = avi_reward_scale * batch['rewards']                     # α·r  (AVI only)
            target_q_internal = rewards_internal + agent.config['discount'] * batch['masks'] * next_q_internal

            # --- 2) current Q (INTERNAL) ------------------------------------------
            q_vals_internal = agent.q_network(batch['observations'], params=q_params)  # (B, A) INTERNAL

            # --- 3) Double-DQN style vector target (INTERNAL) ---------------------
            q_bar_internal = agent.target_q_network(batch['observations'])             # (B, A) INTERNAL
            actions = batch['actions'].astype('int32')
            a_onehot = jax.nn.one_hot(actions, q_bar_internal.shape[-1])               # (B, A)
            q_bar_sa_internal = jnp.take_along_axis(q_bar_internal, actions[:, None], axis=-1).squeeze(-1)  # (B,)
            delta_internal = target_q_internal - q_bar_sa_internal                     # (B,)
            y_vec_internal = q_bar_internal + a_onehot * delta_internal[:, None]       # (B, A)
            y_vec_internal = jax.lax.stop_gradient(y_vec_internal)
            # ----------------------------------------------------------------------

            # --- 4) EXTERNAL symmetric scale for loss only ------------------------
            q_vals_for_loss = loss_scale_s * q_vals_internal
            y_vec_for_loss = loss_scale_s * y_vec_internal
            # ----------------------------------------------------------------------

            # --- 5) Constant-free Bregman data term on PHYSICAL(loss) scale -------
            # h(x) = (1/p)||x||_p^p ; ∇h(y) = |y|^{p-2} y
            grad_h_y = (jnp.abs(y_vec_for_loss) + eps) ** (p - 2.0) * y_vec_for_loss
            grad_h_y = jax.lax.stop_gradient(grad_h_y)

            h_u = jnp.sum((jnp.abs(q_vals_for_loss) + eps) ** p, axis=-1) / p          # (B,)
            inner = jnp.sum(grad_h_y * q_vals_for_loss, axis=-1)                       # (B,)
            data_loss = jnp.mean(h_u - inner)                                          # scalar (no constant term)
            # ----------------------------------------------------------------------

            # --- 6) log-barrier on INTERNAL outputs (keep ||Q||_p ≤ R) -----------
            norm_p_p_internal = jnp.sum((jnp.abs(q_vals_internal) + eps) ** p, axis=-1)  # (B,)
            slack = jnp.clip(R ** p - norm_p_p_internal, a_min=1e-12)
            barrier = -tau * jnp.mean(jnp.log(slack))
            # ----------------------------------------------------------------------

            loss = data_loss + barrier

            return loss, {
                'loss': loss,
                'data_loss': data_loss,
                'barrier': barrier,
                'q_mean_internal': q_vals_internal.mean(),
                'avi_reward_scale': avi_reward_scale,
                'loss_scale_s': loss_scale_s,
            }

        new_q_network, info = agent.q_network.apply_loss_fn(loss_fn=loss_fn, has_aux=True)
        return agent.replace(q_network=new_q_network), info

    @jax.jit
    def hard_target_update(agent):
        new_target_q_network = target_update(agent.q_network, agent.target_q_network, 1.0)
        return agent.replace(target_q_network=new_target_q_network)

    @jax.jit
    def sample_actions(agent,
                       observations: np.ndarray,
                       *,
                       seed: PRNGKey,
                       epsilon: float) -> jnp.ndarray:
        """Epsilon-greedy. INTERNAL scale does not affect argmax."""
        rng, key = jax.random.split(seed)
        q_values_internal = agent.q_network(observations)  # INTERNAL
        greedy_actions = jnp.argmax(q_values_internal, axis=-1)
        random_actions = jax.random.randint(key, shape=greedy_actions.shape,
                                            minval=0, maxval=q_values_internal.shape[-1])
        use_random = jax.random.uniform(rng, shape=greedy_actions.shape) < epsilon
        actions = jnp.where(use_random, random_actions, greedy_actions)
        return actions


# ===================== Factory =====================
def create_learner(
        random_key: int,
        observations: jnp.ndarray,
        num_actions: int,
        learning_rate: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        target_update_freq: int = 50,
        loss_p: float = 4.0,
        # tau: float = 0.005,
        **kwargs):
    print('Extra kwargs:', kwargs)

    rng = random_key
    rng, q_key = jax.random.split(rng)

    # --- INTERNAL scaling parameters -------------------------------------------
    R = float(kwargs.get('radius_R', 0.5))                 # recommend R ≤ 1/2
    p = float(loss_p)
    A = float(num_actions)

    # INTERNAL output scale: ensure A^{1/p} * S ≤ safety_R * R
    safety_R = float(kwargs.get('out_scale_safety_R', 0.9))
    out_scale_in = safety_R * R / (A ** (1.0 / p))         # S

    # Physical Q upper bound (e.g., CartPole r_max=1, γ=0.99 → ~100)
    r_max = float(kwargs.get('r_max', 1.0))
    qmax_upper = r_max / max(1e-8, (1.0 - discount))

    # AVI-only reward scale α: match physical → internal
    avi_safety = float(kwargs.get('avi_reward_safety', 1.0))
    avi_reward_scale = avi_safety * out_scale_in / max(1e-8, qmax_upper)  # α ∈ (0,1]
    # --------------------------------------------------------------------------

    # --- EXTERNAL loss-only symmetric scale (small & fixed recommended) -------
    loss_scale_s = float(kwargs.get('loss_scale_s', 10.0))  # s (keep modest; avoid huge s^p)
    # --------------------------------------------------------------------------

    # DQN network with INTERNAL tanh head
    q_network_def = DQNNet(
        hidden_dims,
        num_actions=num_actions,
        out_scale_in=out_scale_in,
        use_tanh_head=True,
        tanh_temp=kwargs.get('tanh_temp', 1.0),
        head_mode=kwargs.get('head_mode', 'signed')
    )
    q_params = q_network_def.init(q_key, observations)['params']

    optimizer = optax.chain(
        # optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=learning_rate)
    )

    q_network = TrainState.create(q_network_def, q_params, tx=optimizer)
    target_q_network = TrainState.create(q_network_def, q_params)

    # Config pack
    config = flax.core.FrozenDict(dict(
        discount=discount,
        target_update_freq=target_update_freq,
        loss_p=loss_p,
        # Bregman + barrier + scaling params
        radius_R=R,
        barrier_tau=kwargs.get('barrier_tau', 1e-3),
        # INTERNAL
        out_scale_in=out_scale_in,
        out_scale_safety_R=safety_R,
        avi_reward_scale=avi_reward_scale,     # α (AVI only)
        avi_reward_safety=avi_safety,
        # EXTERNAL (loss-only)
        loss_scale_s=loss_scale_s,             # s (small, fixed)
        # logging refs
        r_max=r_max,
        qmax_upper=qmax_upper,
    ))

    return WPDQNAgent(rng, q_network=q_network, target_q_network=target_q_network, config=config)


def get_default_config():
    import ml_collections
    return ml_collections.ConfigDict({
        'learning_rate': 1e-4,
        'hidden_dims': (256, 256),
        'discount': 0.99,
        'target_update_freq': 80, #50 fore
        'loss_p': 6.0,
        # barrier & scaling defaults
        'radius_R': 0.5,              # Lp half-ball
        'barrier_tau': 1e-3,          # log-barrier strength
        'out_scale_safety_R': 0.9,    # INTERNAL scale safety
        'avi_reward_safety': 1.0,     # reward scaling safety (AVI only)
        'loss_scale_s': 10.0,         # EXTERNAL loss symmetric scale (small)
        'r_max': 1.0,                 # CartPole step reward
        'tanh_temp': 1.0,
        'head_mode': 'signed',
    })
