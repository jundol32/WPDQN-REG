"""DQN with Lp-Bregman + log-barrier.
- INTERNAL scale for contraction (AVI/TD only)
- EXTERNAL rescale for loss to avoid tiny magnitudes
- Tanh head limits Q_internal into an Lp-ball (helps barrier)
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
    # -------------------- NEW: internal output scaling & head mode --------------------
    out_scale_in: float = 1.0         # 내부 출력 스케일 (Q_internal의 최대 크기 스케일)
    use_tanh_head: bool = True        # tanh 헤드 사용
    tanh_temp: float = 1.0            # tanh 온도
    head_mode: str = "signed"         # "signed" → [-S,S], "unsigned" → [0,S]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = observations
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        logits = nn.Dense(self.num_actions)(x)
        # -------------------- NEW: head range control --------------------
        if self.use_tanh_head:
            t = jnp.tanh(logits / self.tanh_temp)
            if self.head_mode == "unsigned":
                # [0, S]: (t+1)/2 ∈ [0,1] → × out_scale_in
                q_values_internal = self.out_scale_in * (t + 1.0) * 0.5
            else:
                # [-S, S]
                q_values_internal = self.out_scale_in * t
        else:
            # Head를 제한하지 않을 때도 내부 스케일만 곱해줌
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

            # -------------------- NEW: AVI-only reward scale α --------------------
            # 내부(AVI/TD)에서만 스케일 다운하여 수축/경계 유지
            avi_reward_scale = agent.config.get('avi_reward_scale', 1.0)  # α
            loss_rescale = agent.config.get('loss_rescale', 1.0)          # 손실 계산 시 다시 ×(1/α) 효과
            # ---------------------------------------------------------------------

            # --- 1) TD target (INTERNAL scale) -----------------------------------
            # next_q_internal: contraction/AVI 계산에만 관여
            next_q_vals_internal = agent.target_q_network(batch['next_observations'])  # (B, A), INTERNAL
            next_q_internal = jnp.max(next_q_vals_internal, axis=-1)                   # (B,)
            # 보상도 AVI 내부에서만 스케일 다운 (α)
            rewards_internal = avi_reward_scale * batch['rewards']                     # NEW
            target_q_internal = rewards_internal + agent.config['discount'] * batch['masks'] * next_q_internal  # (B,)
            # ---------------------------------------------------------------------

            # --- 2) 현재 Q (INTERNAL) --------------------------------------------
            q_vals_internal = agent.q_network(batch['observations'], params=q_params)  # (B, A), INTERNAL
            # ---------------------------------------------------------------------

            # --- 3) Double-DQN style vector target (INTERNAL) --------------------
            q_bar_internal = agent.target_q_network(batch['observations'])             # (B, A), INTERNAL
            actions = batch['actions'].astype('int32')
            a_onehot = jax.nn.one_hot(actions, q_bar_internal.shape[-1])               # (B, A)
            q_bar_sa_internal = jnp.take_along_axis(q_bar_internal, actions[:, None], axis=-1).squeeze(-1)  # (B,)
            delta_internal = target_q_internal - q_bar_sa_internal                     # (B,)
            y_vec_internal = q_bar_internal + a_onehot * delta_internal[:, None]       # (B, A)
            y_vec_internal = jax.lax.stop_gradient(y_vec_internal)
            # ---------------------------------------------------------------------

            # --- 4) EXTERNAL rescale for loss (PHYSICAL scale) -------------------
            # 손실/그래디언트 규모 보존: INTERNAL → × loss_rescale
            q_vals_for_loss = q_vals_internal * loss_rescale                           # NEW
            y_vec_for_loss = y_vec_internal * loss_rescale                             # NEW
            # ---------------------------------------------------------------------

            # --- 5) Bregman 데이터 항 (PHYSICAL scale에서 계산) -------------------
            # h(x)= (1/p)||x||_p^p → ∇h(y)=|y|^{p-2} y
            grad_h_y = (jnp.abs(y_vec_for_loss) + eps) ** (p - 2.0) * y_vec_for_loss
            grad_h_y = jax.lax.stop_gradient(grad_h_y)

            h_u = jnp.sum((jnp.abs(q_vals_for_loss) + eps) ** p, axis=-1) / p          # (B,)
            inner = jnp.sum(grad_h_y * q_vals_for_loss, axis=-1)                       # (B,)
            data_loss = jnp.mean(h_u - inner)                                          # scalar
            # ---------------------------------------------------------------------

            # --- 6) log-barrier on INTERNAL outputs (keep ||Q||_p ≤ R) ----------
            norm_p_p_internal = jnp.sum((jnp.abs(q_vals_internal) + eps) ** p, axis=-1)  # (B,)
            slack = jnp.clip(R ** p - norm_p_p_internal, a_min=1e-12)
            barrier = -tau * jnp.mean(jnp.log(slack))
            # ---------------------------------------------------------------------

            loss = data_loss + barrier

            return loss, {
                'loss': loss,
                'data_loss': data_loss,
                'barrier': barrier,
                'q_mean_internal': q_vals_internal.mean(),
                'avi_reward_scale': avi_reward_scale,   # 모니터링
                'loss_rescale': loss_rescale,           # 모니터링
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
        """Epsilon-greedy. 내부 스케일은 argmax에 영향 없음."""
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

    # -------------------- NEW: choose INTERNAL/AVI scaling --------------------
    R = float(kwargs.get('radius_R', 0.5))                   # R ≤ 1/2 권장
    p = float(loss_p)
    A = float(num_actions)

    # 내부 출력 스케일(out_scale_in): 1 state에서 모든 action 동일 최대치일 때
    #   Lp-노름이 R을 넘지 않도록 설정:  A^{1/p} * out_scale_in ≤ R
    safety_R = float(kwargs.get('out_scale_safety_R', 0.9))
    out_scale_in = safety_R * R / (A ** (1.0 / p))           # Q_internal 크기 제한

    # CartPole 예시: r_max=1, γ=0.99 → Q_max ≈ 1/(1-γ) ≈ 100
    r_max = float(kwargs.get('r_max', 1.0))
    qmax_upper = r_max / max(1e-8, (1.0 - discount))

    # -------------------- NEW: AVI reward scale α --------------------
    # "물리적 Q"를 내부 크기에 맞게 줄이기 위한 보상 스케일(AVI/TD 한정)
    # 내부 최대 out_scale_in 및 물리적 Q 상계 qmax_upper을 매칭 → α ≈ out_scale_in / qmax_upper
    avi_safety = float(kwargs.get('avi_reward_safety', 1.0))  # 살짝 더 줄이고 싶으면 <1
    avi_reward_scale = avi_safety * out_scale_in / max(1e-8, qmax_upper)  # α
    # -----------------------------------------------------------------

    # -------------------- NEW: Loss rescale (EXTERNAL) ----------------
    # 손실 계산 시 INTERNAL을 다시 물리 스케일로 복구: × (1/α)
    phys_scale_safety = float(kwargs.get('phys_scale_safety', 1.0))
    loss_rescale = phys_scale_safety / max(1e-8, avi_reward_scale)        # ≈ qmax_upper/out_scale_in
    # ------------------------------------------------------------------

    # DQN 네트워크 정의 (INTERNAL tanh head 포함)
    q_network_def = DQNNet(
        hidden_dims,
        num_actions=num_actions,
        out_scale_in=out_scale_in,                         # NEW
        use_tanh_head=True,                                # NEW
        tanh_temp=kwargs.get('tanh_temp', 1.0),            # NEW
        head_mode=kwargs.get('head_mode', 'signed')        # NEW: 'signed' or 'unsigned'
    )
    q_params = q_network_def.init(q_key, observations)['params']

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=learning_rate)
    )

    q_network = TrainState.create(q_network_def, q_params, tx=optimizer)
    target_q_network = TrainState.create(q_network_def, q_params)

    # Config
    config = flax.core.FrozenDict(dict(
        discount=discount,
        target_update_freq=target_update_freq,
        loss_p=loss_p,
        # -------------------- NEW: Bregman + barrier + (in/out) scaling params ----
        radius_R=R,
        barrier_tau=kwargs.get('barrier_tau', 1e-3),
        out_scale_in=out_scale_in,
        out_scale_safety_R=safety_R,
        # AVI/TD 전용 보상 스케일(축소)
        avi_reward_scale=avi_reward_scale,       # α  (AVI에서만 사용)
        avi_reward_safety=avi_safety,
        # 손실 계산 시 복구 스케일(확대)
        loss_rescale=loss_rescale,               # ≈ 1/α
        phys_scale_safety=phys_scale_safety,
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
        'target_update_freq': 20,
        'loss_p': 4.0,
        # -------------------- NEW defaults --------------------
        'radius_R': 0.5,              # barrier half-ball
        'barrier_tau': 1e-3,          # log-barrier strength
        'out_scale_safety_R': 0.9,    # INTERNAL scale safety
        'avi_reward_safety': 1.0,     # reward scaling safety (AVI only)
        'phys_scale_safety': 1.0,     # loss rescale safety (EXTERNAL)
        'r_max': 1.0,                 # CartPole step reward
        'tanh_temp': 1.0,             # tanh head temperature
        'head_mode': 'signed',        # 'signed'([-S,S]) or 'unsigned'([0,S])
    })
