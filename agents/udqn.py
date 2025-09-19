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

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = observations
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        q_values = nn.Dense(self.num_actions)(x)
        return q_values


# ===================== Agent =====================
class WPDQNAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    q_network: TrainState
    target_q_network: TrainState
    config: dict = nonpytree_field()

    @jax.jit
    def update(agent, batch: Batch):

        def loss_fn(q_params):
            p = agent.config.get('loss_p', 4.0)
            eps = 1e-6

            # --- 1) TD target -----------------------------------------
            q_next_values = agent.target_q_network(batch['next_observations'])
            q_next = jnp.max(q_next_values, axis=-1)
            rewards = batch['rewards']
            target_q = rewards + agent.config['discount'] * batch['masks'] * q_next

            # --- 2) current Q ------------------------------------------
            q_values = agent.q_network(batch['observations'], params=q_params)

            # --- 3) Double-DQN style vector target ---------------------
            q_bar = agent.target_q_network(batch['observations'])
            actions = batch['actions'].astype('int32')
            a_onehot = jax.nn.one_hot(actions, q_bar.shape[-1])
            q_bar_sa = jnp.take_along_axis(q_bar, actions[:, None], axis=-1).squeeze(-1)
            delta = target_q - q_bar_sa
            y_vec = q_bar + a_onehot * delta[:, None]
            y_vec = jax.lax.stop_gradient(y_vec)

            # --- 4) Bregman data term ---------------------------------
            grad_h_y = (jnp.abs(y_vec) + eps) ** (p - 2.0) * y_vec
            grad_h_y = jax.lax.stop_gradient(grad_h_y)

            h_u = jnp.sum((jnp.abs(q_values) + eps) ** p, axis=-1) / p
            inner = jnp.sum(grad_h_y * q_values, axis=-1)
            data_loss = jnp.mean(h_u - inner)

            loss = data_loss

            return loss, {
                'loss': loss,
                'data_loss': data_loss,
                'q_mean': q_values.mean(),
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
        rng, key = jax.random.split(seed)
        q_values = agent.q_network(observations)
        greedy_actions = jnp.argmax(q_values, axis=-1)
        random_actions = jax.random.randint(key, shape=greedy_actions.shape,
                                            minval=0, maxval=q_values.shape[-1])
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
        **kwargs):
    print('Extra kwargs:', kwargs)

    rng = random_key
    rng, q_key = jax.random.split(rng)

    q_network_def = DQNNet(
        hidden_dims,
        num_actions=num_actions,
    )
    q_params = q_network_def.init(q_key, observations)['params']

    optimizer = optax.chain(
        optax.adam(learning_rate=learning_rate)
    )

    q_network = TrainState.create(q_network_def, q_params, tx=optimizer)
    target_q_network = TrainState.create(q_network_def, q_params)

    config = flax.core.FrozenDict(dict(
        discount=discount,
        target_update_freq=target_update_freq,
        loss_p=loss_p,
    ))

    return WPDQNAgent(rng, q_network=q_network, target_q_network=target_q_network, config=config)


def get_default_config():
    import ml_collections
    return ml_collections.ConfigDict({
        'learning_rate': 1e-4,
        'hidden_dims': (256, 256),
        'discount': 0.99,
        'target_update_freq': 80,
        'loss_p': 6.0,
    })
