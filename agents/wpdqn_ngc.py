"""Implementation of the DQN algorithm for discrete control."""
import functools
from jaxrl_m.typing import *

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxrl_m.common import TrainState, target_update, nonpytree_field

import flax
import flax.linen as nn


# DQN은 상태(observation)를 입력으로 받아 각 이산적인 행동(discrete action)의
# Q-value를 출력하는 네트워크를 사용합니다.
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


class WPDQNNGCAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    q_network: TrainState
    target_q_network: TrainState
    config: dict = nonpytree_field()

    @jax.jit
    def update(agent, batch: Batch):
        # def loss_fn(q_params):
        #     # --- 1. 타겟 Q-value (y) 계산 ---
        #     next_q_vals = agent.target_q_network(batch['next_observations'])
        #     next_q = jnp.max(next_q_vals, axis=-1)
        #     target_q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_q
        #
        #     # --- 2. 타겟 네트워크를 이용한 가중치 계산 ---
        #     # Q_θk(s, a) 계산
        #     q_vals_target = agent.target_q_network(batch['observations'])
        #     action_q_target = jnp.take_along_axis(q_vals_target,
        #                                           jnp.expand_dims(batch['actions'].astype('int32'), axis=-1),
        #                                           axis=-1).squeeze(axis=-1)
        #
        #     # 타겟 네트워크 기준 TD 오차: |y - Q_θk(s, a)|
        #     td_error_target = jnp.abs(target_q - action_q_target)
        #
        #     # p 값 가져오기
        #     p = agent.config.get('loss_p', 4.0)
        #
        #     # 가중치 분자: |y - Q_θk(s, a)|^(p-2)
        #     weights_numerator = td_error_target ** (p - 2.0)
        #
        #     # 가중치 분모: Σ |¯y - Q_θk(¯s, ¯a)|^(p-2)
        #     weights_denominator = jnp.sum(weights_numerator)
        #
        #     # 최종 가중치 (안정성을 위해 epsilon 추가)
        #     weights = weights_numerator / (weights_denominator + 1e-6)
        #
        #     # --- 3. 현재 네트워크를 이용한 TD 오차 계산 ---
        #     # Q_θ(s, a) 계산
        #     q_vals_current = agent.q_network(batch['observations'], params=q_params)
        #     action_q_current = jnp.take_along_axis(q_vals_current,
        #                                            jnp.expand_dims(batch['actions'].astype('int32'), axis=-1),
        #                                            axis=-1).squeeze(axis=-1)
        #
        #     # 제곱 오차: (y - Q_θ(s, a))^2
        #     squared_error = (target_q - action_q_current) ** 2
        #
        #     # --- 4. 최종 가중 손실 계산 ---
        #     # L = 0.5 * (1/|B|) * Σ [ weights * squared_error ]
        #     weighted_squared_error = weights * squared_error
        #     loss = 0.5 * jnp.mean(weighted_squared_error)
        #
        #     return loss, {
        #         'loss': loss,
        #         'q': action_q_current.mean(),
        #     }

        def loss_fn(q_params):
            # --- 1. 타겟 Q-value (y) 계산 (기존과 동일) ---
            next_q_vals = agent.target_q_network(batch['next_observations'])
            next_q = jnp.max(next_q_vals, axis=-1)
            target_q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_q

            # --- 2. '현재' 네트워크를 이용한 가중치 계산 (수정된 부분) ---
            # 현재 네트워크 Q_θ(s, a) 계산
            q_vals_current_for_weights = agent.q_network(batch['observations'], params=q_params)
            action_q_current_for_weights = jnp.take_along_axis(q_vals_current_for_weights,
                                                               jnp.expand_dims(batch['actions'].astype('int32'),
                                                                               axis=-1),
                                                               axis=-1).squeeze(axis=-1)

            # stop_gradient를 사용하여 가중치 계산이 역전파에 영향을 미치지 않도록 함
            # 이 TD 오차는 오직 가중치를 만드는 데에만 사용
            td_error_current = jnp.abs(target_q - jax.lax.stop_gradient(action_q_current_for_weights))

            p = agent.config.get('loss_p', 4.0)
            weights_numerator = td_error_current ** (p - 2.0)
            weights_denominator = jnp.sum(weights_numerator)
            weights = weights_numerator / (weights_denominator + 1e-6)

            # --- 3. 실제 손실 계산을 위한 TD 오차 계산 (기존과 유사) ---
            # (주의: 여기서는 그래디언트가 흘러야 하므로 stop_gradient 사용 안 함)
            q_vals_current = agent.q_network(batch['observations'], params=q_params)
            action_q_current = jnp.take_along_axis(q_vals_current,
                                                   jnp.expand_dims(batch['actions'].astype('int32'), axis=-1),
                                                   axis=-1).squeeze(axis=-1)

            squared_error = (target_q - action_q_current) ** 2

            # --- 4. 최종 가중 손실 계산 (기존과 동일) ---
            weighted_squared_error = weights * squared_error  # 실시간 가중치 적용
            loss = 0.5 * jnp.mean(weighted_squared_error)

            return loss, {
                'loss': loss,
                'q': action_q_current.mean(),
            }



        # Q-Network 업데이트
        new_q_network, info = agent.q_network.apply_loss_fn(loss_fn=loss_fn, has_aux=True)

        # Target Q-Network를 천천히 업데이트 (soft update)
        # new_target_q_network = target_update(new_q_network, agent.target_q_network, agent.config['target_update_rate'])




        return agent.replace(
            q_network=new_q_network,
            # target_q_network=new_target_q_network
        ), info

    @jax.jit
    def hard_target_update(agent):
        """
        타겟 네트워크의 가중치를 현재 Q-네트워크의 가중치로 복사합니다. (Hard Update)
        이 함수는 JIT으로 컴파일되어 효율적으로 실행됩니다.
        """
        new_target_q_network = target_update(agent.q_network, agent.target_q_network, 1.0)
        return agent.replace(target_q_network=new_target_q_network)



    @jax.jit
    def sample_actions(agent,
                       observations: np.ndarray,
                       *,
                       seed: PRNGKey,
                       epsilon: float) -> jnp.ndarray:
        """
        Epsilon-greedy 정책에 따라 행동을 샘플링합니다.
        """
        rng, key = jax.random.split(seed)

        # 현재 Q-Network로부터 각 행동의 Q-value를 계산합니다.
        q_values = agent.q_network(observations)
        # Q-value가 가장 높은 행동을 선택합니다 (greedy action).
        greedy_actions = jnp.argmax(q_values, axis=-1)

        # Epsilon 확률로 무작위 행동을 선택하고, (1-Epsilon) 확률로 그리디한 행동을 선택합니다.
        random_actions = jax.random.randint(key, shape=greedy_actions.shape, minval=0, maxval=q_values.shape[-1])
        use_random = jax.random.uniform(rng, shape=greedy_actions.shape) < epsilon

        actions = jnp.where(use_random, random_actions, greedy_actions)
        return actions


def create_learner(
        random_key: int,
        observations: jnp.ndarray,
        num_actions: int,
        learning_rate: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        target_update_freq: int = 50,
        loss_p: float = 4.0,
        # tau: float = 0.005, # target_update_freq 대신 tau를 받습니다.
        **kwargs):
    print('Extra kwargs:', kwargs)

    rng = random_key
    rng, q_key = jax.random.split(rng)

    # DQN 네트워크 정의
    q_network_def = DQNNet(hidden_dims, num_actions=num_actions)
    q_params = q_network_def.init(q_key, observations)['params']

    # Q-Network와 Target Q-Network 생성

    optimizer = optax.chain(
        # optax.clip_by_global_norm(1.0),  # 그래디언트의 전체 L2 norm이 1.0을 넘지 않도록 클리핑
        optax.adam(learning_rate=learning_rate)
    )

    q_network = TrainState.create(q_network_def, q_params, tx=optimizer)
    target_q_network = TrainState.create(q_network_def, q_params)

    # DQN 설정값
    config = flax.core.FrozenDict(dict(
        discount=discount,
        target_update_freq=target_update_freq,
        loss_p=loss_p,
    ))

    return WPDQNNGCAgent(rng, q_network=q_network, target_q_network=target_q_network, config=config)


def get_default_config():
    import ml_collections

    return ml_collections.ConfigDict({
        'learning_rate': 3e-4,
        'hidden_dims': (256, 256),
        'discount': 0.99,
        # 'tau': 0.005,
        'target_update_freq' : 5, # target_update_freq 대신 tau로 변경
        'loss_p': 4.0,
    })
