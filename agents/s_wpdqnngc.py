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

        # ### 변경된 부분 시작 ###
        # Target Q-Network를 천천히 업데이트 (soft update)
        new_target_q_network = target_update(new_q_network, agent.target_q_network, agent.config['tau'])
        # ### 변경된 부분 끝 ###


        return agent.replace(
            q_network=new_q_network,
            target_q_network=new_target_q_network # Soft update 적용
        ), info

    @jax.jit
    def hard_target_update(agent):
        """
        ### 변경된 부분 ###
        Soft update 방식으로 변경되었으므로 이 함수는 아무 작업도 수행하지 않습니다.
        단순히 agent를 그대로 반환하여 외부 학습 코드와의 호환성을 유지합니다.
        """
        return agent



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
        # target_update_freq: int = 50, # ### 변경된 부분 ###
        loss_p: float = 4.0,
        tau: float = 0.005, # target_update_freq 대신 tau를 받습니다. ### 변경된 부분 ###
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
        # target_update_freq=target_update_freq, # ### 변경된 부분 ###
        tau=tau, # ### 변경된 부분 ###
        loss_p=loss_p,
    ))

    return WPDQNNGCAgent(rng, q_network=q_network, target_q_network=target_q_network, config=config)


def get_default_config():
    import ml_collections

    return ml_collections.ConfigDict({
        'learning_rate': 3e-4,
        'hidden_dims': (256, 256),
        'discount': 0.99,
        'tau': 0.005, # ### 변경된 부분 ###
        # 'target_update_freq' : 5, # target_update_freq 대신 tau로 변경 ### 변경된 부분 ###
        'loss_p': 4.0,
    })