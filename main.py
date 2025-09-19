import os
import random
import pickle
from functools import partial

# GPU 메모리 관련 설정 (필요에 따라 조절)
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.80'
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'

from absl import app, flags
import numpy as np
import jax
import tqdm
import gymnasium as gym
import importlib

from jaxrl_m.wandb import setup_wandb, default_wandb_config, get_flag_dict
import wandb
from jaxrl_m.evaluation import evaluate_gymnasium, GymnasiumEpisodeMonitor, supply_rng, flatten
from jaxrl_m.dataset import ReplayBuffer
from ml_collections import config_flags, config_dict
from flax.training import checkpoints

FLAGS = flags.FLAGS

# --- 기본 설정 플래그 ---
flags.DEFINE_string('project_name', 'DQN-CartPole', 'WandB 프로젝트 이름')
flags.DEFINE_string('env_name', 'CartPole-v1', 'Gymnasium 환경 이름.')  # CartPole-v1으로 변경
flags.DEFINE_string('agent_name', 'dqn', 'agents 폴더에 있는 에이전트 모듈 이름 (예: dqn)')
flags.DEFINE_string('save_dir', None, '체크포인트 및 로그 저장 디렉토리.')
flags.DEFINE_integer('seed', 42, '실험을 위한 랜덤 시드.')
flags.DEFINE_integer('eval_episodes', 20, '평가에 사용할 에피소드 수.')
flags.DEFINE_integer('log_interval', 1000, '로그 기록 간격 (스텝).')
flags.DEFINE_integer('eval_interval', 1000, '평가 간격 (스텝).')
flags.DEFINE_integer('save_interval', 20000, '모델 저장 간격 (스텝).')
flags.DEFINE_integer('batch_size', 128, '학습에 사용할 미니배치 크기.')
flags.DEFINE_integer('max_steps', 1000_000, '총 학습 스텝 수.')
flags.DEFINE_integer('start_steps', 1000, '랜덤 액션으로 탐험을 시작할 스텝 수.')

# --- DQN 전용 하이퍼파라미터 플래그 ---
flags.DEFINE_float('epsilon_start', 1.0, 'Epsilon-greedy 탐험의 시작 값.')
flags.DEFINE_float('epsilon_end', 0.1, 'Epsilon-greedy 탐험의 최종 값.')
flags.DEFINE_integer('epsilon_decay_steps', 50000, 'Epsilon이 최종 값까지 감소하는 데 걸리는 스텝 수.')

# --- WandB 및 에이전트 설정 ---
wandb_config = default_wandb_config()
wandb_config.update({
    'group': '{env_name}',
    'name': '{agent_name}_{env_name}_{seed}',
})
config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('config', config_dict.ConfigDict(), '에이전트별 설정.', lock_config=False)


def main(_):
    # 시드 설정
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    # JAX 키 분리
    key = jax.random.PRNGKey(FLAGS.seed)
    key, buffer_key, agent_key, exploration_key, eval_key = jax.random.split(key, 5)

    # WandB 설정
    FLAGS.wandb.project = FLAGS.project_name
    setup_wandb(FLAGS.config.to_dict(), **FLAGS.wandb)

    # 저장 디렉토리 설정
    if FLAGS.save_dir is not None:
        FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, wandb.config.exp_prefix,
                                      wandb.config.experiment_id)
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        print(f'설정 파일 저장 위치: {FLAGS.save_dir}/config.pkl')
        with open(os.path.join(FLAGS.save_dir, 'config.pkl'), 'wb') as f:
            pickle.dump(get_flag_dict(), f)

    # 에이전트 모듈 동적 로딩
    try:
        agent_module_name = f"agents.{FLAGS.agent_name}"
        learner = importlib.import_module(agent_module_name)
        print(f"에이전트 로딩 성공: {FLAGS.agent_name}")
    except ImportError:
        raise ImportError(
            f"'{FLAGS.agent_name}' 에이전트를 불러올 수 없습니다. 'agents/{FLAGS.agent_name}.py' 파일이 있는지 확인하세요.")

    # 에이전트 기본 설정과 커맨드라인 설정 병합


    default_agent_config = learner.get_default_config()






    default_agent_config.update(FLAGS.config)
    FLAGS.config = default_agent_config

    # 환경 생성
    env = GymnasiumEpisodeMonitor(gym.make(FLAGS.env_name))
    eval_env = GymnasiumEpisodeMonitor(gym.make(FLAGS.env_name))
    env.reset(seed=FLAGS.seed)
    eval_env.reset(seed=FLAGS.seed + 100)

    obs, _ = env.reset()

    # 리플레이 버퍼 생성을 위한 예시 데이터
    example_transition = dict(
        observations=obs,
        actions=env.action_space.sample(),
        rewards=0.0,
        masks=1.0,
        next_observations=obs,
    )

    replay_buffer = ReplayBuffer.create(example_transition, size=int(1e4), seed_key=buffer_key)

    # DQN 에이전트 생성 (num_actions 전달)
    agent = learner.create_learner(agent_key,
                                   example_transition['observations'][None],
                                   num_actions=env.action_space.n,  # 이 부분이 핵심적인 변경사항입니다.
                                   **FLAGS.config)

    exploration_metrics = dict()

    # Epsilon 스케줄러 함수 정의
    def get_epsilon(step):
        return np.clip(
            FLAGS.epsilon_start - (FLAGS.epsilon_start - FLAGS.epsilon_end) * (step / FLAGS.epsilon_decay_steps),
            FLAGS.epsilon_end,
            FLAGS.epsilon_start
        )

    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):

        if i < FLAGS.start_steps:
            action = env.action_space.sample()
        else:
            exploration_key, key = jax.random.split(exploration_key)
            epsilon = get_epsilon(i - FLAGS.start_steps)
            # sample_actions 호출 시 epsilon 전달
            action = agent.sample_actions(obs, seed=key, epsilon=epsilon)
            action = np.asarray(action)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        mask = float(not done)

        replay_buffer.add_transition(dict(
            observations=obs,
            actions=action,
            rewards=reward,
            masks=mask,
            next_observations=next_obs,
        ))
        obs = next_obs

        if done:
            exploration_metrics = {f'exploration/{k}': v for k, v in flatten(info).items()}
            obs, _ = env.reset()

        if replay_buffer.size < FLAGS.start_steps:
            continue

        batch = replay_buffer.sample(FLAGS.batch_size)


        agent, update_info = agent.update(batch)  # JIT 컴파일된 핵심 업데이트



        # 주기적으로 Target Network 동기화 (JIT 컴파일 안 됨)
        if i % agent.config['target_update_freq'] == 0:
            agent = agent.hard_target_update()



        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics['training/epsilon'] = get_epsilon(i - FLAGS.start_steps)
            wandb.log(train_metrics, step=i)
            wandb.log(exploration_metrics, step=i)
            exploration_metrics = dict()

        if i % FLAGS.eval_interval == 0:
            # 평가 시에는 epsilon=0 (탐험 없음)
            policy_fn = partial(supply_rng(agent.sample_actions), epsilon=0.0)

            eval_key, eval_subkey = jax.random.split(eval_key)
            eval_info, _ = evaluate_gymnasium(
                policy_fn,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                key=eval_subkey,
            )
            eval_metrics = {f'evaluation/{k}': v for k, v in eval_info.items()}
            wandb.log(eval_metrics, step=i)

        if i % FLAGS.save_interval == 0 and FLAGS.save_dir is not None:
            checkpoints.save_checkpoint(FLAGS.save_dir, agent, i, keep=3)

    wandb.finish()


if __name__ == '__main__':
    app.run(main)
