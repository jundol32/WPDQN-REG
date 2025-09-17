# 필요한 모듈을 임포트합니다.
from typing import Dict
import jax
import gymnasium as gym
import numpy as np
from collections import defaultdict
import time
import wandb
import tempfile
from pathlib import Path
from gymnasium.wrappers import RecordVideo


# ... (유틸리티 함수 및 GymnasiumEpisodeMonitor 클래스는 동일) ...
def supply_rng(f, rng=jax.random.PRNGKey(0)):
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class GymnasiumEpisodeMonitor(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_start_time = time.time()
        self.total_timesteps = 0

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.episode_reward += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}
        if done:
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.episode_length,
                't': time.time() - self.episode_start_time
            }
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_start_time = time.time()
        return obs, info


# ------------------------------------------------------------------
# -- ✨ JAX Key를 사용하도록 수정된 최종 평가 함수 --
# ------------------------------------------------------------------
def evaluate_gymnasium(
        policy_fn,
        env: gym.Env,
        num_episodes: int,
        key: jax.random.PRNGKey,  # ✨ 정수 시드 대신 JAX 키를 받습니다.
        record_video: bool = False
):
    stats = defaultdict(list)
    temp_dir = None

    eval_env = env
    if record_video:
        temp_dir = tempfile.TemporaryDirectory()
        video_dir = temp_dir.name
        eval_env = RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda x: x == 0,
            name_prefix="eval_video"
        )

    # 헬퍼 함수에 JAX 키를 전달합니다.
    run_evaluation_episodes(policy_fn, eval_env, num_episodes, key, stats)

    if record_video:
        video_path = next(Path(video_dir).glob("*.mp4"), None)
        if video_path:
            stats['rollout'] = wandb.Video(str(video_path), fps=30, format="mp4")

    final_stats = {k: np.mean(v) for k, v in stats.items() if k != 'rollout'}
    if 'rollout' in stats:
        final_stats['rollout'] = stats['rollout']

    return final_stats, temp_dir


def run_evaluation_episodes(policy_fn, env, num_episodes, key, stats):
    """
    주어진 환경에서 여러 에피소드를 실행하고 통계를 기록하는 헬퍼 함수.
    """
    # ✨ 각 에피소드에 사용할 키들을 한 번에 생성합니다.
    keys = jax.random.split(key, num_episodes)

    for i in range(num_episodes):
        # ✨ JAX 키로부터 결정론적인 정수 시드를 생성합니다.
        # Gymnasium env는 정수 시드를 받기 때문입니다.
        seed = jax.random.randint(keys[i], shape=(), minval=0, maxval=np.iinfo(np.int32).max)

        observation, _ = env.reset(seed=int(seed))
        done = False
        while not done:
            action = policy_fn(observation)
            action = np.asarray(action)
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if 'episode' in info:
            stats['return'].append(info['episode']['r'])
            stats['length'].append(info['episode']['l'])
