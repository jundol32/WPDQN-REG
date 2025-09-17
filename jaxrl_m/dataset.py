import numpy as np
import jax
import jax.numpy as jnp  # ✨ JAX numpy 사용
from jaxrl_m.typing import Data


from jax import tree_util


# 이 함수는 변경사항 없습니다.
def get_size(data: Data) -> int:
    sizes = tree_util.tree_map(lambda arr: len(arr), data)
    return max(tree_util.tree_leaves(sizes))


# ✨ FrozenDict 상속 대신 일반 클래스로 변경하여 내부 상태(key)를 가질 수 있도록 함
class Dataset:
    def __init__(self, data_dict: dict, seed_key: jax.random.PRNGKey = None):
        self._dict = data_dict
        self.size = get_size(self._dict)
        self.key = seed_key  # ✨ JAX 키를 내부 상태로 저장

    def sample(self, batch_size: int, indx=None):
        if indx is None:
            # ✨ 키가 없으면 에러 발생
            if self.key is None:
                raise ValueError(
                    "Cannot sample randomly without a seed_key. Please provide a key during Dataset creation."
                )

            # ✨ JAX 키를 사용한 샘플링 로직
            # 1. 현재 키를 다음 키(new_key)와 이번 샘플링에 사용할 키(subkey)로 분리
            new_key, subkey = jax.random.split(self.key)

            # 2. subkey를 사용해 무작위 인덱스 생성
            indx = jax.random.randint(subkey, (batch_size,), 0, self.size)

            # 3. 내부 키를 다음을 위해 업데이트
            self.key = new_key

        return self.get_subset(indx)

    def get_subset(self, indx):
        # jax array로 인덱싱해도 numpy array는 잘 동작합니다.
        return tree_util.tree_map(lambda arr: arr[indx], self._dict)


class ReplayBuffer(Dataset):
    @classmethod
    def create(cls, transition: Data, size: int, seed_key: jax.random.PRNGKey):  # ✨ seed_key 인자 추가
        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict, seed_key=seed_key)  # ✨ 생성자에 seed_key 전달

    @classmethod
    def create_from_initial_dataset(cls, init_dataset: dict, size: int,
                                    seed_key: jax.random.PRNGKey):  # ✨ seed_key 인자 추가
        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict, seed_key=seed_key)  # ✨ 생성자에 seed_key 전달
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, buffer_dict: dict, seed_key: jax.random.PRNGKey):  # ✨ seed_key 인자 추가
        super().__init__(buffer_dict, seed_key=seed_key)  # ✨ 부모 클래스에 seed_key 전달

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)





def main():
    import pprint

    # 초기 데이터셋 정의 (예시 transition)
    init_transition = {
        'observations': np.random.randn(100, 4).astype(np.float32),
        'actions': np.random.randn(100, 2).astype(np.float32),
        'rewards': np.random.randn(100, 1).astype(np.float32),
    }

    seed = 42
    key1 = jax.random.PRNGKey(seed)
    key2 = jax.random.PRNGKey(seed)  # 동일한 seed로 재현성 검증용

    # 동일한 seed로 2개의 버퍼 생성
    buffer1 = ReplayBuffer.create_from_initial_dataset(init_transition, size=200, seed_key=key1)
    buffer2 = ReplayBuffer.create_from_initial_dataset(init_transition, size=200, seed_key=key2)

    # 동일하게 샘플링
    batch1 = buffer1.sample(batch_size=10)
    batch2 = buffer2.sample(batch_size=10)

    # 결과 비교
    print("샘플 일치 여부:")
    for key in batch1:
        same = np.allclose(batch1[key], batch2[key])
        print(f"- {key}: {'✅ 동일' if same else '❌ 다름'}")

    # 다시 샘플링하면 결과가 달라지는지 확인
    batch1_next = buffer1.sample(batch_size=10)
    batch2_next = buffer2.sample(batch_size=10)

    print("\n2번째 샘플 일치 여부 (PRNGKey 업데이트 확인):")
    for key in batch1_next:
        same = np.allclose(batch1_next[key], batch2_next[key])
        print(f"- {key}: {'✅ 동일' if same else '❌ 다름'}")


if __name__ == "__main__":
    main()
    #재현성 확인 완료