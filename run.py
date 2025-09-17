import subprocess
import random
import os


# --- 실험 설정 ---
# 실행할 메인 스크립트 파일 이름
script_to_run = "main.py"

# (수정됨) WandB에 표시될 기본 프로젝트 이름
project_name = "DQN-Paper-REG-[USETHIS]"

# (수정됨) 실험할 환경 목록
env_names = ["CartPole-v1", "LunarLander-v3"
             # "Acrobot-v1", "MountainCar-v0", "LunarLander-v3"
             ]  # 나중에 ["CartPole-v1", "LunarLander-v2"] 와 같이 추가 가능

# 실험할 에이전트 목록
agents = ["wpdqn", "ordqn_ngc"]

# 생성할 랜덤 시드 개수
num_seeds = 10

# 결과를 저장할 기본 디렉토리
base_save_dir = "./results"


# --- 스크립트 실행 ---
def main():
    """지정된 환경, 에이전트, 랜덤 시드로 여러 번의 실험을 실행합니다."""

    print(f"'{script_to_run}' 스크립트를 사용하여 실험을 시작합니다.")
    print(f"프로젝트 이름: {project_name}")
    print(f"환경: {env_names}")
    print(f"에이전트: {agents}")
    print(f"시드 개수: {num_seeds}")
    print("-" * 30)

    # 랜덤 시드 생성
    seeds = [random.randint(0, 100000) for _ in range(num_seeds)]
    print(f"생성된 랜덤 시드: {seeds}")
    print("-" * 30)

    # (수정됨) 각 환경에 대해 순회
    for seed in seeds:
        for env_name in env_names:
            print(f"\n{'=' * 10} 환경: {env_name} 실험 시작 {'=' * 10}")

            # 각 에이전트와 시드에 대해 실험 실행
            for agent in agents:
                print(f"==> 에이전트: {agent}, 시드: {seed} 실험 시작...")

                # 실행할 커맨드 구성
                command = [
                    "python",
                    script_to_run,
                    f"--agent_name={agent}",
                    f"--seed={seed}",
                    f"--env_name={env_name}",
                    # (수정됨) 지정된 project_name에 env_name을 추가하여 WandB에 기록
                    f"--project_name={project_name}",
                ]
                # 사용자 확인을 위해 실행할 커맨드를 출력
                command_str = " ".join(command)
                print(f"실행 커맨드: {command_str}")

                try:
                    # subprocess를 사용하여 외부 스크립트 실행
                    subprocess.run(command, check=True)
                    print(f"==> 에이전트: {agent}, 시드: {seed} 실험 성공적으로 완료.")
                except subprocess.CalledProcessError as e:
                    print(f"==> 에러 발생! 에이전트: {agent}, 시드: {seed} 실험 실패.")
                    print(f"에러 코드: {e.returncode}")
                except FileNotFoundError:
                    print(f"에러: '{script_to_run}' 파일을 찾을 수 없습니다.")
                    print("제공된 코드를 'main.py' 이름으로 저장했는지 확인하세요.")
                    return  # 스크립트 중단
                print("-" * 20)

    print("\n모든 실험이 완료되었습니다.")


if __name__ == '__main__':
    main()