"""
논문 단계:
    Risky Overcooked에서 DDQN baseline/학습을 실행하기 전, 설정값과 저장 경로를 준비하는 초기화 단계.

호출 위치:
    `risky_overcooked_rl.algorithms.DDQN` 패키지를 import할 때 실행된다.

전체 역할:
    DDQN의 기본 config yaml, batching yaml, 모델 저장 경로를 불러오고,
    중첩 dictionary 형태의 config에서 특정 key를 찾거나 수정하는 helper 함수를 제공한다.
"""

import sys
import os

print('\\'.join(os.getcwd().split('\\')[:-1]))
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))


def search_config_value(config, target_key, level=0):
    """중첩 config dictionary에서 target_key 하나를 찾아 해당 값을 반환한다."""
    found = None
    if isinstance(config, dict):
        for key, val in config.items():
            if key == target_key:
                assert found is None, f"Key '{target_key}' found multiple times in the configuration."
                found = config[key]
            else:
                _found = search_config_value(val, target_key, level=level + 1)
                if _found is not None:
                    assert found is None, f"Key '{target_key}' found multiple times in the configuration."
                    found = _found
    if level == 0:
        assert found is not None, f"Key '{target_key}' not found in the configuration."
    return found

# 중첩 config dictionary에서 target_key를 찾아 new_value로 바꾼다.
def set_config_value(config, target_key, new_value, level=0):
    """
    Recursively searches for target_key in a nested dictionary d and sets its value to new_value.
    Parameters:
    - config (dict): The dictionary to search.
    - target_key (str): The key to search for.
    - new_value: The value to set when the key is found.
    """
    was_found = False
    if isinstance(config, dict):
        for key,val in config.items():
            if key == target_key:
                config[key] = new_value
                assert was_found == False, f"Key '{target_key}' found multiple times in the configuration."
                was_found = True
            else:
                _found = set_config_value(val, target_key, new_value,level=level+1)
                if _found:
                    assert was_found == False, f"Key '{target_key}' found multiple times in the configuration."
                    was_found = True
    if level == 0:
        assert was_found, f"Key '{target_key}' not found in the configuration."
    return was_found

# 여기서 어떤 _config.yaml을 읽는지 확인
def get_default_config(path = '\\risky_overcooked_rl\\algorithms\\DDQN\\_config.yaml'):
    """DDQN 기본 학습 config yaml을 읽어 dictionary로 반환한다."""
    import yaml
    import os

    dirs = os.getcwd().split('\\')
    src_idx = dirs.index('src')  # find index of src directory
    src_dir = '\\'.join(dirs[:src_idx+1])
    with open(f'{src_dir}{path}') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config

def get_default_batching(path = '\\risky_overcooked_rl\\algorithms\\DDQN\\_batching.yaml'):
    """DDQN batching 설정 yaml을 읽어 dictionary로 반환한다."""
    import yaml
    import os

    dirs = os.getcwd().split('\\')
    src_idx = dirs.index('src')  # find index of src directory
    src_dir = '\\'.join(dirs[:src_idx+1])
    with open(f'{src_dir}{path}') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config

def get_absolute_save_dir(path = '\\risky_overcooked_rl\\algorithms\\DDQN\\models\\'):
    """현재 작업 경로의 src 위치를 기준으로 DDQN 모델 저장 절대 경로를 만든다."""
    dirs = os.getcwd().split('\\')
    src_idx = dirs.index('src') # find index of src directory
    return '\\'.join(dirs[:src_idx+1]) + path


def get_save_dir():
    """기존 legacy 모델 저장 경로를 반환한다."""
    # TODO: implement this is save confing to generalize to other algs
    return '\\risky_overcooked_rl\\models'


# SAVE_DIR: DDQN 모델 checkpoint/weight를 저장할 기본 경로.
SAVE_DIR = get_absolute_save_dir()

# CONFIG: `_config.yaml`에서 읽어온 DDQN 기본 학습 설정 dictionary.
CONFIG = get_default_config()
