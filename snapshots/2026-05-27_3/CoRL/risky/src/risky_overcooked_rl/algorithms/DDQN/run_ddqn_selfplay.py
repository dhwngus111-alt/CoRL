from risky_overcooked_rl.algorithms.DDQN.utils.curriculum import CurriculumTrainer
from risky_overcooked_rl.algorithms.DDQN.utils.agents import SelfPlay_QRE_OSA_CPT
import risky_overcooked_rl.algorithms.DDQN as Algorithm

def main(): # 여기가 단일 학습 시작점 
    config = Algorithm.get_default_config() # Curriculum- prefix 추가함
    config["ALGORITHM"] = 'Curriculum-' + config['ALGORITHM'] # Add Curriculum to Algorithm name
    
    # 여기가 __init__ 진입점이다.
    CurriculumTrainer(SelfPlay_QRE_OSA_CPT, config).run() # 배치 학습을 시작한다.


if __name__ == "__main__":
    main() 