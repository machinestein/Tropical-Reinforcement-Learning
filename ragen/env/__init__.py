from .bandit.config import BanditEnvConfig
from .bandit.env import BanditEnv
from .countdown.config import CountdownEnvConfig
from .countdown.env import CountdownEnv
from .countdown.tropic_env import CountdownStepEnv
from .sokoban.config import SokobanEnvConfig
from .sokoban.env import SokobanEnv
from .frozen_lake.config import FrozenLakeEnvConfig
from .frozen_lake.env import FrozenLakeEnv
from .frozen_lake.tropic_config import FrozenLakeTropicEnvConfig
from .frozen_lake.tropic_env import FrozenLakeTropicEnv
from .metamathqa.env import MetaMathQAEnv
from .metamathqa.config import MetaMathQAEnvConfig
from .lean.config import LeanEnvConfig
from .lean.env import LeanEnv
from .lean.tropic_env import LeanTropicEnv
from .sudoku.config import SudokuEnvConfig
from .sudoku.env import SudokuEnv
from .sudoku.tropic_env import SudokuTropicEnv
from .deepcoder.config import DeepCoderEnvConfig
from .deepcoder.env import DeepCoderEnv
from .game_2048.config import Game2048EnvConfig
from .game_2048.env import Game2048Env
from .rubikscube.config import RubiksCube2x2Config
from .rubikscube.env import RubiksCube2x2Env


REGISTERED_ENVS = {
    'bandit': BanditEnv,
    'countdown': CountdownEnv,
    'countdown_step': CountdownStepEnv,
    'sokoban': SokobanEnv,
    'frozen_lake': FrozenLakeEnv,
    'frozen_lake_tropic': FrozenLakeTropicEnv,
    'metamathqa': MetaMathQAEnv,
    'lean': LeanEnv,
    'lean_tropic': LeanTropicEnv,
    'deepcoder': DeepCoderEnv,
    'sudoku': SudokuEnv,
    'sudoku_tropic': SudokuTropicEnv,
    'game_2048': Game2048Env,
    'rubikscube': RubiksCube2x2Env,
}

REGISTERED_ENV_CONFIGS = {
    'bandit': BanditEnvConfig,
    'countdown': CountdownEnvConfig,
    'countdown_step': CountdownEnvConfig,
    'sokoban': SokobanEnvConfig,
    'frozen_lake': FrozenLakeEnvConfig,
    'frozen_lake_tropic': FrozenLakeTropicEnvConfig,
    'metamathqa': MetaMathQAEnvConfig,
    'deepcoder': DeepCoderEnvConfig,
    'lean': LeanEnvConfig,
    'lean_tropic': LeanEnvConfig,
    'sudoku': SudokuEnvConfig,
    'sudoku_tropic': SudokuEnvConfig,
    'game_2048': Game2048EnvConfig,   
    'rubikscube': RubiksCube2x2Config,
}

try:
    from .alfworld.env import AlfredTXTEnv
    from .alfworld.config import AlfredEnvConfig
    REGISTERED_ENVS['alfworld'] = AlfredTXTEnv
    REGISTERED_ENV_CONFIGS['alfworld'] = AlfredEnvConfig
except ImportError:
    pass

try:
    from .webshop.env import WebShopEnv
    from .webshop.config import WebShopEnvConfig
    REGISTERED_ENVS['webshop'] = WebShopEnv
    REGISTERED_ENV_CONFIGS['webshop'] = WebShopEnvConfig
except ImportError:
    pass

try:
    from .webshop.tropic_env import WebShopTropicEnv
    from .webshop.tropic_config import WebShopTropicEnvConfig
    REGISTERED_ENVS['webshop_tropic'] = WebShopTropicEnv
    REGISTERED_ENV_CONFIGS['webshop_tropic'] = WebShopTropicEnvConfig
except ImportError:
    pass

try:
    from .search.env import SearchEnv
    from .search.config import SearchEnvConfig
    REGISTERED_ENVS['search'] = SearchEnv
    REGISTERED_ENV_CONFIGS['search'] = SearchEnvConfig
except ImportError:
    pass
