import cu


DATA_PATH = cu.get_env_path('DATA_PATH') / 'ac'
GAME_DB_PATH = DATA_PATH / 'db/game'
TASK_DS_PATH = DATA_PATH / 'ds/task'
UNDERPROMO_DS_PATH = DATA_PATH / 'ds/underpromo'

ELO_BINS = [1500, 1750, 2000, 2250, 2500]
PLY_BINS = [8, 16, 32, 64]
