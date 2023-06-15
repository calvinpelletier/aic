import torch

import cu


TEST_DATA_PATH = cu.get_env_path('CODE_PATH') / 'aic/test/data'
LICHESS_TEST_DATA_PATH = TEST_DATA_PATH / 'lichess'

DEVICE = torch.device('cuda:0')
