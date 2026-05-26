# Папка с тайлами Planck (960x960 .fits файлы)
PLANCK_DIR = ""

# Папка с тайлами ACT+Planck (960x960 .fits файлы)
ACT_DIR = ""

# Папка для сохранения чекпоинтов
CHECKPOINT_DIR = ""

# Папка проекта модели
PROJECT_DIR = ""

PATCH_SIZE = 480
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SEED = 42
MAX_ZERO_FRAC = 0.2


IN_CH = 1
BASE_CH = 64
TIME_DIM = 256
GROUPS = 8


T = 1000  # число шагов диффузии
S = 200  # число шагов сэмплирования на инференсе
S_VAR = 0.5  # масштаб дисперсии моста (s)
ALPHA = 0.0
ETA = 0.01


N_EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-4
EMA_START = 0
EMA_DECAY = 0.995
GRAD_CLIP = 1.0

SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3000
