# Configuration file for experiment parameters

# Model and Dataset Settings
MODEL="VTN_HCPF"
DATASET="handcrop_poseflow"
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOG_DIR="${PROJECT_ROOT}/logs/run_methods"

# Training Parameters
BATCH_SIZE=8
ACCUMULATE_GRAD_BATCHES=4
GRADIENT_CLIP_VAL=1.0
LEARNING_RATE=0.0001
NUM_WORKERS=4
DROPOUT=0.2

# Model Architecture Parameters
EMBED_SIZE=512
NUM_HEADS=8
NUM_LAYERS=4

CNN="rn18"
ENC_CALCULATE_NUM=4
RK_TYPE="initialization"
ENCODER_HISTORY_TYPE="dense"
NORM_FIRST=True

