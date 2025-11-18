# Deney parametreleri için yapılandırma dosyası

# Model ve Veriseti Ayarları
MODEL="VTN_HCPF"
DATASET="handcrop_poseflow"
LOG_DIR="/home/omer/Masaüstü/tez_calismasi/codebase/ChaLearn-2021-LAP/logs/run_methods"

# Eğitim Parametreleri
BATCH_SIZE=8
ACCUMULATE_GRAD_BATCHES=4
GRADIENT_CLIP_VAL=1.0
LEARNING_RATE=0.0001
NUM_WORKERS=4
DROPOUT=0.2

# Model Mimarisi Parametreleri
EMBED_SIZE=512
NUM_HEADS=8
NUM_LAYERS=4

CNN="convnext_tiny"
ENC_CALCULATE_NUM=2
RK_TYPE="learnable"
ENCODER_HISTORY_TYPE="none"
NORM_FIRST=True

