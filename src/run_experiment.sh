#!/bin/bash

# Hata durumunda betiği sonlandır
set -e

# Yapılandırma dosyasının yolu
CONFIG_FILE="config.sh"

# Yapılandırma dosyasının varlığını kontrol et
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Hata: Yapılandırma dosyası bulunamadı: $CONFIG_FILE"
    exit 1
fi

# Yapılandırma dosyasındaki değişkenleri bu betiğe dahil et
source "$CONFIG_FILE"

# NORM_FIRST bayrağını yönetmek için bir değişken oluşturalım
NORM_FLAG=""
if [ "$NORM_FIRST" = "False" ] || [ "$NORM_FIRST" = "false" ]; then
    NORM_FLAG="--no-norm-first"
fi

# train.py'yi çalıştırmak için komutu oluştur
# Değişkenleri kullanarak argümanları bir araya getiriyoruz.
COMMAND="python -m train \
    --model $MODEL \
    --dataset $DATASET \
    --log_dir $LOG_DIR \
    --batch_size $BATCH_SIZE \
    --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES \
    --gradient_clip_val $GRADIENT_CLIP_VAL \
    --learning_rate $LEARNING_RATE \
    --num_workers $NUM_WORKERS \
    --dropout $DROPOUT \
    --cnn $CNN \
    --embed_size $EMBED_SIZE \
    --num_heads $NUM_HEADS \
    --num_layers $NUM_LAYERS \
    --enc_calculate_num $ENC_CALCULATE_NUM \
    --rk_type $RK_TYPE \
    --encoder_history_type $ENCODER_HISTORY_TYPE \
    $NORM_FLAG"

# Oluşturulan komutu ekrana yazdır
echo "Çalıştırılan Komut:"
echo "$COMMAND"
echo "-----------------------------------------------------"

# Komutu çalıştır
eval $COMMAND