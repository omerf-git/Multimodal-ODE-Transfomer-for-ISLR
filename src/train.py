"""
Train a neural network on a given dataset with fixed hyperparameters.
For tuning of the hyperparameters, see tune.py.

Updated for PyTorch Lightning 2.2.2+ (with fix for add_argparse_args removal)
"""

# Ortam değişkenlerini TORCH/CUDA inisyalizasyonundan önce ver
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"

import importlib
from argparse import ArgumentParser
import random
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    TQDMProgressBar,
    DeviceStatsMonitor,
    ModelCheckpoint,  # <-- eklendi
)
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from models import module

# Deterministik ayarlar
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)  # nondeterministik op kullanılırsa hata fırlatır
# TF32 kapat (sayısal farklılıkları azaltır)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# CUDA memory management için
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

if __name__ == '__main__':
    # -------------------------------- #
    # ARGUMENT PARSING
    # -------------------------------- #
    parser = ArgumentParser()

    # Program specific
    parser.add_argument('--log_dir', type=str, help='Directory to which experiment logs will be written', required=True)
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--dataset', type=str, help='Dataset module', required=True)

    # Geçici olarak bilinen argümanları ayıkla
    program_args, _ = parser.parse_known_args()

    # Model specific
    parser = module.get_model_def().add_model_specific_args(parser)

    # Data module specific
    data_module = importlib.import_module(f'datasets.{program_args.dataset}')
    parser = data_module.get_datamodule_def().add_datamodule_specific_args(parser)

    # --- YENİ BÖLÜM: Trainer argümanlarını manuel ekleme ---
    # pl.Trainer.add_argparse_args kaldırıldığı için temel Trainer argümanlarını elle ekliyoruz.
    parser.add_argument('--accelerator', type=str, default='auto', help='Accelerator to use (e.g., "cpu", "gpu", "auto")')
    parser.add_argument('--devices', default=1, help='Number of devices to use (e.g., 1, 8, "auto")')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs to train for')
    parser.add_argument('--min_epochs', type=int, default=1, help='Minimum number of epochs to train for')
    parser.add_argument('--fast_dev_run', type=bool, default=False, help='Run a single batch for train, val, and test to check for errors')
    parser.add_argument('--overfit_batches', type=float, default=0.0, help='Overfit on a percentage of training data')
    parser.add_argument('--val_check_interval', type=float, default=1.0, help='How often to check the validation set')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Accumulates grads every k batches')
    parser.add_argument('--gradient_clip_val', type=float, default=0.0, help='Gradient clipping value')
    parser.add_argument('--profiler', type=str, default=None, help='Profiler to use (e.g., "simple", "advanced")')
    # İsteğe bağlı bayrak: Trainer(deterministic=...)
    parser.add_argument('--deterministic', type=lambda x: str(x).lower() in ['1','true','yes'], default=True,
                        help='Force deterministic behavior (default: True)')
    
    # Eski/kullanılmayan argümanlar
    parser.add_argument('--log_gpu_memory', action='store_true', help='DEPRECATED: Now controlled by DeviceStatsMonitor callback.')
    parser.add_argument('--progress_bar_refresh_rate', type=int, default=1, help='DEPRECATED: Use TQDMProgressBar callback.')
    
    args = parser.parse_args()
    # -------------------------------- #
    # SETUP
    # -------------------------------- #
    # Tüm RNG'ler için seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    pl.seed_everything(args.seed, workers=True)

    # Matmul precision (deterministik/fikst sonuçlar için "highest")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('highest')

    # 1. Logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.model
    )

    # 2. Callback'ler
    checkpoint_cb = ModelCheckpoint(
        monitor='val_loss',      # en iyi modeli val_loss'a göre seç
        mode='min',
        save_top_k=1,            # en iyi 1 modeli tut
        save_last=True,          # son epoch'u da ayrıca kaydetmek isterseniz
        filename='min-val-loss',
        auto_insert_metric_name=False
    )
    callbacks = [
        EarlyStopping(monitor='val_loss', mode='min', verbose=True, patience=7),
        LearningRateMonitor(logging_interval='epoch'),
        TQDMProgressBar(refresh_rate=1),
        checkpoint_cb,           # <-- eklendi
    ]

    # 3. Trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        fast_dev_run=args.fast_dev_run,
        overfit_batches=args.overfit_batches,
        val_check_interval=args.val_check_interval,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        profiler=args.profiler,
        callbacks=callbacks,
        logger=logger,
        deterministic=args.deterministic
    )

    # -------------------------------- #
    # FITTING THE MODEL
    # -------------------------------- #
    dict_args = vars(args)

    # Model ve DataModule, kendi init metotlarında **dict_args alabilir,
    # çünkü genellikle bilinmeyen argümanları görmezden gelecek şekilde yazılırlar.
    # try:
    #     model = module.get_model_def().load_from_checkpoint("/home/omer/Masaüstü/tez_calismasi/codebase/ChaLearn-2021-LAP/logs/VTN_HCPF/version_40/checkpoints/epoch=15-step=14080.ckpt", strict=False)
    # except Exception as e:
    # print(f"Model loading error: {e}")
    model = module.get_model(**dict_args)
    dm = data_module.get_datamodule(**dict_args)

    trainer.fit(model, datamodule=dm)