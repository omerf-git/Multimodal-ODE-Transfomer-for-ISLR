"""
Train a neural network on a given dataset with fixed hyperparameters.
For tuning of the hyperparameters, see tune.py.

Updated for PyTorch Lightning 2.2.2+ (with fix for add_argparse_args removal)
"""

import importlib
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    TQDMProgressBar,
    DeviceStatsMonitor
)
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from models import module

torch.autograd.set_detect_anomaly(True)

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
    
    # Eski/kullanılmayan argümanlar
    parser.add_argument('--log_gpu_memory', action='store_true', help='DEPRECATED: Now controlled by DeviceStatsMonitor callback.')
    parser.add_argument('--progress_bar_refresh_rate', type=int, default=1, help='DEPRECATED: Use TQDMProgressBar callback.')
    
    args = parser.parse_args()
# -------------------------------- #
# SETUP
# -------------------------------- #
pl.seed_everything(args.seed, workers=True)

# 1. Logger'ı doğru argümanlarla oluştur
# TensorBoardLogger, logların kaydedileceği ana dizin olarak 'save_dir' bekler.
logger = TensorBoardLogger(
    save_dir=args.log_dir,
    name=args.model
)

# 2. Callback'leri oluştur
callbacks = [
    EarlyStopping(monitor='val_loss', mode='min', verbose=True, patience=10),
    LearningRateMonitor(logging_interval='epoch'),
    TQDMProgressBar(refresh_rate=1) # Refresh rate'i burada sabit veya argüman olarak verebilirsiniz
]
# GPU bellek takibini ekle (isteğe bağlı)
# ...

# 3. Trainer'ı SADECE kendi argümanları ve nesneleriyle başlat
# **vars(args) KULLANMIYORUZ! Çünkü 'log_dir', 'model' gibi
# Trainer'a ait olmayan argümanlar hata veriyor.
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
    callbacks=callbacks,  # Callback nesnelerini buraya
    logger=logger         # Logger nesnesini buraya veriyoruz
)

# -------------------------------- #
# FITTING THE MODEL
# -------------------------------- #
dict_args = vars(args)

# Model ve DataModule, kendi init metotlarında **dict_args alabilir,
# çünkü genellikle bilinmeyen argümanları görmezden gelecek şekilde yazılırlar.
model = module.get_model(**dict_args)
dm = data_module.get_datamodule(**dict_args)

trainer.fit(model, datamodule=dm)