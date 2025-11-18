"""PyTorch Lightning module definition.
Delegates computation to one of the defined networks (vtn.py, vtn_hc.py, vtn_hcpf.py)"""
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import StepLR
import torchmetrics

from .vtn_hcpf import VTNHCPF
from .vtn_hcpf_d import VTNHCPFD


def get_model_def():
    return Module


def get_model(**kwargs):
    return Module(**kwargs)


class Module(pl.LightningModule):

    def __init__(self, model, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        NUM_CLASSES = 226
        if self.hparams.encoder_history_type == 'none':
            self.encoder_history_type = None
        else:
            self.encoder_history_type = self.hparams.encoder_history_type
        if model == 'VTN_HCPF':
            self.model = VTNHCPF(NUM_CLASSES, self.hparams.num_heads, self.hparams.num_layers, self.hparams.embed_size,
                                 self.hparams.sequence_length, self.hparams.cnn,
                                 self.hparams.freeze_layers,
                                 self.hparams.dropout,  norm_first=self.hparams.norm_first,  enc_calculate_num=self.hparams.enc_calculate_num, rk_type=self.hparams.rk_type, encoder_history_type=self.encoder_history_type)
        elif model == 'VTN_HCPF_D':
            self.model = VTNHCPFD(NUM_CLASSES, self.hparams.num_heads, self.hparams.num_layers, self.hparams.embed_size,
                                  self.hparams.sequence_length, self.hparams.cnn,
                                  self.hparams.freeze_layers,
                                  self.hparams.dropout)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro")
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro")
        self.train_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES, top_k=5)
        self.val_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES, top_k=5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = self.criterion(z, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', self.train_accuracy(z, y), prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = self.criterion(z, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', self.val_accuracy(z, y), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        # StepLR yerine val_loss izleyen ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=2,
            threshold=1e-4,
            cooldown=3,
            min_lr=0
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate,
    #                                  weight_decay=self.hparams.weight_decay)
    #     scheduler = StepLR(optimizer, step_size=self.hparams.lr_step_size, gamma=0.1)
    #     return [optimizer], [{"scheduler": scheduler, "monitor": "val_accuracy"}]
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--num_heads', type=int, default=4)
        parser.add_argument('--num_layers', type=int, default=4)
        parser.add_argument('--embed_size', type=int, default=512)
        parser.add_argument('--cnn', type=str, default='rn18')
        parser.add_argument('--freeze_layers', type=int, default=0,
                            help='Freeze all CNN layers up to this index (default: 0, no frozen layers)')
        parser.add_argument('--weight_decay', type=float, default=0)
        parser.add_argument('--dropout', help='Dropout before MHA and FC', type=float, default=0)
        parser.add_argument('--lr_step_size', type=int, default=5)
        parser.add_argument('--model', type=str, required=True)
        parser.add_argument('--no-norm-first', dest='norm_first', action='store_false', help='Disable Transformer normalization first (default: enabled)')
        parser.add_argument('--enc_calculate_num', type=int, default=1, help='Number of calculations in encoder ODE')
        parser.add_argument('--rk_type', type=str, default="none", help='Runge-Kutta type for ODE transformer (standard, initialization, learnable, none)')
        parser.add_argument('--encoder_history_type', type=str, default=None, help='Type of history for encoder ODE transformer (None, dense, residual)')
        return parser
