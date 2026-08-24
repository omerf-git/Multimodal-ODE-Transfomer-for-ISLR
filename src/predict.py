"""Use a trained neural network to predict on a data set."""
import csv
import importlib
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch

from models import module

if __name__ == '__main__':
    # -------------------------------- #
    # ARGUMENT PARSING
    # -------------------------------- #
    parser = ArgumentParser()

    # Program specific
    parser.add_argument('--log_dir', type=str, help='Directory to which experiment logs will be written', required=True)
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--dataset', type=str, help='Dataset module', required=True)
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint', required=True)
    parser.add_argument('--submission_template', type=str, help='Path to the submission template', required=True)
    parser.add_argument('--out', type=str, help='Output file path', required=True)

    program_args, _ = parser.parse_known_args()

    # Model specific
    parser = module.get_model_def().add_model_specific_args(parser)

    # Data module specific
    data_module = importlib.import_module(f'datasets.{program_args.dataset}')
    parser = data_module.get_datamodule_def().add_datamodule_specific_args(parser)

    # Trainer specific arguments (manually added for PyTorch Lightning 2.x compatibility)
    parser.add_argument('--accelerator', type=str, default='auto', help='Accelerator type')
    parser.add_argument('--strategy', type=str, default='auto', help='Training strategy')
    parser.add_argument('--devices', default='auto', help='Number of devices')
    parser.add_argument('--precision', default='32-true', help='Precision type')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--enable_checkpointing', type=bool, default=True, help='Enable checkpointing')
    parser.add_argument('--enable_progress_bar', type=bool, default=True, help='Enable progress bar')
    parser.add_argument('--enable_model_summary', type=bool, default=True, help='Enable model summary')

    args = parser.parse_args()

    # -------------------------------- #
    # SETUP
    # -------------------------------- #
    # Seed setting for PyTorch Lightning 2.x
    pl.seed_everything(args.seed, workers=True)

    dict_args = vars(args)

    # Model loading - PyTorch Lightning 2.x compatible
    try:
        model = module.get_model_def().load_from_checkpoint(args.checkpoint, strict=False)
    except Exception as e:
        print(f"Model loading error: {e}")
        # Fallback to strict=False for compatibility
        model = module.get_model_def().load_from_checkpoint(args.checkpoint, strict=False)

    dm = data_module.get_datamodule(**dict_args)

    # Device handling for PyTorch 2.x
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        print("Using CPU device")

    model.to(device)
    model.eval()

    # -------------------------------- #
    # USING MODEL TO PREDICT
    # -------------------------------- #
    submission = dict()

    # Setup dataloader
    dm.setup('test')
    dataloader = dm.test_dataloader()
    
    print(f"Starting prediction on {len(dataloader)} batches...")
    
    # Use inference_mode for PyTorch 2.x (more efficient)
    with torch.inference_mode():
        for i, batch in enumerate(dataloader):
            if i % 10 == 0:
                print(f"Processing batch {i}/{len(dataloader)}")
                
            x, paths = batch
            if isinstance(x, list):
                # Move list elements to device
                x_device = [e.to(device, non_blocking=True) for e in x]
                logits = model(x_device).cpu()
            else:
                # Move single tensor to device
                x_device = x.to(device, non_blocking=True)
                logits = model(x_device).cpu()

            predictions = torch.argmax(logits, dim=1)
            for j in range(logits.size(0)):
                submission[paths[j]] = predictions[j].item()

    print(f"Generated predictions for {len(submission)} samples")

    # Safer encoding for CSV processing
    try:
        with open(args.submission_template, 'r', encoding='utf-8') as stf:
            reader = csv.reader(stf)
            with open(args.out, 'w', encoding='utf-8', newline='') as of:
                writer = csv.writer(of)
                written_count = 0
                for row in reader:
                    sample = row[0]
                    if sample in submission:
                        print(f'Predicting {sample} as {submission[sample]}')
                        writer.writerow([sample, submission[sample]])
                        written_count += 1
                    else:
                        print(f'Warning: {sample} not found in predictions')
                
                print(f"Written {written_count} predictions to file")
    except Exception as e:
        print(f"Error writing submission file: {e}")
        raise

    print(f'Successfully wrote submission to {args.out}')
