import yaml
import subprocess
import argparse
import sys
import os
import time
import csv
from collections import Counter, defaultdict

from prediction_analysis import calculate_accuracy

# Create a list of arguments accepted by predict.py
# This list should match parser.add_argument calls in predict.py
PREDICT_ARGS = {
    'log_dir', 'seed', 'dataset', 'checkpoint', 'submission_template', 'out',
    'learning_rate', 'num_heads', 'num_layers', 'embed_size', 'cnn',
    'freeze_layers', 'weight_decay', 'dropout', 'lr_step_size', 'model',
    'norm_first', 'enc_calculate_num', 'rk_type', 'encoder_history_type',
    'batch_size', 'num_workers', 'data_dir', 'sequence_length',
    'temporal_stride', 'accelerator', 'strategy', 'devices', 'precision',
    'max_epochs', 'enable_checkpointing', 'enable_progress_bar',
    'enable_model_summary'
}


def _read_id_label_csv(path: str):
    """
    Reads {id: label} from CSV.
    Expected: at least two columns (id, label). Automatically skips header if present.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    out = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return out

    # Header heuristic check: e.g. if second column is not numeric (to keep it robust)
    start_idx = 0
    if len(rows[0]) >= 2:
        # If it contains words like "label", "class", "target", count as header
        header_join = ",".join([c.strip().lower() for c in rows[0]])
        if any(k in header_join for k in ("label", "class", "target", "id", "sample")):
            start_idx = 1

    for r in rows[start_idx:]:
        if len(r) < 2:
            continue
        sid = str(r[0]).strip()
        lab = str(r[1]).strip()
        if sid == "":
            continue
        out[sid] = lab
    return out


def _macro_f1_score(y_true, y_pred):
    """
    Calculates macro-F1 manually if sklearn is not available.
    y_true/y_pred: label lists of same length (can be string).
    """
    # Class set: true union pred
    classes = sorted(set(y_true) | set(y_pred))
    if not classes:
        return 0.0

    tp = Counter()
    fp = Counter()
    fn = Counter()

    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    f1s = []
    for c in classes:
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    return sum(f1s) / len(f1s)


def calculate_f1_macro(true_labels_csv: str, predicted_labels_csv: str) -> float:
    """
    Reads true/pred CSVs and returns macro-F1 on common ids.
    """
    true_map = _read_id_label_csv(true_labels_csv)
    pred_map = _read_id_label_csv(predicted_labels_csv)

    common_ids = [sid for sid in true_map.keys() if sid in pred_map]
    if not common_ids:
        raise ValueError(
            f"No common sample ids found. true: {len(true_map)} pred: {len(pred_map)}"
        )

    y_true = [true_map[sid] for sid in common_ids]
    y_pred = [pred_map[sid] for sid in common_ids]

    # use sklearn if available; else manual
    try:
        from sklearn.metrics import f1_score
        return float(f1_score(y_true, y_pred, average="macro"))
    except Exception:
        return float(_macro_f1_score(y_true, y_pred))


def count_predictions(predicted_labels_csv: str) -> int:
    """
    Returns number of samples in predictions.csv (size of id->label map).
    """
    pred_map = _read_id_label_csv(predicted_labels_csv)
    return len(pred_map)


def run_from_config(config_path, checkpoint_path, out_path):
    """
    Loads config from a YAML file and runs the predict.py script.
    Returns: elapsed_seconds (float)
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file: {e}")
        sys.exit(1)

    # Start building command
    command = [sys.executable, '-m', 'predict']

    # Add arguments from configuration file that predict.py recognizes
    for key, value in config.items():
        if key in PREDICT_ARGS:  # Only add known arguments
            if key == 'norm_first':
                # Add --no-norm-first flag if value is 'False'
                if str(value).lower() == 'false':
                    command.append('--no-norm-first')
                # Do nothing if 'True' (default behavior)
                continue
            command.append(f'--{key}')
            command.append(str(value))

    # Add embedded parameters
    command.extend([
        '--submission_template', PREDICTIONS_TEST_TEMPLATE,
        '--out', out_path,
        '--checkpoint', checkpoint_path
    ])

    print("Running the following command:")
    print(' '.join(command))

    # Execute command
    try:
        t0 = time.perf_counter()
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        print("Script output:\n", result.stdout)
        return elapsed
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the script: {e}")
        print(f"Error Output (stderr):\n{e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'python -m predict' command not found. Ensure your environment is properly configured.")
        sys.exit(1)


if __name__ == '__main__':
    from config_paths import TRUE_LABELS_FILE, PREDICTED_LABELS_FILE, LOG_DIR, TEST_LOGS_DIR, PREDICTIONS_TEST_TEMPLATE

    parser = argparse.ArgumentParser(description='Runs the prediction script from a YAML configuration file.')
    parser.add_argument(
        '--version',
        type=str,
        default='0',
        help='Version of the configuration YAML file to use for running.'
    )

    args = parser.parse_args()
    config_file = os.path.join(LOG_DIR, f'VTN_HCPF/version_{args.version}/hparams.yaml')
    checkpoint_path = os.path.join(LOG_DIR, f'VTN_HCPF/version_{args.version}/checkpoints/min-val-loss.ckpt')
    checkpoint_path_last = os.path.join(LOG_DIR, f'VTN_HCPF/version_{args.version}/checkpoints/last.ckpt')

    # 1) min-val-loss
    elapsed_min = run_from_config(config_file, checkpoint_path, PREDICTED_LABELS_FILE)
    accuracy_min_val_loss = calculate_accuracy(TRUE_LABELS_FILE, PREDICTED_LABELS_FILE)
    f1_min_val_loss = calculate_f1_macro(TRUE_LABELS_FILE, PREDICTED_LABELS_FILE)
    n_min = count_predictions(PREDICTED_LABELS_FILE)
    avg_inf_min = (elapsed_min / n_min) if n_min > 0 else None

    # 2) last
    elapsed_last = run_from_config(config_file, checkpoint_path_last, PREDICTED_LABELS_FILE)
    accuracy_last = calculate_accuracy(TRUE_LABELS_FILE, PREDICTED_LABELS_FILE)
    f1_last = calculate_f1_macro(TRUE_LABELS_FILE, PREDICTED_LABELS_FILE)
    n_last = count_predictions(PREDICTED_LABELS_FILE)
    avg_inf_last = (elapsed_last / n_last) if n_last > 0 else None

    print("\n")
    print("-----------------------------------------------------")
    print("\n")
    print(f"Accuracy (min-val-loss checkpoint): {accuracy_min_val_loss:.4f}")
    print(f"F1-macro (min-val-loss checkpoint): {f1_min_val_loss:.4f}")
    if avg_inf_min is not None:
        print(f"Avg inference (s/sample) (min-val-loss checkpoint): {avg_inf_min:.6f}")

    print(f"Accuracy (last checkpoint): {accuracy_last:.4f}")
    print(f"F1-macro (last checkpoint): {f1_last:.4f}")
    if avg_inf_last is not None:
        print(f"Avg inference (s/sample) (last checkpoint): {avg_inf_last:.6f}")

    # --- LOGGING SECTION (YAML) ---
    os.makedirs(TEST_LOGS_DIR, exist_ok=True)
    log_file_path = os.path.join(TEST_LOGS_DIR, f"test_log_version_{args.version}.yaml")

    try:
        with open(config_file, 'r') as f:
            config_params = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"hparams.yaml not found for logging: {config_file}")
        config_params = {}

    # Create data structure for YAML file
    log_data = {
        'test_info': {
            'version': args.version
        },
        'parameters': {},
        'results': {
            'accuracy_min_val_loss': float(f"{accuracy_min_val_loss:.4f}"),
            'f1_macro_min_val_loss': float(f"{f1_min_val_loss:.4f}"),
            'elapsed_seconds_min_val_loss': float(f"{elapsed_min:.6f}"),
            'num_samples_min_val_loss': int(n_min),
            'avg_inference_seconds_per_sample_min_val_loss': (float(f"{avg_inf_min:.8f}") if avg_inf_min is not None else None),

            'accuracy_last': float(f"{accuracy_last:.4f}"),
            'f1_macro_last': float(f"{f1_last:.4f}"),
            'elapsed_seconds_last': float(f"{elapsed_last:.6f}"),
            'num_samples_last': int(n_last),
            'avg_inference_seconds_per_sample_last': (float(f"{avg_inf_last:.8f}") if avg_inf_last is not None else None),
        }
    }

    # Add parameters from hparams.yaml
    for key, value in config_params.items():
        if key in PREDICT_ARGS:
            log_data['parameters'][key] = value

    # Add embedded parameters
    log_data['parameters']['submission_template'] = PREDICTIONS_TEST_TEMPLATE
    log_data['parameters']['out'] = PREDICTED_LABELS_FILE

    # Write data structure to YAML file
    with open(log_file_path, 'w') as log_file:
        yaml.dump(log_data, log_file, default_flow_style=False, sort_keys=False, indent=4)

    print("\n-----------------------------------------------------")
    print(f"Test results and parameters saved to: {log_file_path}")