import yaml
import subprocess
import argparse
import sys
import os
import time
import csv
from collections import Counter, defaultdict

from prediction_analiz import calculate_accuracy

# predict.py'nin kabul ettiği argümanların bir listesini oluşturun.
# Bu liste predict.py dosyasındaki parser.add_argument çağrılarından alınmalıdır.
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
    CSV'den {id: label} okur.
    Beklenen: en az iki kolon (id, label). Header varsa otomatik atlar.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV bulunamadı: {path}")

    out = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return out

    # Header sezgisel kontrol: ikinci kolon sayısal değilse vs. (robust tutmak için)
    start_idx = 0
    if len(rows[0]) >= 2:
        # "label", "class", "target" gibi kelimeler içeriyorsa header say
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
    sklearn yoksa macro-F1'ı manuel hesaplar.
    y_true/y_pred: aynı uzunlukta label listeleri (string olabilir).
    """
    # Sınıf kümesi: true union pred
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
    true/pred CSV'leri okuyup ortak id'ler üzerinde macro-F1 döner.
    """
    true_map = _read_id_label_csv(true_labels_csv)
    pred_map = _read_id_label_csv(predicted_labels_csv)

    common_ids = [sid for sid in true_map.keys() if sid in pred_map]
    if not common_ids:
        raise ValueError(
            f"Ortak örnek id bulunamadı. true: {len(true_map)} pred: {len(pred_map)}"
        )

    y_true = [true_map[sid] for sid in common_ids]
    y_pred = [pred_map[sid] for sid in common_ids]

    # sklearn varsa kullan; yoksa manuel
    try:
        from sklearn.metrics import f1_score
        return float(f1_score(y_true, y_pred, average="macro"))
    except Exception:
        return float(_macro_f1_score(y_true, y_pred))


def count_predictions(predicted_labels_csv: str) -> int:
    """
    predictions.csv içindeki örnek sayısını döndürür (id->label map boyutu).
    """
    pred_map = _read_id_label_csv(predicted_labels_csv)
    return len(pred_map)


def run_from_config(config_path, checkpoint_path, out_path):
    """
    Yapılandırmayı bir YAML dosyasından yükler ve predict.py betiğini çalıştırır.
    Dönüş: elapsed_seconds (float)
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Hata: Yapılandırma dosyası bulunamadı: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Hata: YAML dosyası ayrıştırılırken hata oluştu: {e}")
        sys.exit(1)

    # Komutu oluşturmaya başla
    command = [sys.executable, '-m', 'predict']

    # Yapılandırma dosyasındaki argümanları, sadece predict.py'nin tanıgrounds ekle
    for key, value in config.items():
        if key in PREDICT_ARGS:  # Sadece bilinen argümanları ekle
            if key == 'norm_first':
                # Değer 'False' (string olarak) ise --no-norm-first bayrağını ekle
                if str(value).lower() == 'false':
                    command.append('--no-norm-first')
                # Değer 'True' ise hiçbir şey ekleme (varsayılan davranış)
                continue
            command.append(f'--{key}')
            command.append(str(value))

    # Gömülü parametreleri ekle
    command.extend([
        '--submission_template', '/home/omer/Masaüstü/tez_calismasi/codebase/ChaLearn-2021-LAP/predictions_test_template.csv',
        '--out', out_path,
        '--checkpoint', checkpoint_path
    ])

    print("Aşağıdaki komut çalıştırılıyor:")
    print(' '.join(command))

    # Komutu çalıştır
    try:
        t0 = time.perf_counter()
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        print("Betiğin çıktısı:\n", result.stdout)
        return elapsed
    except subprocess.CalledProcessError as e:
        print(f"Betiği çalıştırırken bir hata oluştu: {e}")
        print(f"Hata Çıktısı (stderr):\n{e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("Hata: 'python -m predict' komutu bulunamadı. Ortamınızın doğru ayarlandığından emin olun.")
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YAML yapılandırma dosyasından tahmin betiğini çalıştırır.')
    parser.add_argument(
        '--version',
        type=str,
        default='0',
        help='Çalıştırma için kullanılacak yapılandırma YAML dosyasının yolu.'
    )

    true_labels_file = "/home/omer/Masaüstü/tez_calismasi/codebase/ChaLearn-2021-LAP/predictions/RGB_TEST_VTN_HCPF.csv"  # Doğru etiketler
    predicted_labels_file = "/home/omer/Masaüstü/tez_calismasi/codebase/ChaLearn-2021-LAP/src/predictions.csv"  # Tahmin etiketler

    args = parser.parse_args()
    config_file = f'/home/omer/Masaüstü/tez_calismasi/codebase/ChaLearn-2021-LAP/logs/run_methods/VTN_HCPF/version_{args.version}/hparams.yaml'
    checkpoint_path = f'/home/omer/Masaüstü/tez_calismasi/codebase/ChaLearn-2021-LAP/logs/run_methods/VTN_HCPF/version_{args.version}/checkpoints/min-val-loss.ckpt'
    checkpoint_path_last = f'/home/omer/Masaüstü/tez_calismasi/codebase/ChaLearn-2021-LAP/logs/run_methods/VTN_HCPF/version_{args.version}/checkpoints/last.ckpt'

    # 1) min-val-loss
    elapsed_min = run_from_config(config_file, checkpoint_path, predicted_labels_file)
    accuracy_min_val_loss = calculate_accuracy(true_labels_file, predicted_labels_file)
    f1_min_val_loss = calculate_f1_macro(true_labels_file, predicted_labels_file)
    n_min = count_predictions(predicted_labels_file)
    avg_inf_min = (elapsed_min / n_min) if n_min > 0 else None

    # 2) last
    elapsed_last = run_from_config(config_file, checkpoint_path_last, predicted_labels_file)
    accuracy_last = calculate_accuracy(true_labels_file, predicted_labels_file)
    f1_last = calculate_f1_macro(true_labels_file, predicted_labels_file)
    n_last = count_predictions(predicted_labels_file)
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

    # --- LOGLAMA BÖLÜMÜ (YAML) ---
    log_dir = "/home/omer/Masaüstü/tez_calismasi/codebase/ChaLearn-2021-LAP/test_logs-f1"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"test_log_version_{args.version}.yaml")

    try:
        with open(config_file, 'r') as f:
            config_params = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Loglama için hparams.yaml dosyası bulunamadı: {config_file}")
        config_params = {}

    # YAML dosyası için veri yapısını oluştur
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

    # hparams.yaml'dan gelen parametreleri ekle
    for key, value in config_params.items():
        if key in PREDICT_ARGS:
            log_data['parameters'][key] = value

    # Koda gömülü parametreleri de ekle
    log_data['parameters']['submission_template'] = '/home/omer/Masaüstü/tez_calismasi/codebase/ChaLearn-2021-LAP/predictions_test_template.csv'
    log_data['parameters']['out'] = predicted_labels_file

    # Veri yapısını YAML dosyasına yaz
    with open(log_file_path, 'w') as log_file:
        yaml.dump(log_data, log_file, default_flow_style=False, sort_keys=False, indent=4)

    print("\n-----------------------------------------------------")
    print(f"Test sonuçları ve parametreler şuraya kaydedildi: {log_file_path}")