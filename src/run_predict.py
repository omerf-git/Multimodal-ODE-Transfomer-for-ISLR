import yaml
import subprocess
import argparse
import sys
import os
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


def run_from_config(config_path, checkpoint_path):
    """
    Yapılandırmayı bir YAML dosyasından yükler ve predict.py betiğini çalıştırır.
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

    # Yapılandırma dosyasındaki argümanları, sadece predict.py'nin tanıdıklarını ekle
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
        '--out', 'predictions.csv',
        '--checkpoint', checkpoint_path
    ])

    print("Aşağıdaki komut çalıştırılıyor:")
    print(' '.join(command))
    
    # Komutu çalıştır
    try:
        # Hata durumunda daha fazla çıktı almak için capture_output ve text ekleyin
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Betiğin çıktısı:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Betiği çalıştırırken bir hata oluştu: {e}")
        print(f"Hata Çıktısı (stderr):\n{e.stderr}") # Hata mesajını detaylı göster
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

    run_from_config(config_file, checkpoint_path)

    accuracy_min_val_loss = calculate_accuracy(true_labels_file, predicted_labels_file)

    run_from_config(config_file, checkpoint_path_last)

    accuracy_last = calculate_accuracy(true_labels_file, predicted_labels_file)
    print("\n")
    print("-----------------------------------------------------")
    print("\n")
    print(f"Accuracy (min-val-loss checkpoint): {accuracy_min_val_loss:.4f}")
    print(f"Accuracy (last checkpoint): {accuracy_last:.4f}")

    # --- LOGLAMA BÖLÜMÜ (YAML) ---
    log_dir = "/home/omer/Masaüstü/tez_calismasi/codebase/ChaLearn-2021-LAP/test_logs"
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
            'accuracy_last': float(f"{accuracy_last:.4f}")
        }
    }

    # hparams.yaml'dan gelen parametreleri ekle
    for key, value in config_params.items():
        if key in PREDICT_ARGS:
            log_data['parameters'][key] = value
    
    # Koda gömülü parametreleri de ekle
    log_data['parameters']['submission_template'] = '/home/omer/Masaüstü/tez_calismasi/codebase/ChaLearn-2021-LAP/predictions_test_template.csv'
    log_data['parameters']['out'] = 'predictions.csv'

    # Veri yapısını YAML dosyasına yaz
    with open(log_file_path, 'w') as log_file:
        yaml.dump(log_data, log_file, default_flow_style=False, sort_keys=False, indent=4)

    print("\n-----------------------------------------------------")
    print(f"Test sonuçları ve parametreler şuraya kaydedildi: {log_file_path}")