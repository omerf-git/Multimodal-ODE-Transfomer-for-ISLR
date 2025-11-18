import subprocess
import sys
import re
from pathlib import Path

def parse_config(config_path: Path) -> dict:
    """
    Basit bir .sh yapılandırma dosyasını ayrıştırır ve bir sözlük döndürür.
    'KEY="VALUE"' veya 'KEY=VALUE' formatını destekler.
    """
    if not config_path.is_file():
        print(f"Hata: Yapılandırma dosyası bulunamadı: {config_path}")
        sys.exit(1)

    config = {}
    # Değişken atamalarını bulmak için regex: KEY=VALUE veya KEY="VALUE"
    pattern = re.compile(r'^\s*([\w_]+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s#]+))')

    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            match = pattern.match(line)
            if match:
                key = match.group(1)
                # Eşleşen değeri bul (tırnaklı veya tırnaksız)
                value = next((g for g in match.groups()[1:] if g is not None), None)
                config[key] = value
    return config

def main():
    """
    Yapılandırmayı okur ve eğitim betiğini çalıştırır.
    """
    # Betiğin bulunduğu dizini al
    base_dir = Path(__file__).parent
    config_path = base_dir / 'config.sh'

    # 1. Yapılandırmayı ayrıştır
    config = parse_config(config_path)

    # 2. Komut satırı argümanlarını oluştur
    # Temel komut (src dizininden çalıştırıldığını varsayarak)
    command = ['python', '-m', 'train']

    # Yapılandırmadaki her bir anahtar/değer çifti için argüman ekle
    for key, value in config.items():
        # NORM_FIRST için özel durum yönetimi
        if key == 'NORM_FIRST':
            # Değer 'False' (string olarak) ise --no-norm-first bayrağını ekle
            if str(value).lower() == 'false':
                command.append('--no-norm-first')
            # Değer 'True' ise hiçbir şey ekleme (varsayılan davranış)
            continue

        # Diğer tüm parametreleri ekle
        arg_name = f'--{key.lower()}'
        command.append(arg_name)
        command.append(str(value))

    # Oluşturulan komutu ekrana yazdır
    print("Çalıştırılan Komut:")
    # Daha iyi okunabilirlik için komutu birleştirip yazdır
    print(' '.join(command))
    print("-----------------------------------------------------")

    # 3. Komutu çalıştır
    try:
        # subprocess.run, komut tamamlanana kadar bekler
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Eğitim sırasında bir hata oluştu. Çıkış kodu: {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("Hata: 'python' komutu bulunamadı. Lütfen Python'un kurulu ve PATH'de olduğundan emin olun.")
        sys.exit(1)

if __name__ == '__main__':
    main()