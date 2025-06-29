import pandas as pd
from sklearn.metrics import accuracy_score

def calculate_accuracy(true_labels_csv, predicted_labels_csv):
    """
    İki CSV dosyasını karşılaştırarak accuracy hesaplar.
    
    Args:
        true_labels_csv: Doğru etiketleri içeren CSV dosyası yolu
        predicted_labels_csv: Tahmin etiketlerini içeren CSV dosyası yolu
    
    Returns:
        float: Accuracy değeri
    """
    
    # CSV dosyalarını oku (başlık yok)
    true_df = pd.read_csv(true_labels_csv, header=None, names=['sample_name', 'true_label'])
    pred_df = pd.read_csv(predicted_labels_csv, header=None, names=['sample_name', 'predicted_label'])
    
    # Tahmin CSV'sindeki sample'lar üzerinden birleştirme yap
    merged_df = pred_df.merge(true_df, on='sample_name', how='left')
    
    # Eşleşmeyen sample'ları kontrol et
    missing_samples = merged_df[merged_df['true_label'].isna()]
    if not missing_samples.empty:
        print(f"Uyarı: {len(missing_samples)} sample için doğru etiket bulunamadı:")
        print(missing_samples['sample_name'].tolist())
    
    # Eşleşen sample'ları filtrele
    valid_df = merged_df.dropna()
    
    if valid_df.empty:
        print("Hata: Hiçbir sample eşleşmedi!")
        return None
    
    # Accuracy hesapla
    accuracy = accuracy_score(valid_df['true_label'], valid_df['predicted_label'])
    
    print(f"Toplam tahmin edilen sample sayısı: {len(pred_df)}")
    print(f"Eşleşen sample sayısı: {len(valid_df)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy

# Kullanım örneği
if __name__ == "__main__":
    # CSV dosya yollarını belirtin
    true_labels_file = "/home/omer/Masaüstü/tez_calismasi/codebase/ChaLearn-2021-LAP/predictions/RGB_TEST_VTN_HCPF.csv"  # Doğru etiketler
    predicted_labels_file = "/home/omer/Masaüstü/tez_calismasi/codebase/ChaLearn-2021-LAP/src/predictions.csv"  # Tahmin etiketler
    
    # Accuracy hesapla
    accuracy = calculate_accuracy(true_labels_file, predicted_labels_file)