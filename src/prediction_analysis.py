import pandas as pd
from sklearn.metrics import accuracy_score

def calculate_accuracy(true_labels_csv, predicted_labels_csv):
    """
    Calculates accuracy by comparing two CSV files.
    
    Args:
        true_labels_csv: Path to the CSV file containing true labels
        predicted_labels_csv: Path to the CSV file containing predicted labels
    
    Returns:
        float: Accuracy score
    """
    
    # Read CSV files (no headers)
    true_df = pd.read_csv(true_labels_csv, header=None, names=['sample_name', 'true_label'])
    pred_df = pd.read_csv(predicted_labels_csv, header=None, names=['sample_name', 'predicted_label'])
    
    # Merge on sample_name based on the predictions CSV
    merged_df = pred_df.merge(true_df, on='sample_name', how='left')
    
    # Check for unmatched samples
    missing_samples = merged_df[merged_df['true_label'].isna()]
    if not missing_samples.empty:
        print(f"Warning: True label not found for {len(missing_samples)} samples:")
        print(missing_samples['sample_name'].tolist())
    
    # Filter out matched samples
    valid_df = merged_df.dropna()
    
    if valid_df.empty:
        print("Error: No samples matched!")
        return None
    
    # Calculate accuracy
    accuracy = accuracy_score(valid_df['true_label'], valid_df['predicted_label'])
    
    print(f"Total predicted samples: {len(pred_df)}")
    print(f"Matched samples: {len(valid_df)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy

# Usage example
if __name__ == "__main__":
    from config_paths import TRUE_LABELS_FILE, PREDICTED_LABELS_FILE
    
    # Calculate accuracy
    accuracy = calculate_accuracy(TRUE_LABELS_FILE, PREDICTED_LABELS_FILE)