import os

# Base project root (assumes this file is in src/ and project root is one level up)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Data directories
DATA_DIR_MP4 = os.path.join(PROJECT_ROOT, 'data', 'mp4')
DATA_DIR_KP = os.path.join(PROJECT_ROOT, 'data', 'kp')
DATA_DIR_KPFLOW = os.path.join(PROJECT_ROOT, 'data', 'kpflow2')

# Logs and outputs
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs', 'run_methods')
TEST_LOGS_DIR = os.path.join(PROJECT_ROOT, 'test_logs-f1')

# Predictions
PREDICTIONS_TEST_TEMPLATE = os.path.join(PROJECT_ROOT, 'predictions_test_template.csv')
TRUE_LABELS_FILE = os.path.join(PROJECT_ROOT, 'predictions', 'RGB_TEST_VTN_HCPF.csv')
PREDICTED_LABELS_FILE = os.path.join(PROJECT_ROOT, 'src', 'predictions.csv')
