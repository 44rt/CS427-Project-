# # # config.py
# # DATA_PATH = "data/Encoded.csv"
# # RANDOM_STATE = 42
# # TEST_SIZE = 0.2
# # VAL_SIZE = 0.2

# # # Define the order of introduction for new attacks
# # TASK_ATTACK_ORDER = [
# #     ['DDoS'],        # Task 0: Introduce DDoS
# #     ['PortScan'],    # Task 1: Introduce PortScan  
# #     ['Botnet']       # Task 2: Introduce Botnet
# # ]

# # # Model parameters
# # INPUT_SIZE = 100  # Update this based on your actual data shape
# # BATCH_SIZE = 32
# # LEARNING_RATE = 0.001
# # EPOCHS = 10
# ################################################################
# # # config.py
# # DATA_PATH = "data/Encoded.csv"
# # RANDOM_STATE = 42
# # TEST_SIZE = 0.2
# # VAL_SIZE = 0.2

# # # UPDATE THIS WITH YOUR REAL ATTACK TYPES:
# # TASK_ATTACK_ORDER = [
# #     ['UDPFlood'],        # Task 0: Most common attack
# #     ['HTTPFlood'],       # Task 1: Second most common
# #     ['SlowrateDoS'],     # Task 2: Third most common
# #     ['TCPConnectScan'],  # Task 3
# #     ['SYNScan'],         # Task 4UDPScan: 15,906 samples
# #     ['SYNFlood'],
# #     ['ICMPFlood'],
# # ]

# # # Model parameters
# # INPUT_SIZE = 88  # Update this based on your debug output (88 features)
# # BATCH_SIZE = 32
# # LEARNING_RATE = 0.001
# # EPOCHS = 10

# # config.py
# import os

# # ====== RUNTIME CONFIGURATION ======
# # AUTO-DETECT ENVIRONMENT
# IS_COLAB = 'COLAB_GPU' in os.environ
# IS_LOCAL = not IS_COLAB

# # QUICK TESTING (for local) vs FULL RUN (for Colab)
# QUICK_TEST = True  # Set to False when running on Colab

# # MODEL SETTINGS (adjust based on environment)
# ENABLE_SVM = False and IS_LOCAL  # Disable SVM for local testing

# # ====== DATA SAMPLE SETTINGS ======
# if QUICK_TEST:
#     SAMPLE_FRACTION = 0.01  # 1% for local testing
#     print(" LOCAL TEST MODE: Using 1% of data")
# else:
#     SAMPLE_FRACTION = 1.0   # 100% for Colab
#     print(" COLAB FULL RUN: Using 100% of data")

# # ====== MODEL PARAMETERS ======
# BATCH_SIZE = 128 if IS_COLAB else 32
# EPOCHS = 20 if IS_COLAB else 5  # Fewer epochs for local testing

# # ====== ATTACK CONFIGURATION ======
# TASK_ATTACK_ORDER = [
#     ['UDPFlood'],        
#     ['HTTPFlood'],       
#     ['SlowrateDoS'],
#     # Remove comments for full run on Colab:
#     # ['TCPConnectScan'],
#     # ['SYNScan'],
# ]

# # ====== PATHS ======
# DATA_PATH = "data/Encoded.csv"
# RANDOM_STATE = 42
# TEST_SIZE = 0.2
# VAL_SIZE = 0.2

# print(f" Environment: {'Colab' if IS_COLAB else 'Local'}")
# print(f" Sample Fraction: {SAMPLE_FRACTION}")
# print(f" Batch Size: {BATCH_SIZE}")
# print(f" Epochs: {EPOCHS}")


# # config.py - COMPLETE VERSION
# import os

# # ====== RUNTIME CONFIGURATION ======
# IS_COLAB = 'COLAB_GPU' in os.environ
# IS_LOCAL = not IS_COLAB
# QUICK_TEST = True
# ENABLE_SVM = True and IS_LOCAL
# ENABLE_CNN_RNN = True

# # ====== DATA SAMPLE SETTINGS ======
# SAMPLE_FRACTION = 0.01 if QUICK_TEST else 1.0

# # ====== PATHS AND CONSTANTS ======
# DATA_PATH = "data/Encoded.csv"
# RANDOM_STATE = 42
# TEST_SIZE = 0.2
# VAL_SIZE = 0.2
# BATCH_SIZE = 128 if IS_COLAB else 32
# EPOCHS = 20 if IS_COLAB else 5
# LEARNING_RATE = 0.001
# # INPUT_SIZE = 88
# # config.py
# # Model Configuration
# INPUT_SIZE = None  # Will be set dynamically
# HIDDEN_DIMS = [128, 64, 32]
# OUTPUT_SIZE = 2  # Binary classification (adjust if needed)

# # Training Configuration
# LEARNING_RATE = 0.001
# EPOCHS = 10
# BATCH_SIZE = 32

# # Environment Configuration
# SAMPLE_FRACTION = 0.01
# DATA_PATH = "data/Encoded.csv"

# # ====== ATTACK CONFIGURATION ======
# TASK_ATTACK_ORDER = [
#     ['UDPFlood'],        
#     ['HTTPFlood'],       
#     ['SlowrateDoS'],
# ]

# print(f" Configuration loaded:")
# print(f" Environment: {'Colab' if IS_COLAB else 'Local'}")
# print(f" Sample Fraction: {SAMPLE_FRACTION}")
# print(f" Data path: {DATA_PATH}")

# # config.py - UPDATE THE ATTACK NAMES SECTION
# # ====== ATTACK NAME MAPPING ======
# ATTACK_NAMES = {
#     0: "UDPFlood",
#     1: "HTTPFlood", 
#     2: "SlowrateDoS",
# }

# def get_attack_name(task_id):
#     """Get the actual attack name for a task ID"""
#     return ATTACK_NAMES.get(task_id, f"Task_{task_id}")

# config.py - CLEANED VERSION
import os
import torch

# ====== RUNTIME CONFIGURATION ======
IS_COLAB = 'COLAB_GPU' in os.environ
IS_LOCAL = not IS_COLAB
QUICK_TEST = True
ENABLE_SVM = True and IS_LOCAL
ENABLE_CNN_RNN = True

# ====== DATA SAMPLE SETTINGS ======
SAMPLE_FRACTION = 0.01 if QUICK_TEST else 1.0

# ====== PATHS AND CONSTANTS ======
DATA_PATH = "data/Encoded.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2
BATCH_SIZE = 64  # Reduced for stability
EPOCHS = 10 if IS_COLAB else 5
LEARNING_RATE = 1e-5  # SIGNIFICANTLY REDUCED

# ====== MODEL CONFIGURATION ======
INPUT_SIZE = None  # Will be set dynamically
HIDDEN_DIMS = [128, 64, 32]
OUTPUT_SIZE = 2  # Binary classification

# ====== ATTACK CONFIGURATION ======
TASK_ATTACK_ORDER = [
    ['UDPFlood'],        
    ['HTTPFlood'],       
    ['SlowrateDoS'],
]

# ====== ATTACK NAME MAPPING ======
ATTACK_NAMES = {
    0: "UDPFlood",
    1: "HTTPFlood", 
    2: "SlowrateDoS",
}

# ====== CONTINUAL LEARNING SETTINGS ======
EWC_LAMBDA = 1000  # Regularization strength for EWC
LWF_ALPHA = 0.5    # Knowledge distillation strength for LwF
MAX_TASKS = 50     # Maximum number of tasks to handle

def get_attack_name(task_id):
    """Get the actual attack name for a task ID"""
    return ATTACK_NAMES.get(task_id, f"Task_{task_id}")

print(f" Configuration loaded:")
print(f" Environment: {'Colab' if IS_COLAB else 'Local'}")
print(f" Sample Fraction: {SAMPLE_FRACTION}")
print(f" Data path: {DATA_PATH}")