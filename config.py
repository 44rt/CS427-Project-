# # config.py
# DATA_PATH = "data/Encoded.csv"
# RANDOM_STATE = 42
# TEST_SIZE = 0.2
# VAL_SIZE = 0.2

# # Define the order of introduction for new attacks
# TASK_ATTACK_ORDER = [
#     ['DDoS'],        # Task 0: Introduce DDoS
#     ['PortScan'],    # Task 1: Introduce PortScan  
#     ['Botnet']       # Task 2: Introduce Botnet
# ]

# # Model parameters
# INPUT_SIZE = 100  # Update this based on your actual data shape
# BATCH_SIZE = 32
# LEARNING_RATE = 0.001
# EPOCHS = 10

# config.py
DATA_PATH = "data/Encoded.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2

# UPDATE THIS WITH YOUR REAL ATTACK TYPES:
TASK_ATTACK_ORDER = [
    ['UDPFlood'],        # Task 0: Most common attack
    ['HTTPFlood'],       # Task 1: Second most common
    ['SlowrateDoS'],     # Task 2: Third most common
    ['TCPConnectScan'], # Task 3
    ['SYNScan'],        # Task 4UDPScan: 15,906 samples
    ['SYNFlood'],
    ['ICMPFlood'],
]

# Model parameters
INPUT_SIZE = 88  # Update this based on your debug output (88 features)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10