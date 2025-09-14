# 5G IDS Continuous Learning Project

This project implements and compares continuous learning techniques for 5G Intrusion Detection Systems.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

---

test the test files

# First, test if your data is loaded correctly

python test_data_loading.py

# Then test the metrics calculations

python test_metrics.py

# Then test the baseline models work

python test_baselines.py

# Then test the MLP model works

python test_mlp.py

---

when it works try the main ones

# Run the full baseline experiment

python 01_train_baselines.py

# Run the full MLP experiment

python 02_train_mlp.py
