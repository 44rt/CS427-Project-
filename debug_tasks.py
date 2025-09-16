# debug_tasks.py
from utils.data_loader import create_incremental_tasks
import numpy as np

print("🔍 Debugging task creation...")
tasks, task_test_sets, global_test_set = create_incremental_tasks()

for i, ((X_train, y_train, X_val, y_val), (X_test, y_test)) in enumerate(zip(tasks, task_test_sets)):
    unique_train = np.unique(y_train)
    unique_test = np.unique(y_test)
    
    print(f"Task {i}:")
    print(f"  Training set: {X_train.shape}, Classes: {unique_train}")
    print(f"  Test set: {X_test.shape}, Classes: {unique_test}")
    print(f"  Class distribution - Train: {np.bincount(y_train.astype(int))}")
    print(f"  Class distribution - Test: {np.bincount(y_test.astype(int))}")
    print()

X_global, y_global = global_test_set
print(f"Global test set: {X_global.shape}")
print(f"Global classes: {np.unique(y_global)}")
print(f"Global class distribution: {np.bincount(y_global.astype(int))}")