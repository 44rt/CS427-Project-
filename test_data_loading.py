# test_data_loading.py
from utils.data_loader import load_and_preprocess_data, create_incremental_tasks

print("🔍 Testing data loading...")
df = load_and_preprocess_data()
print(f"✅ Data loaded. Shape: {df.shape}")
print(f"   Columns: {len(df.columns)}")
print(f"   Label counts: \n{df['Label'].value_counts()}")
print(f"   Attack types: \n{df['Attack Type'].value_counts()}")

print("\n🔍 Testing incremental task creation...")
tasks, task_test_sets, global_test_set = create_incremental_tasks()
print(f"✅ Tasks created: {len(tasks)}")

for i, (X_train, y_train, X_val, y_val) in enumerate(tasks):
    print(f"   Task {i}: X_train.shape={X_train.shape}")
    print(f"           y_train distribution: {y_train.value_counts().to_dict()}")

X_global_test, y_global_test = global_test_set
print(f"✅ Global test set: {X_global_test.shape}")

print("\n🎉 Data loading test passed! ✅")