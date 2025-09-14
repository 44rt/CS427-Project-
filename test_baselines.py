# test_baselines.py
from models.baseline_models import get_baseline_models
from utils.data_loader import create_incremental_tasks
from utils.metrics import evaluate_model

print("🔍 Testing baseline models...")
tasks, task_test_sets, global_test_set = create_incremental_tasks()

# Test with just the FIRST task and FIRST model for speed
X_train, y_train, X_val, y_val = tasks[0]
X_test, y_test = task_test_sets[0]

models = get_baseline_models()
model_name = "Perceptron"  # Test with just one model first
model = models[model_name]

print(f"🔍 Testing {model_name} on Task 0...")
if hasattr(model, 'partial_fit'):
    model.partial_fit(X_train, y_train, classes=[0, 1])
else:
    model.fit(X_train, y_train)

metrics = evaluate_model(model, X_test, y_test)
print(f"✅ {model_name} trained successfully:")
print(f"   Accuracy: {metrics['accuracy']:.4f}")
print(f"   F1 Score: {metrics['f1_score']:.4f}")

print("🎉 Baseline models test passed! ✅")