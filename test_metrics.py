# test_metrics.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from utils.metrics import evaluate_model, calculate_forgetting

# Create dummy data for testing
print("🔍 Testing metrics with dummy data...")
X, y = make_classification(n_samples=100, n_features=20, random_state=42)
model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

metrics = evaluate_model(model, X, y)
print(f"✅ Metrics calculated:")
for k, v in metrics.items():
    if k != 'confusion_matrix':
        print(f"   {k}: {v:.4f}")

# Test forgetting calculation
print("\n🔍 Testing forgetting calculation...")
dummy_accuracy_matrix = [
    [0.9, 0.8, 0.7],    # After task 0
    [0.85, 0.75, 0.6],  # After task 1  
    [0.8, 0.7, 0.9]     # After task 2
]
forgetting = calculate_forgetting(dummy_accuracy_matrix)
print(f"✅ Forgetting rates: {forgetting}")

print("🎉 Metrics test passed! ✅")