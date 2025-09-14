# test_mlp.py
import torch
from models.mlp_models import MLP, train_torch_model
from utils.data_loader import create_incremental_tasks, create_torch_dataloaders
from utils.metrics import evaluate_torch_model

print("🔍 Testing MLP model...")
tasks, task_test_sets, global_test_set = create_incremental_tasks()
torch_tasks = create_torch_dataloaders(tasks)

# Use only FIRST task for quick test
train_loader, val_loader = torch_tasks[0]
X_test, y_test = task_test_sets[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Using device: {device}")

# Get input size from the data itself
input_size = X_test.shape[1]
model = MLP(input_size=input_size)
print(f"   Model created with input size: {input_size}")

print("🔍 Training MLP (2 epochs for test)...")
trained_model = train_torch_model(
    model, train_loader, val_loader, 
    device=device, epochs=2, lr=0.001
)

print("🔍 Evaluating MLP...")
metrics = evaluate_torch_model(trained_model, X_test, y_test, device=device)
print(f"✅ MLP test results:")
print(f"   Accuracy: {metrics['accuracy']:.4f}")
print(f"   F1 Score: {metrics['f1_score']:.4f}")

print("🎉 MLP test passed! ✅")