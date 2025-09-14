# utils/metrics.py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import torch

def evaluate_model(model, X_test, y_test):
    """Evaluate a sklearn model"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else 0.5,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    return metrics

def evaluate_torch_model(model, X_test, y_test, device='cpu'):
    """Evaluate a PyTorch model"""
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test.values).to(device)
        y_test_tensor = torch.FloatTensor(y_test.values).to(device)
        
        outputs = model(X_test_tensor)
        y_pred = (outputs > 0.5).float().cpu().numpy()
        y_proba = outputs.cpu().numpy()
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    return metrics

def calculate_forgetting(accuracy_matrix):
    """Calculate forgetting rate from accuracy matrix"""
    n_tasks = len(accuracy_matrix)
    forgetting_rates = []
    
    for task in range(n_tasks - 1):
        task_forgetting = 0
        for prev_task in range(task + 1):
            max_acc = max(accuracy_matrix[t][prev_task] for t in range(prev_task, task + 1))
            current_acc = accuracy_matrix[task][prev_task]
            task_forgetting += (max_acc - current_acc)
        forgetting_rates.append(task_forgetting / (task + 1))
    
    return forgetting_rates

def create_accuracy_matrix(results):
    """Create accuracy matrix from results dictionary"""
    n_tasks = len(results)
    accuracy_matrix = [[0] * n_tasks for _ in range(n_tasks)]
    
    for task_trained in range(n_tasks):
        for task_evaluated in range(n_tasks):
            if task_evaluated in results[task_trained]:
                accuracy_matrix[task_trained][task_evaluated] = results[task_trained][task_evaluated]['accuracy']
    
    return accuracy_matrix