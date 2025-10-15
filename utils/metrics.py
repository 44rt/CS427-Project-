# # utils/metrics.py
# import numpy as np
# import torch
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score
# import matplotlib.pyplot as plt
# import seaborn as sns

# # def evaluate_model(model, X_test, y_test):
# #     """Evaluate a sklearn model"""
# #     y_pred = model.predict(X_test)
# #     y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
# #     metrics = {
# #         'accuracy': accuracy_score(y_test, y_pred),
# #         'f1_score': f1_score(y_test, y_pred),
# #         'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else 0.5,
# #         'confusion_matrix': confusion_matrix(y_test, y_pred)
# #     }
# #     return metrics

# # utils/metrics.py

# def evaluate_torch_model(model, X_test, y_test, device='cpu'):
#     """Evaluate a PyTorch model"""
#     model.eval()
#     with torch.no_grad():
#         X_test = X_test.to(device)
#         y_test = y_test.to(device)
        
#         outputs = model(X_test)
#         # For binary classification with sigmoid output
#         predictions = (outputs > 0.5).float().squeeze()
        
#         accuracy = (predictions == y_test).float().mean().item()
        
#         # Convert to numpy for other metrics
#         y_pred_np = predictions.cpu().numpy()
#         y_true_np = y_test.cpu().numpy()
        
#         # Calculate metrics with zero_division to handle cases with no positive predictions
#         precision = precision_score(y_true_np, y_pred_np, zero_division=0)
#         recall = recall_score(y_true_np, y_pred_np, zero_division=0)
#         f1 = f1_score(y_true_np, y_pred_np, zero_division=0)
        
#         return {
#             'accuracy': accuracy,
#             'precision': precision,
#             'recall': recall,
#             'f1_score': f1
#         }

# def evaluate_sklearn_model(model, X_test, y_test):
#     """Evaluate a scikit-learn model"""
#     y_pred = model.predict(X_test)
    
#     return {
#         'accuracy': accuracy_score(y_test, y_pred),
#         'precision': precision_score(y_test, y_pred, zero_division=0),
#         'recall': recall_score(y_test, y_pred, zero_division=0),
#         'f1_score': f1_score(y_test, y_pred, zero_division=0)
#     }

# def create_accuracy_matrix(results):
#     """Create accuracy matrix from results"""
#     if not results:
#         return None
    
#     num_tasks = len(results)
#     accuracy_matrix = np.zeros((num_tasks, num_tasks))
    
#     for task_trained in range(num_tasks):
#         for task_eval in range(num_tasks):
#             if task_eval in results[task_trained]:
#                 accuracy_matrix[task_trained, task_eval] = results[task_trained][task_eval]['accuracy']
    
#     return accuracy_matrix

# # The rest of your metrics.py functions remain the same...

# # In utils/metrics.py - make sure evaluate_torch_model works with your MLP output
# def evaluate_torch_model(model, X_test, y_test, device='cpu'):
#     """Evaluate a PyTorch model"""
#     model.eval()
#     with torch.no_grad():
#         X_test = X_test.to(device)
#         y_test = y_test.to(device)
        
#         outputs = model(X_test)
#         # For binary classification with sigmoid output
#         predictions = (outputs > 0.5).float().squeeze()
        
#         accuracy = (predictions == y_test).float().mean().item()
        
#         # Convert to numpy for other metrics
#         y_pred_np = predictions.cpu().numpy()
#         y_true_np = y_test.cpu().numpy()
        
#         precision = precision_score(y_true_np, y_pred_np, zero_division=0)
#         recall = recall_score(y_true_np, y_pred_np, zero_division=0)
#         f1 = f1_score(y_true_np, y_pred_np, zero_division=0)
        
#         return {
#             'accuracy': accuracy,
#             'precision': precision,
#             'recall': recall,
#             'f1_score': f1
#         }

# def calculate_forgetting(accuracy_matrix):
#     """Calculate forgetting rate from accuracy matrix"""
#     n_tasks = len(accuracy_matrix)
#     forgetting_rates = []
    
#     for task in range(n_tasks - 1):
#         task_forgetting = 0
#         for prev_task in range(task + 1):
#             max_acc = max(accuracy_matrix[t][prev_task] for t in range(prev_task, task + 1))
#             current_acc = accuracy_matrix[task][prev_task]
#             task_forgetting += (max_acc - current_acc)
#         forgetting_rates.append(task_forgetting / (task + 1))
    
#     return forgetting_rates

# def create_accuracy_matrix(results):
#     """Create accuracy matrix from results dictionary"""
#     n_tasks = len(results)
#     accuracy_matrix = [[0] * n_tasks for _ in range(n_tasks)]
    
#     for task_trained in range(n_tasks):
#         for task_evaluated in range(n_tasks):
#             if task_evaluated in results[task_trained]:
#                 accuracy_matrix[task_trained][task_evaluated] = results[task_trained][task_evaluated]['accuracy']
    
#     return accuracy_matrix

# utils/metrics.py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_torch_model(model, X_test, y_test, device='cpu'):
    """Evaluate a PyTorch model - UPDATED for BCEWithLogitsLoss"""
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        
        outputs = model(X_test)
        # Apply sigmoid since we're using BCEWithLogitsLoss
        # Handle both old (with sigmoid) and new (without sigmoid) model outputs
        if outputs.min() >= 0 and outputs.max() <= 1:  # Old model with sigmoid
            predictions = (outputs > 0.5).float().squeeze()
        else:  # New model with BCEWithLogitsLoss (no sigmoid)
            predictions = (torch.sigmoid(outputs) > 0.5).float().squeeze()
        
        accuracy = (predictions == y_test).float().mean().item()
        
        # Convert to numpy for other metrics
        y_pred_np = predictions.cpu().numpy()
        y_true_np = y_test.cpu().numpy()
        
        # Calculate metrics with zero_division to handle cases with no positive predictions
        precision = precision_score(y_true_np, y_pred_np, zero_division=0)
        recall = recall_score(y_true_np, y_pred_np, zero_division=0)
        f1 = f1_score(y_true_np, y_pred_np, zero_division=0)
        
        # Add AUC placeholder to match baseline format
        auc_score = accuracy  # Simple placeholder
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': auc_score  # Added for compatibility
        }

def evaluate_sklearn_model(model, X_test, y_test):
    """Evaluate a scikit-learn model"""
    y_pred = model.predict(X_test)
    
    # Add AUC placeholder to match baseline format
    auc_score = accuracy_score(y_test, y_pred)  # Simple placeholder
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': auc_score  # Added for compatibility
    }

def create_accuracy_matrix(results):
    """Create accuracy matrix from results dictionary"""
    if not results:
        return None
    
    n_tasks = len(results)
    accuracy_matrix = [[0] * n_tasks for _ in range(n_tasks)]
    
    for task_trained in range(n_tasks):
        for task_evaluated in range(n_tasks):
            if task_evaluated in results[task_trained]:
                accuracy_matrix[task_trained][task_evaluated] = results[task_trained][task_evaluated]['accuracy']
    
    return accuracy_matrix

def calculate_forgetting(accuracy_matrix):
    """Calculate forgetting rate from accuracy matrix"""
    if accuracy_matrix is None:
        return []
    
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

# Additional utility functions that might be in your original file
def plot_confusion_matrix(cm, labels=['Benign', 'Attack'], title='Confusion Matrix'):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt

def calculate_classification_metrics(y_true, y_pred, y_proba=None):
    """Calculate comprehensive classification metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['roc_auc'] = 0.5
    
    # Add confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics

def print_metrics(metrics, model_name=""):
    """Pretty print metrics"""
    if model_name:
        print(f"\n{model_name} Metrics:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")

def get_average_metrics(metrics_list):
    """Calculate average metrics from a list of metric dictionaries"""
    if not metrics_list:
        return {}
    
    avg_metrics = {}
    for key in metrics_list[0].keys():
        if key != 'confusion_matrix':  # Skip confusion matrix for averaging
            values = [m[key] for m in metrics_list if key in m]
            if values:
                avg_metrics[key] = np.mean(values)
    
    return avg_metrics

# Backward compatibility function
def evaluate_model(model, X_test, y_test):
    """
    Generic evaluate function that works with both sklearn and torch models
    Maintains backward compatibility
    """
    if hasattr(model, 'predict_proba'):  # Likely sklearn model
        return evaluate_sklearn_model(model, X_test, y_test)
    elif isinstance(model, torch.nn.Module):  # PyTorch model
        return evaluate_torch_model(model, X_test, y_test)
    else:
        raise ValueError("Unsupported model type")