# # utils/visualization.py
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np
# from .metrics import calculate_forgetting

# # utils/visualization.py - MODIFY the plotting functions
# from config import get_attack_name  # ADD THIS IMPORT

# def plot_accuracy_matrix(accuracy_matrix, model_name, task_names):
#     """Plot accuracy matrix with actual attack names"""
#     # Convert task numbers to attack names
#     attack_names = [get_attack_name(i) for i in range(len(accuracy_matrix))]
    
#     plt.figure(figsize=(10, 8))
#     df = pd.DataFrame(accuracy_matrix, 
#                      index=[f'After {get_attack_name(i)}' for i in range(len(accuracy_matrix))],
#                      columns=attack_names)  # USE ATTACK NAMES
    
#     sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0)
#     plt.title(f'Accuracy Matrix - {model_name}')
#     plt.tight_layout()
#     return plt

# def plot_forgetting_rates(results_dict, task_names):
#     """Plot forgetting rates for all models"""
#     plt.figure(figsize=(12, 6))
    
#     for model_name, results in results_dict.items():
#         accuracy_matrix = []
#         for task_trained in range(len(task_names)):
#             row = []
#             for task_evaluated in range(len(task_names)):
#                 if task_evaluated in results[task_trained]:
#                     row.append(results[task_trained][task_evaluated]['accuracy'])
#                 else:
#                     row.append(0)
#             accuracy_matrix.append(row)
        
#         forgetting = calculate_forgetting(accuracy_matrix)
#         plt.plot(range(1, len(forgetting) + 1), forgetting, 'o-', label=model_name, markersize=8)
    
#     plt.xlabel('Task Number')
#     plt.ylabel('Average Forgetting Rate')
#     plt.title('Catastrophic Forgetting Comparison')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     return plt

# def plot_global_accuracy(results_dict, task_names):
#     """Plot global accuracy progression"""
#     plt.figure(figsize=(12, 6))
    
#     for model_name, results in results_dict.items():
#         global_acc = []
#         for task_trained in range(len(task_names)):
#             if 'global' in results[task_trained]:
#                 global_acc.append(results[task_trained]['global']['accuracy'])
        
#         plt.plot(range(len(global_acc)), global_acc, 'o-', label=model_name, markersize=8)
    
#     plt.xlabel('Tasks Learned')
#     plt.ylabel('Global Accuracy')
#     plt.title('Global Performance After Learning Each Task')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     return plt

# def create_summary_table(results_dict):
#     """Create summary table of results"""
#     summary_data = []
    
#     for model_name, results in results_dict.items():
#         # Calculate final metrics
#         final_task = max(results.keys())
# if 'global' in results[final_task]:
#     final_global = results[final_task]['global']
# else:
#     # fallback if missing
#     final_global = {'accuracy': 0, 'f1_score': 0, 'roc_auc': 0}
#         accuracy_matrix = []
        
#         for task_trained in range(len(results)):
#             row = []
#             for task_evaluated in range(len(results)):
#                 if task_evaluated in results[task_trained]:
#                     row.append(results[task_trained][task_evaluated]['accuracy'])
#             accuracy_matrix.append(row)
        
#         forgetting = calculate_forgetting(accuracy_matrix)
#         avg_forgetting = np.mean(forgetting) if forgetting else 0
        
#         summary_data.append({
#             'Model': model_name,
#             'Final Accuracy': f"{final_global['accuracy']:.3f}",
#             'Final F1-Score': f"{final_global['f1_score']:.3f}",
#             'Final AUC': f"{final_global['roc_auc']:.3f}",
#             'Avg. Forgetting': f"{avg_forgetting:.3f}",
#             'Status': 'PASS' if avg_forgetting < 0.1 else 'FAIL'
#         })
    
#     return pd.DataFrame(summary_data)

# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from .metrics import calculate_forgetting
from config import get_attack_name  # Ensure this import is present

def plot_accuracy_matrix(accuracy_matrix, model_name, task_names):
    """Plot accuracy matrix with actual attack names"""
    attack_names = [get_attack_name(i) for i in range(len(accuracy_matrix))]
    
    plt.figure(figsize=(10, 8))
    df = pd.DataFrame(
        accuracy_matrix, 
        index=[f'After {get_attack_name(i)}' for i in range(len(accuracy_matrix))],
        columns=attack_names
    )
    
    sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0)
    plt.title(f'Accuracy Matrix - {model_name}')
    plt.tight_layout()
    return plt

def plot_forgetting_rates(results_dict, task_names):
    """Plot forgetting rates for all models"""
    plt.figure(figsize=(12, 6))
    
    for model_name, results in results_dict.items():
        accuracy_matrix = []
        for task_trained in range(len(task_names)):
            row = []
            for task_evaluated in range(len(task_names)):
                if task_evaluated in results.get(task_trained, {}):
                    row.append(results[task_trained][task_evaluated].get('accuracy', 0))
                else:
                    row.append(0)
            accuracy_matrix.append(row)
        
        forgetting = calculate_forgetting(accuracy_matrix)
        plt.plot(range(1, len(forgetting) + 1), forgetting, 'o-', label=model_name, markersize=8)
    
    plt.xlabel('Task Number')
    plt.ylabel('Average Forgetting Rate')
    plt.title('Catastrophic Forgetting Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt

def plot_global_accuracy(results_dict, task_names):
    """Plot global accuracy progression"""
    plt.figure(figsize=(12, 6))
    
    for model_name, results in results_dict.items():
        global_acc = []
        for task_trained in range(len(task_names)):
            task_results = results.get(task_trained, {})
            if 'global' in task_results:
                global_acc.append(task_results['global'].get('accuracy', 0))
        
        plt.plot(range(len(global_acc)), global_acc, 'o-', label=model_name, markersize=8)
    
    plt.xlabel('Tasks Learned')
    plt.ylabel('Global Accuracy')
    plt.title('Global Performance After Learning Each Task')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt

def create_summary_table(results_dict):
    """Create summary table of results"""
    summary_data = []
    
    for model_name, results in results_dict.items():
        if not results:
            continue  # skip empty results
        
        final_task = max(results.keys())
        final_global = results.get(final_task, {}).get('global', {'accuracy': 0, 'f1_score': 0, 'roc_auc': 0})
        
        accuracy_matrix = []
        for task_trained in range(len(results)):
            row = []
            for task_evaluated in range(len(results)):
                task_data = results.get(task_trained, {})
                row.append(task_data.get(task_evaluated, {}).get('accuracy', 0))
            accuracy_matrix.append(row)
        
        forgetting = calculate_forgetting(accuracy_matrix)
        avg_forgetting = np.mean(forgetting) if forgetting else 0
        
        summary_data.append({
            'Model': model_name,
            'Final Accuracy': f"{final_global.get('accuracy', 0):.3f}",
            'Final F1-Score': f"{final_global.get('f1_score', 0):.3f}",
            'Final AUC': f"{final_global.get('roc_auc', 0):.3f}",
            'Avg. Forgetting': f"{avg_forgetting:.3f}",
            'Status': 'PASS' if avg_forgetting < 0.1 else 'FAIL'
        })
    
    return pd.DataFrame(summary_data)
