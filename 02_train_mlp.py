# 02_train_mlp.py
import torch
import json
import pandas as pd
from utils.data_loader import create_incremental_tasks, create_torch_dataloaders
from utils.metrics import evaluate_torch_model, create_accuracy_matrix
from utils.visualization import plot_accuracy_matrix, plot_forgetting_rates, plot_global_accuracy, create_summary_table
from models.mlp_models import MLP, MLPWithLwF, MLPWithEWC, train_torch_model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading data and creating incremental tasks...")
    tasks, task_test_sets, global_test_set = create_incremental_tasks()
    torch_tasks = create_torch_dataloaders(tasks)
    
    X_global_test, y_global_test = global_test_set
    
    # Define models to train
    models = {
        "MLP (Naive)": MLP(INPUT_SIZE),
        "MLP + LwF": MLPWithLwF(INPUT_SIZE),
        "MLP + EWC": MLPWithEWC(INPUT_SIZE)
    }
    
    all_results = {}
    
    for model_name, model in models.items():
        print(f"\n--- Training {model_name} ---")
        model_results = {}
        previous_model = None
        
        for task_id, ((train_loader, val_loader), (X_task_test, y_task_test)) in enumerate(zip(torch_tasks, task_test_sets)):
            print(f"  Learning Task {task_id}...")
            
            # For LwF, set the previous model
            if model_name == "MLP + LwF" and previous_model is not None:
                model.set_previous_model(previous_model)
            
            # Train the model
            trained_model = train_torch_model(
                model, train_loader, val_loader, device=device,
                epochs=EPOCHS, lr=LEARNING_RATE,
                model_type='lwf' if 'LwF' in model_name else 'ewc' if 'EWC' in model_name else 'basic'
            )
            
            # For EWC, compute Fisher information after training
            if model_name == "MLP + EWC":
                trained_model.compute_fisher(train_loader, device=device)
            
            # Evaluate on all task test sets
            task_results = {}
            
            for eval_task in range(task_id + 1):
                X_eval, y_eval = task_test_sets[eval_task]
                metrics = evaluate_torch_model(trained_model, X_eval, y_eval, device=device)
                task_results[eval_task] = metrics
                print(f"    Task {eval_task} Accuracy: {metrics['accuracy']:.3f}")
            
            # Evaluate on global test set
            global_metrics = evaluate_torch_model(trained_model, X_global_test, y_global_test, device=device)
            task_results['global'] = global_metrics
            print(f"    Global Accuracy: {global_metrics['accuracy']:.3f}")
            
            model_results[task_id] = task_results
            previous_model = copy.deepcopy(trained_model)
        
        all_results[model_name] = model_results
        # Save the trained model
        torch.save(trained_model.state_dict(), f'{model_name.replace(" ", "_")}_model.pth')
    
    # Save results and create visualizations (same as baseline script)
    # ... [identical to the end of 01_train_baselines.py] ...

if __name__ == "__main__":
    main()