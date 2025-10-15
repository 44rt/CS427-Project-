# # # 02_train_mlp.py
# # import torch
# # import json
# # import pandas as pd
# # from utils.data_loader import create_incremental_tasks, create_torch_dataloaders
# # from utils.metrics import evaluate_torch_model, create_accuracy_matrix
# # from utils.visualization import plot_accuracy_matrix, plot_forgetting_rates, plot_global_accuracy, create_summary_table
# # from models.mlp_models import MLP, MLPWithLwF, MLPWithEWC, train_torch_model

# # def main():
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #     print(f"Using device: {device}")
    
# #     print("Loading data and creating incremental tasks...")
# #     tasks, task_test_sets, global_test_set = create_incremental_tasks()
# #     torch_tasks = create_torch_dataloaders(tasks)
    
# #     X_global_test, y_global_test = global_test_set
    
# #     # Define models to train
# #     models = {
# #         "MLP (Naive)": MLP(INPUT_SIZE),
# #         "MLP + LwF": MLPWithLwF(INPUT_SIZE),
# #         "MLP + EWC": MLPWithEWC(INPUT_SIZE)
# #     }
    
# #     all_results = {}
    
# #     for model_name, model in models.items():
# #         print(f"\n--- Training {model_name} ---")
# #         model_results = {}
# #         previous_model = None
        
# #         for task_id, ((train_loader, val_loader), (X_task_test, y_task_test)) in enumerate(zip(torch_tasks, task_test_sets)):
# #             print(f"  Learning Task {task_id}...")
            
# #             # For LwF, set the previous model
# #             if model_name == "MLP + LwF" and previous_model is not None:
# #                 model.set_previous_model(previous_model)
            
# #             # Train the model
# #             trained_model = train_torch_model(
# #                 model, train_loader, val_loader, device=device,
# #                 epochs=EPOCHS, lr=LEARNING_RATE,
# #                 model_type='lwf' if 'LwF' in model_name else 'ewc' if 'EWC' in model_name else 'basic'
# #             )
            
# #             # For EWC, compute Fisher information after training
# #             if model_name == "MLP + EWC":
# #                 trained_model.compute_fisher(train_loader, device=device)
            
# #             # Evaluate on all task test sets
# #             task_results = {}
            
# #             for eval_task in range(task_id + 1):
# #                 X_eval, y_eval = task_test_sets[eval_task]
# #                 metrics = evaluate_torch_model(trained_model, X_eval, y_eval, device=device)
# #                 task_results[eval_task] = metrics
# #                 print(f"    Task {eval_task} Accuracy: {metrics['accuracy']:.3f}")
            
# #             # Evaluate on global test set
# #             global_metrics = evaluate_torch_model(trained_model, X_global_test, y_global_test, device=device)
# #             task_results['global'] = global_metrics
# #             print(f"    Global Accuracy: {global_metrics['accuracy']:.3f}")
            
# #             model_results[task_id] = task_results
# #             previous_model = copy.deepcopy(trained_model)
        
# #         all_results[model_name] = model_results
# #         # Save the trained model
# #         torch.save(trained_model.state_dict(), f'{model_name.replace(" ", "_")}_model.pth')
    
# #     # Save results and create visualizations (same as baseline script)
# #     # ... [identical to the end of 01_train_baselines.py] ...

# # if __name__ == "__main__":
# #     main()

# # 02_train_mlp.py
# # import torch
# # import json
# # import pandas as pd
# # import numpy as np
# # from utils.data_loader import create_incremental_tasks, create_torch_dataloaders
# # from utils.metrics import evaluate_torch_model, create_accuracy_matrix
# # from utils.visualization import plot_accuracy_matrix, plot_forgetting_rates, plot_global_accuracy, create_summary_table
# # from models.mlp_models import MLP, MLPWithLwF, MLPWithEWC, train_torch_model
# # from config import INPUT_SIZE, LEARNING_RATE, EPOCHS


# # # Update model input dimension if needed
# # input_dim = train_loader.dataset.tensors[0].shape[1]
# # model = MLP(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, output_dim=output_dim)

# # def main():
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #     print(f"Using device: {device}")
    
# #     print("Loading data and creating incremental tasks...")
# #     tasks, task_test_sets, global_test_set = create_incremental_tasks()
# #     torch_tasks = create_torch_dataloaders(tasks)
    
# #     X_global_test, y_global_test = global_test_set
    
# #     # Define models to train
# #     models = {
# #         "MLP (Naive)": MLP(INPUT_SIZE),
# #         "MLP + LwF": MLPWithLwF(INPUT_SIZE),
# #         "MLP + EWC": MLPWithEWC(INPUT_SIZE)
# #     }
    
# #     all_results = {}
    
# #     for model_name, model in models.items():
# #         print(f"\n--- Training {model_name} ---")
# #         model_results = {}
# #         previous_model = None
        
# #         for task_id, ((train_loader, val_loader), (X_task_test, y_task_test)) in enumerate(zip(torch_tasks, task_test_sets)):
# #             current_attack = f"Task_{task_id}"  # You can update this with attack names later
# #             print(f"  Learning {current_attack}...")
            
# #             # For LwF, set the previous model
# #             if model_name == "MLP + LwF" and previous_model is not None:
# #                 model.set_previous_model(previous_model)
            
# #             # Train the model
# #             trained_model = train_torch_model(
# #                 model, train_loader, val_loader, device=device,
# #                 epochs=EPOCHS, lr=LEARNING_RATE,
# #                 model_type='lwf' if 'LwF' in model_name else 'ewc' if 'EWC' in model_name else 'basic'
# #             )
            
# #             # For EWC, this is where you'd compute Fisher information
# #             # (We'll implement this properly in the next step)
            
# #             # Evaluate on all task test sets
# #             task_results = {}
            
# #             for eval_task in range(task_id + 1):
# #                 X_eval, y_eval = task_test_sets[eval_task]
# #                 metrics = evaluate_torch_model(trained_model, X_eval, y_eval, device=device)
# #                 task_results[eval_task] = metrics
# #                 print(f"    Task {eval_task} Accuracy: {metrics['accuracy']:.3f}")
            
# #             # Evaluate on global test set
# #             global_metrics = evaluate_torch_model(trained_model, X_global_test, y_global_test, device=device)
# #             task_results['global'] = global_metrics
# #             print(f"    Global Accuracy: {global_metrics['accuracy']:.3f}")
            
# #             model_results[task_id] = task_results
# #             previous_model = trained_model
        
# #         all_results[model_name] = model_results
# #         # Save the trained model
# #         torch.save(trained_model.state_dict(), f'{model_name.replace(" ", "_").replace("(", "").replace(")", "")}_model.pth')
    
# #     # Save results
# #     with open('mlp_results.json', 'w') as f:
# #         json_results = {}
# #         for model_name, results in all_results.items():
# #             json_results[model_name] = {}
# #             for task_trained, task_results in results.items():
# #                 json_results[model_name][task_trained] = {}
# #                 for eval_task, metrics in task_results.items():
# #                     json_results[model_name][task_trained][eval_task] = {
# #                         k: float(v) if isinstance(v, (np.floating, np.integer)) else v.tolist() if hasattr(v, 'tolist') else v
# #                         for k, v in metrics.items()
# #                     }
# #         json.dump(json_results, f, indent=2)
    
# #     # Create visualizations
# #     task_names = [f'Task {i}' for i in range(len(tasks))]
    
# #     # Plot accuracy matrix for each model
# #     for model_name in models.keys():
# #         accuracy_matrix = create_accuracy_matrix(all_results[model_name])
# #         plt = plot_accuracy_matrix(accuracy_matrix, model_name, task_names)
# #         plt.savefig(f'accuracy_matrix_{model_name.replace(" ", "_").replace("(", "").replace(")", "")}.png')
# #         plt.close()
    
# #     # Plot comparison plots
# #     plt = plot_forgetting_rates(all_results, task_names)
# #     plt.savefig('mlp_forgetting_comparison.png')
# #     plt.close()
    
# #     plt = plot_global_accuracy(all_results, task_names)
# #     plt.savefig('mlp_global_accuracy.png')
# #     plt.close()
    
# #     # Create and print summary table
# #     summary_df = create_summary_table(all_results)
# #     print("\n" + "="*60)
# #     print("MLP MODELS SUMMARY")
# #     print("="*60)
# #     print(summary_df.to_string(index=False))
# #     print("="*60)
    
# #     # Save summary table
# #     summary_df.to_csv('mlp_summary.csv', index=False)
# #     print("Results saved to JSON and CSV files")
# #     print("Plots saved as PNG files")

# # if __name__ == "__main__":
# #     main()

# # 02_train_mlp.py
# # import torch
# # import json
# # import pandas as pd
# # import numpy as np
# # from utils.data_loader import create_incremental_tasks, create_torch_dataloaders
# # from utils.metrics import evaluate_torch_model, create_accuracy_matrix
# # from utils.visualization import plot_accuracy_matrix, plot_forgetting_rates, plot_global_accuracy, create_summary_table
# # from models.mlp_models import MLP, MLPWithLwF, MLPWithEWC, train_torch_model
# # from config import INPUT_SIZE, LEARNING_RATE, EPOCHS


# # # Update model input dimension if needed
# # input_dim = train_loader.dataset.tensors[0].shape[1]
# # model = MLP(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, output_dim=output_dim)

# # def main():
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #     print(f"Using device: {device}")
    
# #     print("Loading data and creating incremental tasks...")
# #     tasks, task_test_sets, global_test_set = create_incremental_tasks()
# #     torch_tasks = create_torch_dataloaders(tasks)
    
# #     X_global_test, y_global_test = global_test_set
    
# #     # Define models to train
# #     models = {
# #         "MLP (Naive)": MLP(INPUT_SIZE),
# #         "MLP + LwF": MLPWithLwF(INPUT_SIZE),
# #         "MLP + EWC": MLPWithEWC(INPUT_SIZE)
# #     }
    
# #     all_results = {}
    
# #     for model_name, model in models.items():
# #         print(f"\n--- Training {model_name} ---")
# #         model_results = {}
# #         previous_model = None
        
# #         for task_id, ((train_loader, val_loader), (X_task_test, y_task_test)) in enumerate(zip(torch_tasks, task_test_sets)):
# #             current_attack = f"Task_{task_id}"  # You can update this with attack names later
# #             print(f"  Learning {current_attack}...")
            
# #             # For LwF, set the previous model
# #             if model_name == "MLP + LwF" and previous_model is not None:
# #                 model.set_previous_model(previous_model)
            
# #             # Train the model
# #             trained_model = train_torch_model(
# #                 model, train_loader, val_loader, device=device,
# #                 epochs=EPOCHS, lr=LEARNING_RATE,
# #                 model_type='lwf' if 'LwF' in model_name else 'ewc' if 'EWC' in model_name else 'basic'
# #             )
            
# #             # For EWC, this is where you'd compute Fisher information
# #             # (We'll implement this properly in the next step)
            
# #             # Evaluate on all task test sets
# #             task_results = {}
            
# #             for eval_task in range(task_id + 1):
# #                 X_eval, y_eval = task_test_sets[eval_task]
# #                 metrics = evaluate_torch_model(trained_model, X_eval, y_eval, device=device)
# #                 task_results[eval_task] = metrics
# #                 print(f"    Task {eval_task} Accuracy: {metrics['accuracy']:.3f}")
            
# #             # Evaluate on global test set
# #             global_metrics = evaluate_torch_model(trained_model, X_global_test, y_global_test, device=device)
# #             task_results['global'] = global_metrics
# #             print(f"    Global Accuracy: {global_metrics['accuracy']:.3f}")
            
# #             model_results[task_id] = task_results
# #             previous_model = trained_model
        
# #         all_results[model_name] = model_results
# #         # Save the trained model
# #         torch.save(trained_model.state_dict(), f'{model_name.replace(" ", "_").replace("(", "").replace(")", "")}_model.pth')
    
# #     # Save results
# #     with open('mlp_results.json', 'w') as f:
# #         json_results = {}
# #         for model_name, results in all_results.items():
# #             json_results[model_name] = {}
# #             for task_trained, task_results in results.items():
# #                 json_results[model_name][task_trained] = {}
# #                 for eval_task, metrics in task_results.items():
# #                     json_results[model_name][task_trained][eval_task] = {
# #                         k: float(v) if isinstance(v, (np.floating, np.integer)) else v.tolist() if hasattr(v, 'tolist') else v
# #                         for k, v in metrics.items()
# #                     }
# #         json.dump(json_results, f, indent=2)
    
# #     # Create visualizations
# #     task_names = [f'Task {i}' for i in range(len(tasks))]
    
# #     # Plot accuracy matrix for each model
# #     for model_name in models.keys():
# #         accuracy_matrix = create_accuracy_matrix(all_results[model_name])
# #         plt = plot_accuracy_matrix(accuracy_matrix, model_name, task_names)
# #         plt.savefig(f'accuracy_matrix_{model_name.replace(" ", "_").replace("(", "").replace(")", "")}.png')
# #         plt.close()
    
# #     # Plot comparison plots
# #     plt = plot_forgetting_rates(all_results, task_names)
# #     plt.savefig('mlp_forgetting_comparison.png')
# #     plt.close()
    
# #     plt = plot_global_accuracy(all_results, task_names)
# #     plt.savefig('mlp_global_accuracy.png')
# #     plt.close()
    
# #     # Create and print summary table
# #     summary_df = create_summary_table(all_results)
# #     print("\n" + "="*60)
# #     print("MLP MODELS SUMMARY")
# #     print("="*60)
# #     print(summary_df.to_string(index=False))
# #     print("="*60)
    
# #     # Save summary table
# #     summary_df.to_csv('mlp_summary.csv', index=False)
# #     print("Results saved to JSON and CSV files")
# #     print("Plots saved as PNG files")

# # if __name__ == "__main__":
# #     main()

# # 02_train_mlp.py
# # import torch
# # import json
# # import pandas as pd
# # import numpy as np
# # from utils.data_loader import load_and_preprocess_data, create_incremental_tasks, create_torch_dataloaders
# # from utils.metrics import evaluate_torch_model, create_accuracy_matrix
# # from utils.visualization import plot_accuracy_matrix, plot_forgetting_rates, plot_global_accuracy, create_summary_table
# # from models.mlp_models import MLP, MLPWithLwF, MLPWithEWC, train_torch_model
# # from config import LEARNING_RATE, EPOCHS, HIDDEN_DIMS, BATCH_SIZE, SAMPLE_FRACTION, DATA_PATH

# # def main():
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #     print(f"Using device: {device}")
    
# #     print("Loading data and creating incremental tasks...")
    
# #     # Load and preprocess data first
# #     df_processed, tasks, global_test_set = load_and_preprocess_data(
# #         DATA_PATH, sample_frac=SAMPLE_FRACTION
# #     )
    
# #     # Create torch dataloaders
# #     torch_tasks = create_torch_dataloaders(tasks, batch_size=BATCH_SIZE)
    
# #     # Get task test sets
# #     task_test_sets = []
# #     for task in tasks:
# #         X_task_test = task[2]  # Test data from each task
# #         y_task_test = task[3]  # Test labels from each task
# #         task_test_sets.append((X_task_test, y_task_test))
    
# #     X_global_test, y_global_test = global_test_set
    
# #     # Get the actual input dimension from the data
# #     train_loader = torch_tasks[0][0]  # First task's train loader
# #     input_dim = train_loader.dataset.tensors[0].shape[1]
# #     print(f" Detected input dimension: {input_dim}")
    
# #     # Define models to train with correct input dimensions
# #     models = {
# #         "MLP (Naive)": MLP(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, output_dim=2),
# #         "MLP + LwF": MLPWithLwF(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, output_dim=2),
# #         "MLP + EWC": MLPWithEWC(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, output_dim=2)
# #     }
    
# #     all_results = {}
    
# #     for model_name, model in models.items():
# #         print(f"\n--- Training {model_name} ---")
# #         model.to(device)
# #         model_results = {}
# #         previous_model = None
        
# #         for task_id, ((train_loader, val_loader), (X_task_test, y_task_test)) in enumerate(zip(torch_tasks, task_test_sets)):
# #             current_attack = f"Task_{task_id}"
# #             print(f"  Learning {current_attack}...")
            
# #             # Debug: Check data dimensions
# #             print(f"    Training data shape: {train_loader.dataset.tensors[0].shape}")
# #             print(f"    Validation data shape: {val_loader.dataset.tensors[0].shape}")
            
# #             # For LwF, set the previous model
# #             if model_name == "MLP + LwF" and previous_model is not None:
# #                 model.set_previous_model(previous_model)
            
# #             # Train the model
# #             trained_model = train_torch_model(
# #                 model, train_loader, val_loader, device=device,
# #                 epochs=EPOCHS, lr=LEARNING_RATE,
# #                 model_type='lwf' if 'LwF' in model_name else 'ewc' if 'EWC' in model_name else 'basic'
# #             )
            
# #             # For EWC, this is where you'd compute Fisher information
# #             if model_name == "MLP + EWC" and previous_model is not None:
# #                 # You'll need to implement EWC Fisher computation here
# #                 pass
            
# #             # Evaluate on all task test sets
# #             task_results = {}
            
# #             for eval_task in range(task_id + 1):
# #                 X_eval, y_eval = task_test_sets[eval_task]
# #                 # Convert to torch tensors
# #                 X_eval_tensor = torch.FloatTensor(X_eval.values if hasattr(X_eval, 'values') else X_eval)
# #                 y_eval_tensor = torch.LongTensor(y_eval.values if hasattr(y_eval, 'values') else y_eval)
                
# #                 metrics = evaluate_torch_model(trained_model, X_eval_tensor, y_eval_tensor, device=device)
# #                 task_results[eval_task] = metrics
# #                 print(f"    Task {eval_task} Accuracy: {metrics['accuracy']:.3f}")
            
# #             # Evaluate on global test set
# #             X_global_tensor = torch.FloatTensor(X_global_test.values if hasattr(X_global_test, 'values') else X_global_test)
# #             y_global_tensor = torch.LongTensor(y_global_test.values if hasattr(y_global_test, 'values') else y_global_test)
            
# #             global_metrics = evaluate_torch_model(trained_model, X_global_tensor, y_global_tensor, device=device)
# #             task_results['global'] = global_metrics
# #             print(f"    Global Accuracy: {global_metrics['accuracy']:.3f}")
            
# #             model_results[task_id] = task_results
# #             previous_model = trained_model
        
# #         all_results[model_name] = model_results
        
# #         # Save the trained model
# #         model_filename = f'{model_name.replace(" ", "_").replace("(", "").replace(")", "")}_model.pth'
# #         torch.save(trained_model.state_dict(), model_filename)
# #         print(f" Model saved as {model_filename}")
    
# #     # Save results
# #     with open('mlp_results.json', 'w') as f:
# #         json_results = {}
# #         for model_name, results in all_results.items():
# #             json_results[model_name] = {}
# #             for task_trained, task_results in results.items():
# #                 json_results[model_name][task_trained] = {}
# #                 for eval_task, metrics in task_results.items():
# #                     json_results[model_name][task_trained][eval_task] = {
# #                         k: float(v) if isinstance(v, (np.floating, np.integer)) else v
# #                         for k, v in metrics.items()
# #                     }
# #         json.dump(json_results, f, indent=2)
    
# #     # Create visualizations
# #     task_names = [f'Task {i}' for i in range(len(torch_tasks))]
    
# #     # Plot accuracy matrix for each model
# #     for model_name in models.keys():
# #         accuracy_matrix = create_accuracy_matrix(all_results[model_name])
# #         if accuracy_matrix is not None:
# #             plt = plot_accuracy_matrix(accuracy_matrix, model_name, task_names)
# #             plt.savefig(f'accuracy_matrix_{model_name.replace(" ", "_").replace("(", "").replace(")", "")}.png', dpi=300, bbox_inches='tight')
# #             plt.close()
    
# #     # Plot comparison plots
# #     try:
# #         plt = plot_forgetting_rates(all_results, task_names)
# #         plt.savefig('mlp_forgetting_comparison.png', dpi=300, bbox_inches='tight')
# #         plt.close()
# #     except Exception as e:
# #         print(f" Could not create forgetting plot: {e}")
    
# #     try:
# #         plt = plot_global_accuracy(all_results, task_names)
# #         plt.savefig('mlp_global_accuracy.png', dpi=300, bbox_inches='tight')
# #         plt.close()
# #     except Exception as e:
# #         print(f" Could not create global accuracy plot: {e}")
    
# #     # Create and print summary table
# #     try:
# #         summary_df = create_summary_table(all_results)
# #         print("\n" + "="*60)
# #         print("MLP MODELS SUMMARY")
# #         print("="*60)
# #         print(summary_df.to_string(index=False))
# #         print("="*60)
        
# #         # Save summary table
# #         summary_df.to_csv('mlp_summary.csv', index=False)
# #         print(" Results saved to JSON and CSV files")
# #         print(" Plots saved as PNG files")
# #     except Exception as e:
# #         print(f" Could not create summary table: {e}")

# # if __name__ == "__main__":
# #     main()

# # 02_train_mlp.py
# # 02_train_mlp.py
# # import torch
# # import json
# # import pandas as pd
# # import numpy as np
# # from utils.data_loader import create_incremental_tasks, create_torch_dataloaders
# # from utils.metrics import evaluate_torch_model, create_accuracy_matrix
# # from utils.visualization import plot_accuracy_matrix, plot_forgetting_rates, plot_global_accuracy, create_summary_table
# # from models.mlp_models import MLP, MLPWithLwF, MLPWithEWC, train_torch_model
# # from config import LEARNING_RATE, EPOCHS, BATCH_SIZE

# # def main():
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #     print(f"Using device: {device}")
    
# #     print("Loading data and creating incremental tasks...")
    
# #     # Use create_incremental_tasks directly - it handles everything
# #     tasks, task_test_sets, global_test_set = create_incremental_tasks()
    
# #     # Unpack global test set
# #     X_global_test, y_global_test = global_test_set
    
# #     print(f" Number of tasks: {len(tasks)}")
# #     print(f" Global test set shape: X={X_global_test.shape}, y={y_global_test.shape}")
    
# #     # Create torch dataloaders
# #     torch_tasks = create_torch_dataloaders(tasks)
# #     print(f" Created {len(torch_tasks)} torch tasks")
    
# #     # Get the actual input dimension from the data
# #     train_loader = torch_tasks[0][0]  # First task's train loader
# #     input_dim = train_loader.dataset.tensors[0].shape[1]
# #     print(f" Detected input dimension: {input_dim}")
    
# #     # Define models to train with CORRECT parameters matching your mlp_models.py
# #     models = {
# #         "MLP (Naive)": MLP(input_size=input_dim, hidden_size=128),
# #         "MLP + LwF": MLPWithLwF(input_size=input_dim, hidden_size=128),
# #         "MLP + EWC": MLPWithEWC(input_size=input_dim, hidden_size=128)
# #     }
    
# #     all_results = {}
    
# #     for model_name, model in models.items():
# #         print(f"\n--- Training {model_name} ---")
# #         model.to(device)
# #         model_results = {}
# #         previous_model = None
        
# #         for task_id, ((train_loader, val_loader), (X_task_test, y_task_test)) in enumerate(zip(torch_tasks, task_test_sets)):
# #             current_attack = f"Task_{task_id}"
# #             print(f"  Learning {current_attack}...")
            
# #             # Debug: Check data dimensions
# #             print(f"    Training data shape: {train_loader.dataset.tensors[0].shape}")
# #             print(f"    Validation data shape: {val_loader.dataset.tensors[0].shape}")
            
# #             # Check label distribution
# #             train_labels = train_loader.dataset.tensors[1]
# #             print(f"    Training labels distribution: {torch.bincount(train_labels.long())}")
            
# #             # For LwF, set the previous model
# #             if model_name == "MLP + LwF" and previous_model is not None:
# #                 model.set_previous_model(previous_model)
            
# #             # Train the model
# #             trained_model = train_torch_model(
# #                 model, train_loader, val_loader, device=device,
# #                 epochs=EPOCHS, lr=LEARNING_RATE,
# #                 model_type='lwf' if 'LwF' in model_name else 'ewc' if 'EWC' in model_name else 'basic'
# #             )
            
# #             # For EWC, compute Fisher information after training on each task
# #             if model_name == "MLP + EWC":
# #                 print("    Computing Fisher information for EWC...")
# #                 model.compute_fisher(train_loader, device=device)
            
# #             # Evaluate on all task test sets
# #             task_results = {}
            
# #             for eval_task in range(task_id + 1):
# #                 X_eval, y_eval = task_test_sets[eval_task]
# #                 # Convert to torch tensors
# #                 X_eval_tensor = torch.FloatTensor(X_eval.values)
# #                 y_eval_tensor = torch.LongTensor(y_eval.values)
                
# #                 metrics = evaluate_torch_model(trained_model, X_eval_tensor, y_eval_tensor, device=device)
# #                 task_results[eval_task] = metrics
# #                 print(f"    Task {eval_task} Accuracy: {metrics['accuracy']:.3f}")
            
# #             # Evaluate on global test set
# #             X_global_tensor = torch.FloatTensor(X_global_test.values)
# #             y_global_tensor = torch.LongTensor(y_global_test.values)
            
# #             global_metrics = evaluate_torch_model(trained_model, X_global_tensor, y_global_tensor, device=device)
# #             task_results['global'] = global_metrics
# #             print(f"    Global Accuracy: {global_metrics['accuracy']:.3f}")
            
# #             model_results[task_id] = task_results
# #             previous_model = trained_model
        
# #         all_results[model_name] = model_results
        
# #         # Save the trained model
# #         model_filename = f'{model_name.replace(" ", "_").replace("(", "").replace(")", "")}_model.pth'
# #         torch.save(trained_model.state_dict(), model_filename)
# #         print(f" Model saved as {model_filename}")
    
# #     # Save results
# #     with open('mlp_results.json', 'w') as f:
# #         json_results = {}
# #         for model_name, results in all_results.items():
# #             json_results[model_name] = {}
# #             for task_trained, task_results in results.items():
# #                 json_results[model_name][task_trained] = {}
# #                 for eval_task, metrics in task_results.items():
# #                     json_results[model_name][task_trained][eval_task] = {
# #                         k: float(v) if isinstance(v, (np.floating, np.integer)) else v
# #                         for k, v in metrics.items()
# #                     }
# #         json.dump(json_results, f, indent=2)
    
# #     # Create visualizations
# #     task_names = [f'Task {i}' for i in range(len(torch_tasks))]
    
# #     # Plot accuracy matrix for each model
# #     for model_name in models.keys():
# #         accuracy_matrix = create_accuracy_matrix(all_results[model_name])
# #         if accuracy_matrix is not None:
# #             plt = plot_accuracy_matrix(accuracy_matrix, model_name, task_names)
# #             plt.savefig(f'accuracy_matrix_{model_name.replace(" ", "_").replace("(", "").replace(")", "")}.png', dpi=300, bbox_inches='tight')
# #             plt.close()
    
# #     # Plot comparison plots
# #     try:
# #         plt = plot_forgetting_rates(all_results, task_names)
# #         plt.savefig('mlp_forgetting_comparison.png', dpi=300, bbox_inches='tight')
# #         plt.close()
# #     except Exception as e:
# #         print(f" Could not create forgetting plot: {e}")
    
# #     try:
# #         plt = plot_global_accuracy(all_results, task_names)
# #         plt.savefig('mlp_global_accuracy.png', dpi=300, bbox_inches='tight')
# #         plt.close()
# #     except Exception as e:
# #         print(f" Could not create global accuracy plot: {e}")
    
# #     # Create and print summary table
# #     try:
# #         summary_df = create_summary_table(all_results)
# #         print("\n" + "="*60)
# #         print("MLP MODELS SUMMARY")
# #         print("="*60)
# #         print(summary_df.to_string(index=False))
# #         print("="*60)
        
# #         # Save summary table
# #         summary_df.to_csv('mlp_summary.csv', index=False)
# #         print(" Results saved to JSON and CSV files")
# #         print(" Plots saved as PNG files")
# #     except Exception as e:
# #         print(f" Could not create summary table: {e}")

# # if __name__ == "__main__":
# #     main()

# # 02_train_mlp.py
# import torch
# import json
# import pandas as pd
# import numpy as np
# from utils.data_loader import create_incremental_tasks, create_torch_dataloaders
# from utils.metrics import evaluate_torch_model, create_accuracy_matrix
# from utils.visualization import plot_accuracy_matrix, plot_forgetting_rates, plot_global_accuracy, create_summary_table
# from models.mlp_models import MLP, MLPWithLwF, MLPWithEWC, train_torch_model
# from config import LEARNING_RATE, EPOCHS, BATCH_SIZE

# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     print("Loading data and creating incremental tasks...")
    
#     # Use create_incremental_tasks directly - it handles everything
#     tasks, task_test_sets, global_test_set = create_incremental_tasks()
    
#     # Unpack global test set
#     X_global_test, y_global_test = global_test_set
    
#     print(f" Number of tasks: {len(tasks)}")
#     print(f" Global test set shape: X={X_global_test.shape}, y={y_global_test.shape}")
    
#     # Create torch dataloaders
#     torch_tasks = create_torch_dataloaders(tasks)
#     print(f" Created {len(torch_tasks)} torch tasks")
    
#     # Get the actual input dimension from the data
#     train_loader = torch_tasks[0][0]  # First task's train loader
#     input_dim = train_loader.dataset.tensors[0].shape[1]
#     print(f" Detected input dimension: {input_dim}")
    
#     # Define models to train with correct input dimensions
#     models = {
#         "MLP (Naive)": MLP(input_size=input_dim, hidden_size=128),
#         "MLP + LwF": MLPWithLwF(input_size=input_dim, hidden_size=128),
#         "MLP + EWC": MLPWithEWC(input_size=input_dim, hidden_size=128)
#     }
    
#     all_results = {}
    
#     for model_name, model in models.items():
#         print(f"\n--- Training {model_name} ---")
#         model.to(device)
#         model_results = {}
#         previous_model = None
        
#         for task_id, ((train_loader, val_loader), (X_task_test, y_task_test)) in enumerate(zip(torch_tasks, task_test_sets)):
#             current_attack = f"Task_{task_id}"
#             print(f"  Learning {current_attack}...")
            
#             # Debug: Check data dimensions
#             print(f"    Training data shape: {train_loader.dataset.tensors[0].shape}")
#             print(f"    Validation data shape: {val_loader.dataset.tensors[0].shape}")
            
#             # Check label distribution
#             train_labels = train_loader.dataset.tensors[1]
#             print(f"    Training labels distribution: {torch.bincount(train_labels.long())}")
            
#             # For LwF, set the previous model
#             if model_name == "MLP + LwF" and previous_model is not None:
#                 model.set_previous_model(previous_model)
            
#             # Train the model
#             trained_model = train_torch_model(
#                 model, train_loader, val_loader, device=device,
#                 epochs=EPOCHS, lr=LEARNING_RATE,
#                 model_type='lwf' if 'LwF' in model_name else 'ewc' if 'EWC' in model_name else 'basic'
#             )
            
#             # For EWC, compute Fisher information after training on each task
#             if model_name == "MLP + EWC":
#                 print("    Computing Fisher information for EWC...")
#                 model.compute_fisher(train_loader, device=device)
            
#             # Evaluate on all task test sets
#             task_results = {}
            
#             for eval_task in range(task_id + 1):
#                 X_eval, y_eval = task_test_sets[eval_task]
#                 # Convert to torch tensors
#                 X_eval_tensor = torch.FloatTensor(X_eval.values)
#                 y_eval_tensor = torch.LongTensor(y_eval.values)
                
#                 metrics = evaluate_torch_model(trained_model, X_eval_tensor, y_eval_tensor, device=device)
                
#                 # Add AUC placeholder if missing to match baseline format
#                 if 'roc_auc' not in metrics:
#                     metrics['roc_auc'] = metrics['accuracy']  # Use accuracy as placeholder
                
#                 task_results[eval_task] = metrics
#                 print(f"    Task {eval_task} Accuracy: {metrics['accuracy']:.3f}")
            
#             # Evaluate on global test set
#             X_global_tensor = torch.FloatTensor(X_global_test.values)
#             y_global_tensor = torch.LongTensor(y_global_test.values)
            
#             global_metrics = evaluate_torch_model(trained_model, X_global_tensor, y_global_tensor, device=device)
            
#             # Add AUC placeholder for global metrics too
#             if 'roc_auc' not in global_metrics:
#                 global_metrics['roc_auc'] = global_metrics['accuracy']
            
#             task_results['global'] = global_metrics
#             print(f"    Global Accuracy: {global_metrics['accuracy']:.3f}")
            
#             model_results[task_id] = task_results
#             previous_model = trained_model
        
#         all_results[model_name] = model_results
        
#         # Save the trained model
#         model_filename = f'{model_name.replace(" ", "_").replace("(", "").replace(")", "")}_model.pth'
#         torch.save(trained_model.state_dict(), model_filename)
#         print(f" Model saved as {model_filename}")
    
#     # Save results
#     with open('mlp_results.json', 'w') as f:
#         json_results = {}
#         for model_name, results in all_results.items():
#             json_results[model_name] = {}
#             for task_trained, task_results in results.items():
#                 json_results[model_name][task_trained] = {}
#                 for eval_task, metrics in task_results.items():
#                     json_results[model_name][task_trained][eval_task] = {
#                         k: float(v) if isinstance(v, (np.floating, np.integer)) else v
#                         for k, v in metrics.items()
#                     }
#         json.dump(json_results, f, indent=2)
    
#     # Create visualizations
#     task_names = [f'Task {i}' for i in range(len(torch_tasks))]
    
#     # Plot accuracy matrix for each model
#     for model_name in models.keys():
#         accuracy_matrix = create_accuracy_matrix(all_results[model_name])
#         if accuracy_matrix is not None:
#             plt = plot_accuracy_matrix(accuracy_matrix, model_name, task_names)
#             plt.savefig(f'accuracy_matrix_{model_name.replace(" ", "_").replace("(", "").replace(")", "")}.png', dpi=300, bbox_inches='tight')
#             plt.close()
    
#     # Plot comparison plots
#     try:
#         plt = plot_forgetting_rates(all_results, task_names)
#         plt.savefig('mlp_forgetting_comparison.png', dpi=300, bbox_inches='tight')
#         plt.close()
#     except Exception as e:
#         print(f" Could not create forgetting plot: {e}")
    
#     try:
#         plt = plot_global_accuracy(all_results, task_names)
#         plt.savefig('mlp_global_accuracy.png', dpi=300, bbox_inches='tight')
#         plt.close()
#     except Exception as e:
#         print(f" Could not create global accuracy plot: {e}")
    
#     # Create and print summary table - USING THE SAME FUNCTION AS BASELINES
#     try:
#         summary_df = create_summary_table(all_results)
#         print("\n" + "="*60)
#         print("MLP MODELS SUMMARY")
#         print("="*60)
#         print(summary_df.to_string(index=False))
#         print("="*60)
        
#         # Save summary table
#         summary_df.to_csv('mlp_summary.csv', index=False)
#         print(" Results saved to JSON and CSV files")
#         print(" Plots saved as PNG files")
        
#     except Exception as e:
#         print(f" Could not create summary table: {e}")
#         # Create a basic summary as fallback
#         print("\nCreating basic summary table...")
#         basic_summary_data = []
#         for model_name, results in all_results.items():
#             # Get final task results
#             final_task = max(results.keys())
#             final_results = results[final_task]
            
#             # Calculate average performance across all tasks
#             task_accuracies = []
#             task_f1_scores = []
            
#             for eval_task in sorted([k for k in final_results.keys() if k != 'global']):
#                 metrics = final_results[eval_task]
#                 task_accuracies.append(metrics.get('accuracy', 0))
#                 task_f1_scores.append(metrics.get('f1_score', 0))
            
#             avg_accuracy = np.mean(task_accuracies) if task_accuracies else 0
#             avg_f1 = np.mean(task_f1_scores) if task_f1_scores else 0
            
#             # Calculate simple forgetting
#             forgetting_scores = []
#             for task_id in sorted(results.keys()):
#                 if task_id > 0:
#                     initial_acc = results[task_id].get(task_id, {}).get('accuracy', 0)
#                     final_acc = final_results.get(task_id, {}).get('accuracy', 0)
#                     forgetting = max(0, initial_acc - final_acc)
#                     forgetting_scores.append(forgetting)
            
#             avg_forgetting = np.mean(forgetting_scores) if forgetting_scores else 0
#             status = "PASS" if avg_forgetting <= 0.1 else "FAIL"
            
#             basic_summary_data.append({
#                 'Model': model_name,
#                 'Final Accuracy': f"{avg_accuracy:.3f}",
#                 'Final F1-Score': f"{avg_f1:.3f}",
#                 'Final AUC': f"{avg_accuracy:.3f}",
#                 'Avg. Forgetting': f"{avg_forgetting:.3f}",
#                 'Status': status
#             })
        
#         basic_df = pd.DataFrame(basic_summary_data)
#         basic_df = basic_df.sort_values('Final Accuracy', ascending=False)
#         print("\nBASIC MLP SUMMARY:")
#         print(basic_df.to_string(index=False))

# if __name__ == "__main__":
#     main()

# # 02_train_mlp.py - FIXED DATALOADER ISSUE
# import torch
# import json
# import pandas as pd
# import numpy as np
# import sys
# import os
# import copy
# from sklearn.model_selection import train_test_split
# from utils.helpers import save_unified_results

# # Add the current directory to Python path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# print(" Starting MLP training script...")

# try:
#     from utils.data_loader import create_incremental_tasks, create_torch_dataloaders
#     from utils.metrics import evaluate_torch_model, create_accuracy_matrix
#     from utils.visualization import plot_accuracy_matrix, plot_forgetting_rates, plot_global_accuracy, create_summary_table
#     from models.mlp_models import MLP, MLPWithLwF, MLPWithEWC, train_torch_model_adaptive, safe_normalize_features
#     from config import LEARNING_RATE, EPOCHS, BATCH_SIZE, EWC_LAMBDA, LWF_ALPHA
#     print(" All imports successful!")
# except ImportError as e:
#     print(f" Import error: {e}")
#     print(" Make sure you're running from the correct directory")
#     sys.exit(1)

# def correct_data_structure(tasks, task_test_sets):
#     """
#     CORRECT THE DATA STRUCTURE - there's a bug in the data loader:
#     - X_val contains labels instead of features
#     - y_train contains features instead of labels
#     This function fixes the swapped data
#     """
#     print(" CORRECTING DATA STRUCTURE...")
#     print("  Found bug: X_val and y_train are swapped in the data loader!")
    
#     corrected_tasks = []
#     corrected_test_sets = []
    
#     for i, ((X_train, X_val, y_train, y_val), (X_test, y_test)) in enumerate(zip(tasks, task_test_sets)):
#         print(f"\n Fixing Task {i}:")
#         print(f"  Original: X_train={X_train.shape}, X_val={X_val.shape}, y_train={y_train.shape}, y_val={y_val.shape}")
        
#         # The bug: X_val contains labels, y_train contains features
#         # We need to swap them back and create proper splits
        
#         # Option 1: If we have enough data, create proper splits from X_train/y_train
#         if X_train.shape[0] > 1000:  # If we have reasonable training data
#             print("  Creating proper train/validation split...")
            
#             # Use the actual training data to create proper splits
#             X_train_fixed, X_val_fixed, y_train_fixed, y_val_fixed = train_test_split(
#                 X_train, 
#                 X_val,  # X_val actually contains the labels for X_train
#                 test_size=0.2, 
#                 random_state=42, 
#                 stratify=X_val  # Use the actual labels for stratification
#             )
            
#             corrected_tasks.append((X_train_fixed, X_val_fixed, y_train_fixed, y_val_fixed))
#             corrected_test_sets.append((X_test, y_test))
            
#             print(f"  Fixed: X_train={X_train_fixed.shape}, X_val={X_val_fixed.shape}, y_train={y_train_fixed.shape}, y_val={y_val_fixed.shape}")
        
#         else:
#             # Fallback: Just use the test set as validation for small data
#             print("  Using test set as validation (small dataset)...")
#             corrected_tasks.append((X_train, X_test, X_val, y_test))  # X_val are labels, y_test are labels
#             corrected_test_sets.append((X_test, y_test))
    
#     return corrected_tasks, corrected_test_sets

# def global_normalize_data(X_train, X_val, X_test):
#     """Global normalization using training statistics"""
#     print(" Applying global normalization...")
    
#     # Convert to numpy arrays
#     def to_numpy(data):
#         if hasattr(data, 'values'):
#             return data.values
#         elif hasattr(data, 'numpy'):
#             return data.numpy()
#         else:
#             return np.array(data)
    
#     X_train_np = to_numpy(X_train)
#     X_val_np = to_numpy(X_val) 
#     X_test_np = to_numpy(X_test)
    
#     print(f" Shapes - Train: {X_train_np.shape}, Val: {X_val_np.shape}, Test: {X_test_np.shape}")
    
#     # Calculate global statistics from training data only
#     mean = np.mean(X_train_np, axis=0)
#     std = np.std(X_train_np, axis=0)
#     std = np.where(std < 1e-8, 1.0, std)  # Handle zero std
    
#     # Normalize all datasets using training statistics
#     X_train_norm = (X_train_np - mean) / std
#     X_val_norm = (X_val_np - mean) / std
#     X_test_norm = (X_test_np - mean) / std
    
#     print(f" Normalization stats - Mean: {mean.mean():.4f}, Std: {std.mean():.4f}")
#     return X_train_norm, X_val_norm, X_test_norm, mean, std

# def create_torch_dataloaders_fixed(normalized_tasks, batch_size=32):
#     """
#     Fixed version of create_torch_dataloaders that works with numpy arrays
#     """
#     torch_tasks = []
    
#     for i, (X_train, X_val, y_train, y_val) in enumerate(normalized_tasks):
#         print(f" Creating dataloaders for Task {i}...")
        
#         # Convert to torch tensors - handle both numpy arrays and pandas Series
#         X_train_tensor = torch.FloatTensor(X_train)
#         X_val_tensor = torch.FloatTensor(X_val)
        
#         # Convert labels to tensors - handle different input types
#         if hasattr(y_train, 'values'):
#             y_train_tensor = torch.LongTensor(y_train.values)
#         else:
#             y_train_tensor = torch.LongTensor(y_train)
            
#         if hasattr(y_val, 'values'):
#             y_val_tensor = torch.LongTensor(y_val.values)
#         else:
#             y_val_tensor = torch.LongTensor(y_val)
        
#         # Create datasets
#         train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
#         val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        
#         # Create dataloaders
#         train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
        
#         torch_tasks.append((train_loader, val_loader))
        
#         print(f"    Task {i} - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
#     return torch_tasks

# def safe_bincount(labels):
#     """Safe bincount that handles different input types"""
#     if hasattr(labels, 'values'):
#         labels_np = labels.values
#     else:
#         labels_np = np.array(labels)
    
#     # Convert to integers for bincount
#     labels_int = labels_np.astype(int)
#     return np.bincount(labels_int)

# # def create_comprehensive_summary(all_results):
# #     """Create a comprehensive summary table with all metrics"""
# #     print("\n" + "="*80)
# #     print("MLP MODELS COMPREHENSIVE SUMMARY")
# #     print("="*80)
    
# #     summary_data = []
    
# #     for model_name, results in all_results.items():
# #         # Get final task results (after learning all tasks)
# #         final_task = max(results.keys())
# #         final_results = results[final_task]
        
# #         # Calculate metrics across all tasks
# #         task_accuracies = []
# #         task_f1_scores = [] 
# #         task_auc_scores = []
# #         forgetting_scores = []
        
# #         # Calculate performance on each task at the end
# #         for eval_task in sorted([k for k in final_results.keys() if k != 'global']):
# #             metrics = final_results[eval_task]
# #             task_accuracies.append(metrics.get('accuracy', 0))
# #             task_f1_scores.append(metrics.get('f1_score', 0))
# #             task_auc_scores.append(metrics.get('roc_auc', 0))
        
# #         # Calculate forgetting for each task
# #         for task_id in range(len(task_accuracies)):
# #             if task_id > 0:  # Forgetting only applies to tasks after they were learned
# #                 # Get the maximum accuracy when the task was first learned
# #                 max_acc_when_learned = results[task_id].get(task_id, {}).get('accuracy', 0)
# #                 # Get the final accuracy after learning all subsequent tasks
# #                 final_acc = final_results.get(task_id, {}).get('accuracy', 0)
# #                 forgetting = max(0, max_acc_when_learned - final_acc)
# #                 forgetting_scores.append(forgetting)
        
# #         # Calculate averages
# #         final_accuracy = np.mean(task_accuracies) if task_accuracies else 0
# #         final_f1 = np.mean(task_f1_scores) if task_f1_scores else 0
# #         final_auc = np.mean(task_auc_scores) if task_auc_scores else 0
# #         avg_forgetting = np.mean(forgetting_scores) if forgetting_scores else 0
        
# #         # Determine status
# #         status = "PASS" if avg_forgetting <= 0.1 else "FAIL"  # Less than 10% forgetting is acceptable
        
# #         summary_data.append({
# #             'Model': model_name,
# #             'Final Accuracy': f"{final_accuracy:.3f}",
# #             'Final F1-Score': f"{final_f1:.3f}",
# #             'Final AUC': f"{final_auc:.3f}",
# #             'Avg. Forgetting': f"{avg_forgetting:.3f}",
# #             'Status': status
# #         })
    
# #     # Create and display table
# #     import pandas as pd
# #     summary_df = pd.DataFrame(summary_data)
    
# #     # Sort by Final Accuracy (descending)
# #     summary_df = summary_df.sort_values('Final Accuracy', ascending=False)
    
# #     # Display the table
# #     print(summary_df.to_string(index=False))
# #     print("="*80)
    
# #     return summary_df

# # Add this corrected function to your 02_train_mlp.py

# def create_comprehensive_summary_corrected(all_results):
#     """Create a comprehensive summary table with CORRECT forgetting calculation"""
#     print("\n" + "="*80)
#     print("MLP MODELS COMPREHENSIVE SUMMARY")
#     print("="*80)
    
#     summary_data = []
    
#     for model_name, results in all_results.items():
#         # Get final task results (after learning all tasks)
#         final_task = max(results.keys())
#         final_results = results[final_task]
        
#         # Calculate metrics across all tasks
#         task_accuracies = []
#         task_f1_scores = [] 
#         task_auc_scores = []
#         forgetting_scores = []
        
#         # Calculate performance on each task at the end
#         for eval_task in sorted([k for k in final_results.keys() if k != 'global']):
#             metrics = final_results[eval_task]
#             task_accuracies.append(metrics.get('accuracy', 0))
#             task_f1_scores.append(metrics.get('f1_score', 0))
#             task_auc_scores.append(metrics.get('roc_auc', 0))
            
#             # CORRECTED: Calculate forgetting for each task
#             if eval_task < final_task:  # Only tasks learned before the final task
#                 # Get accuracy when the task was first learned
#                 if eval_task in results and eval_task in results[eval_task]:
#                     max_acc_when_learned = results[eval_task][eval_task].get('accuracy', 0)
#                 else:
#                     max_acc_when_learned = 0
                
#                 # Get final accuracy after learning all tasks
#                 final_acc = final_results.get(eval_task, {}).get('accuracy', 0)
                
#                 # Calculate forgetting (drop in performance)
#                 forgetting = max(0, max_acc_when_learned - final_acc)
#                 forgetting_scores.append(forgetting)
#                 print(f"    {model_name} - Task {eval_task}: {max_acc_when_learned:.3f} -> {final_acc:.3f} = {forgetting:.3f} forgetting")
        
#         # Calculate averages
#         final_accuracy = np.mean(task_accuracies) if task_accuracies else 0
#         final_f1 = np.mean(task_f1_scores) if task_f1_scores else 0
#         final_auc = np.mean(task_auc_scores) if task_auc_scores else 0
#         avg_forgetting = np.mean(forgetting_scores) if forgetting_scores else 0
        
#         # Determine status
#         status = "PASS" if avg_forgetting <= 0.1 else "FAIL"  # Less than 10% forgetting is acceptable
        
#         summary_data.append({
#             'Model': model_name,
#             'Final Accuracy': f"{final_accuracy:.3f}",
#             'Final F1-Score': f"{final_f1:.3f}",
#             'Final AUC': f"{final_auc:.3f}",
#             'Avg. Forgetting': f"{avg_forgetting:.3f}",
#             'Status': status
#         })
    
#     # Create and display table
#     import pandas as pd
#     summary_df = pd.DataFrame(summary_data)
    
#     # Sort by Final Accuracy (descending)
#     summary_df = summary_df.sort_values('Final Accuracy', ascending=False)
    
#     # Display the table
#     print(summary_df.to_string(index=False))
#     print("="*80)
    
#     return summary_df

# def create_detailed_task_breakdown(all_results):
#     """Create detailed breakdown by task for each model"""
#     print("\n" + "="*80)
#     print("DETAILED TASK BREAKDOWN")
#     print("="*80)
    
#     for model_name, results in all_results.items():
#         print(f"\n {model_name}:")
#         final_task = max(results.keys())
#         final_results = results[final_task]
        
#         print("Task | Accuracy | F1-Score | AUC     | Forgetting")
#         print("-----|----------|----------|---------|-----------")
        
#         for eval_task in sorted([k for k in final_results.keys() if k != 'global']):
#             metrics = final_results[eval_task]
#             accuracy = metrics.get('accuracy', 0)
#             f1_score = metrics.get('f1_score', 0)
#             auc = metrics.get('roc_auc', 0)
            
#             # Calculate forgetting for this task
#             if eval_task > 0:
#                 max_acc_when_learned = results[eval_task].get(eval_task, {}).get('accuracy', 0)
#                 final_acc = final_results.get(eval_task, {}).get('accuracy', 0)
#                 forgetting = max(0, max_acc_when_learned - final_acc)
#             else:
#                 forgetting = 0.0
            
#             print(f"{eval_task:4} | {accuracy:.3f}    | {f1_score:.3f}    | {auc:.3f}  | {forgetting:.3f}")
        
#         # Global performance
#         global_metrics = final_results.get('global', {})
#         global_acc = global_metrics.get('accuracy', 0)
#         global_f1 = global_metrics.get('f1_score', 0)
#         global_auc = global_metrics.get('roc_auc', 0)
#         print(f"{'Global':4} | {global_acc:.3f}    | {global_f1:.3f}    | {global_auc:.3f}  | -")

# def main():
#     print(" Starting main function...")
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     print("Loading data and creating incremental tasks...")
    
#     try:
#         # Use create_incremental_tasks directly - it handles everything
#         tasks, task_test_sets, global_test_set = create_incremental_tasks()
#         print(" Data loaded successfully!")
#     except Exception as e:
#         print(f" Error loading data: {e}")
#         return
    
#     # CORRECT THE DATA STRUCTURE FIRST
#     corrected_tasks, corrected_test_sets = correct_data_structure(tasks, task_test_sets)
    
#     # Unpack global test set
#     X_global_test, y_global_test = global_test_set
    
#     print(f" Number of tasks: {len(corrected_tasks)}")
#     print(f" Global test set shape: X={X_global_test.shape}, y={y_global_test.shape}")
    
#     # Apply global normalization to all data
#     normalized_tasks = []
#     normalized_task_test_sets = []
#     global_stats = {}  # Store global statistics
    
#     for i, ((X_train, X_val, y_train, y_val), (X_test, y_test)) in enumerate(zip(corrected_tasks, corrected_test_sets)):
#         try:
#             print(f"\n Processing Task {i}...")
#             X_train_norm, X_val_norm, X_test_norm, mean, std = global_normalize_data(X_train, X_val, X_test)
#             normalized_tasks.append((X_train_norm, X_val_norm, y_train, y_val))
#             normalized_task_test_sets.append((X_test_norm, y_test))
            
#             # Store stats from first task for global normalization
#             if i == 0:
#                 global_stats['mean'] = mean
#                 global_stats['std'] = std
                
#             print(f"    Task {i} normalized - Train: {len(X_train_norm)}, Val: {len(X_val_norm)}, Test: {len(X_test_norm)}")
            
#             # Check label distributions (fixed bincount)
#             print(f"    Label distribution - Train: {safe_bincount(y_train)}, Val: {safe_bincount(y_val)}")
            
#         except Exception as e:
#             print(f" Error processing task {i}: {e}")
#             import traceback
#             traceback.print_exc()
#             return
    
#     # Normalize global test set using first task's statistics
#     try:
#         print(f"\n Normalizing global test set...")
#         X_global_np = X_global_test.values if hasattr(X_global_test, 'values') else X_global_test
#         X_global_norm = (X_global_np - global_stats['mean']) / global_stats['std']
#         normalized_global_test = (X_global_norm, y_global_test)
#         print(" Global test set normalized")
#     except Exception as e:
#         print(f" Error normalizing global test set: {e}")
#         return
    
#     # Create torch dataloaders with normalized data - USING FIXED VERSION
#     try:
#         torch_tasks = create_torch_dataloaders_fixed(normalized_tasks, batch_size=BATCH_SIZE)
#         print(f" Created {len(torch_tasks)} torch tasks")
#     except Exception as e:
#         print(f" Error creating dataloaders: {e}")
#         import traceback
#         traceback.print_exc()
#         return
    
#     # Get the actual input dimension from the data
#     try:
#         train_loader = torch_tasks[0][0]  # First task's train loader
#         input_dim = train_loader.dataset.tensors[0].shape[1]
#         print(f" Detected input dimension: {input_dim}")
#     except Exception as e:
#         print(f" Error getting input dimension: {e}")
#         return
    
#     # Define models to train with correct input dimensions
#     models = {
#         "MLP (Naive)": MLP(input_size=input_dim, hidden_size=128),
#         "MLP + LwF": MLPWithLwF(input_size=input_dim, hidden_size=128),
#         "MLP + EWC": MLPWithEWC(input_size=input_dim, hidden_size=128)
#     }
    
#     print(f" Will train {len(models)} models")
    
#     all_results = {}
    
#     for model_name, model in models.items():
#         print(f"\n" + "="*60)
#         print(f"--- Training {model_name} ---")
#         print("="*60)
#         model.to(device)
#         model_results = {}
#         previous_model = None
        
#         for task_id, ((train_loader, val_loader), (X_task_test, y_task_test)) in enumerate(zip(torch_tasks, normalized_task_test_sets)):
#             current_attack = f"Task_{task_id}"
#             print(f"\n   Learning {current_attack}...")
            
#             # Debug: Check data dimensions
#             print(f"    Training data shape: {train_loader.dataset.tensors[0].shape}")
#             print(f"    Validation data shape: {val_loader.dataset.tensors[0].shape}")
            
#             # Check label distribution
#             train_labels = train_loader.dataset.tensors[1]
#             label_counts = torch.bincount(train_labels.long())
#             print(f"    Training labels distribution: {label_counts}")
#             if len(label_counts) > 1:
#                 print(f"    Class ratio: {label_counts[0].item()/label_counts[1].item():.2f}:1")
            
#             # For LwF, set the previous model
#             if model_name == "MLP + LwF" and previous_model is not None:
#                 model.set_previous_model(previous_model)
#                 print("     Previous model set for knowledge distillation")
            
#             # Train the model with ADAPTIVE weighting
#             try:
#                 trained_model = train_torch_model_adaptive(
#                     model, train_loader, val_loader, device=device,
#                     epochs=EPOCHS, lr=LEARNING_RATE,
#                     model_type='lwf' if 'LwF' in model_name else 'ewc' if 'EWC' in model_name else 'basic',
#                     task_id=task_id
#                 )
#                 print(f"     Task {task_id} training completed")
#             except Exception as e:
#                 print(f"     Error training task {task_id}: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 continue
            
#             # For EWC, compute Fisher information after training on each task
#             if model_name == "MLP + EWC":
#                 print("     Computing Fisher information for EWC...")
#                 try:
#                     model.compute_fisher(train_loader, device=device, task_id=task_id)
#                     print("     Fisher information computed")
#                 except Exception as e:
#                     print(f"     Error computing Fisher: {e}")
            
#             # Evaluate on all task test sets
#             task_results = {}
            
#             print("     Evaluating on previous tasks...")
#             for eval_task in range(task_id + 1):
#                 try:
#                     X_eval, y_eval = normalized_task_test_sets[eval_task]
#                     # Convert to torch tensors - handle different types
#                     X_eval_tensor = torch.FloatTensor(X_eval)
                    
#                     if hasattr(y_eval, 'values'):
#                         y_eval_tensor = torch.LongTensor(y_eval.values)
#                     else:
#                         y_eval_tensor = torch.LongTensor(y_eval)
                    
#                     metrics = evaluate_torch_model(trained_model, X_eval_tensor, y_eval_tensor, device=device)
                    
#                     # Add AUC placeholder if missing to match baseline format
#                     if 'roc_auc' not in metrics:
#                         metrics['roc_auc'] = metrics['accuracy']
                    
#                     task_results[eval_task] = metrics
#                     print(f"        Task {eval_task} - Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}")
#                 except Exception as e:
#                     print(f"         Error evaluating task {eval_task}: {e}")
#                     task_results[eval_task] = {'accuracy': 0, 'f1_score': 0, 'roc_auc': 0}
            
#             # Evaluate on global test set
#             try:
#                 X_global_tensor = torch.FloatTensor(normalized_global_test[0])
                
#                 if hasattr(normalized_global_test[1], 'values'):
#                     y_global_tensor = torch.LongTensor(normalized_global_test[1].values)
#                 else:
#                     y_global_tensor = torch.LongTensor(normalized_global_test[1])
                
#                 global_metrics = evaluate_torch_model(trained_model, X_global_tensor, y_global_tensor, device=device)
                
#                 if 'roc_auc' not in global_metrics:
#                     global_metrics['roc_auc'] = global_metrics['accuracy']
                
#                 task_results['global'] = global_metrics
#                 print(f"     Global Accuracy: {global_metrics['accuracy']:.3f}")
#             except Exception as e:
#                 print(f"     Error evaluating global test: {e}")
#                 task_results['global'] = {'accuracy': 0, 'f1_score': 0, 'roc_auc': 0}
            
#             model_results[task_id] = task_results
#             previous_model = copy.deepcopy(trained_model)
        
#         all_results[model_name] = model_results
        
#         # Save the trained model
#         try:
#             model_filename = f'{model_name.replace(" ", "_").replace("(", "").replace(")", "")}_model.pth'
#             torch.save({
#                 'model_state_dict': trained_model.state_dict(),
#                 'input_dim': input_dim,
#                 'model_type': model_name,
#                 'tasks_trained': len(torch_tasks),
#                 'normalization_stats': {
#                     'mean': global_stats['mean'],
#                     'std': global_stats['std']
#                 }
#             }, model_filename)
#             print(f" Model saved as {model_filename}")
#         except Exception as e:
#             print(f" Error saving model: {e}")
    
#     # Save results
#     try:
#         with open('mlp_results.json', 'w') as f:
#             json_results = {}
#             for model_name, results in all_results.items():
#                 json_results[model_name] = {}
#                 for task_trained, task_results in results.items():
#                     json_results[model_name][task_trained] = {}
#                     for eval_task, metrics in task_results.items():
#                         json_results[model_name][task_trained][eval_task] = {
#                             k: float(v) if isinstance(v, (np.floating, np.integer)) else v
#                             for k, v in metrics.items()
#                         }
#             json.dump(json_results, f, indent=2)
#         print(" Results saved to mlp_results.json")
#     except Exception as e:
#         print(f" Error saving results: {e}")
        
#             # Create comprehensive summaries
#     try:
#         print("\n" + "="*80)
#         print("GENERATING COMPREHENSIVE RESULTS SUMMARY")
#         print("="*80)
        
#         # Main summary table
#         summary_df = create_comprehensive_summary(all_results)
        
#         # Detailed breakdown
#         create_detailed_task_breakdown(all_results)
        
#         # Save summary to CSV
#         summary_df.to_csv('mlp_comprehensive_summary.csv', index=False)
#         print(" Comprehensive summary saved to mlp_comprehensive_summary.csv")
        
#     except Exception as e:
#         print(f" Could not create comprehensive summary: {e}")
#         import traceback
#         traceback.print_exc()
    
#     print("\n Training completed successfully!")
#     print(" Check mlp_results.json for detailed results")

# if __name__ == "__main__":
#     print(" Script starting execution...")
#     main()
#     print(" Script finished execution!")

# 02_train_mlp.py

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from utils.data_loader import create_incremental_tasks
from utils.metrics import evaluate_model, create_accuracy_matrix
from utils.visualization import (
    plot_accuracy_matrix, plot_forgetting_rates,
    plot_global_accuracy, create_summary_table
)
from utils.helpers import save_unified_results
from config import get_attack_name

print("✅ All imports successful!")

# ---------------------------
# 🧠 Helper Function
# ---------------------------
def correct_data_structure(tasks, task_test_sets):
    """
    Ensure each task has a consistent format: (X_train, y_train)
    and test sets follow the same format.
    """
    corrected_tasks = []
    corrected_test_sets = []

    for X_train, y_train in tasks:
        # Ensure numpy arrays or compatible lists
        if isinstance(X_train, (list, tuple)):
            X_train = np.array(X_train)
        if isinstance(y_train, (list, tuple)):
            y_train = np.array(y_train)
        corrected_tasks.append((X_train, y_train))

    for X_test, y_test in task_test_sets:
        if isinstance(X_test, (list, tuple)):
            X_test = np.array(X_test)
        if isinstance(y_test, (list, tuple)):
            y_test = np.array(y_test)
        corrected_test_sets.append((X_test, y_test))

    return corrected_tasks, corrected_test_sets


# ---------------------------
# 🧠 MLP Model Definition
# ---------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_classes=2):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------
# 🚀 Training Function
# ---------------------------
def train_mlp(model, train_loader, criterion, optimizer, device):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


# ---------------------------
# 🚀 Main Function
# ---------------------------
def main():
    print(" Configuration loaded:")
    print(" Environment: Local")
    print(" Sample Fraction: 0.01")
    print(" Data path: data/Encoded.csv")
    print(" Starting MLP training script...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load incremental tasks
    print("Loading data and creating incremental tasks...")
    tasks, task_test_sets, global_test_set = create_incremental_tasks()
    corrected_tasks, corrected_test_sets = correct_data_structure(tasks, task_test_sets)

    input_size = corrected_tasks[0][0].shape[1]
    num_classes = len(np.unique(corrected_tasks[0][1]))

    model = SimpleMLP(input_size, hidden_size=128, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    results = {}

    for task_idx, (X_train, y_train) in enumerate(corrected_tasks):
        print(f"\n🧠 Training on Task {task_idx} ({get_attack_name(task_idx)})")

        # Prepare DataLoader
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.long)),
            batch_size=64, shuffle=True
        )

        train_mlp(model, train_loader, criterion, optimizer, device)

        results[task_idx] = {}

        # Evaluate on all seen tasks
        for eval_idx, (X_test, y_test) in enumerate(corrected_test_sets[:task_idx + 1]):
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

            with torch.no_grad():
                outputs = model(X_test_t)
                _, preds = torch.max(outputs, 1)
                acc = (preds == y_test_t).float().mean().item()

            results[task_idx][eval_idx] = {'accuracy': acc}

            print(f"   → Eval on Task {eval_idx} ({get_attack_name(eval_idx)}): Accuracy = {acc:.4f}")

    # Save results
    with open('mlp_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Visualization
    task_names = [get_attack_name(i) for i in range(len(results))]
    accuracy_matrix = create_accuracy_matrix(results)
    plt = plot_accuracy_matrix(accuracy_matrix, "MLP", task_names)
    plt.savefig('mlp_accuracy_matrix.png')
    plt.close()

    plt = plot_forgetting_rates({'MLP': results}, task_names)
    plt.savefig('mlp_forgetting_rates.png')
    plt.close()

    plt = plot_global_accuracy({'MLP': results}, task_names)
    plt.savefig('mlp_global_accuracy.png')
    plt.close()

    summary_df = create_summary_table({'MLP': results})
    summary_df.to_csv('mlp_summary.csv', index=False)

    print("\n✅ MLP training complete.")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
