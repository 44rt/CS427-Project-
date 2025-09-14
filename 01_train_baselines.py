# 01_train_baselines.py
import json
import pandas as pd
from utils.data_loader import create_incremental_tasks
from utils.metrics import evaluate_model, create_accuracy_matrix
from utils.visualization import plot_accuracy_matrix, plot_forgetting_rates, plot_global_accuracy, create_summary_table
from models.baseline_models import get_baseline_models, train_baseline_model

def main():
    print("Loading data and creating incremental tasks...")
    tasks, task_test_sets, global_test_set = create_incremental_tasks()
    
    baseline_models = get_baseline_models()
    all_results = {}
    
    for model_name, model in baseline_models.items():
        results = train_baseline_model(model, model_name, tasks, task_test_sets, global_test_set)
        all_results[model_name] = results
    
    # Save results
    with open('baseline_results.json', 'w') as f:
        # Convert numpy values to Python floats for JSON serialization
        json_results = {}
        for model_name, results in all_results.items():
            json_results[model_name] = {}
            for task_trained, task_results in results.items():
                json_results[model_name][task_trained] = {}
                for eval_task, metrics in task_results.items():
                    json_results[model_name][task_trained][eval_task] = {
                        k: float(v) if isinstance(v, (np.floating, np.integer)) else v.tolist() if hasattr(v, 'tolist') else v
                        for k, v in metrics.items()
                    }
        json.dump(json_results, f, indent=2)
    
    # Create visualizations
    task_names = [f'Task {i}' for i in range(len(tasks))]
    
    # Plot accuracy matrix for each model
    for model_name in baseline_models.keys():
        accuracy_matrix = create_accuracy_matrix(all_results[model_name])
        plt = plot_accuracy_matrix(accuracy_matrix, model_name, task_names)
        plt.savefig(f'accuracy_matrix_{model_name.replace(" ", "_")}.png')
        plt.close()
    
    # Plot comparison plots
    plt = plot_forgetting_rates(all_results, task_names)
    plt.savefig('forgetting_comparison.png')
    plt.close()
    
    plt = plot_global_accuracy(all_results, task_names)
    plt.savefig('global_accuracy.png')
    plt.close()
    
    # Create and print summary table
    summary_df = create_summary_table(all_results)
    print("\n" + "="*60)
    print("BASELINE MODELS SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60)
    
    # Save summary table
    summary_df.to_csv('baseline_summary.csv', index=False)
    print("Results saved to JSON and CSV files")
    print("Plots saved as PNG files")

if __name__ == "__main__":
    main()