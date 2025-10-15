# # 01_train_baselines.py
# import numpy as np
# import json
# import pandas as pd
# from utils.data_loader import create_incremental_tasks
# from utils.metrics import evaluate_model, create_accuracy_matrix
# from utils.visualization import plot_accuracy_matrix, plot_forgetting_rates, plot_global_accuracy, create_summary_table
# from models.baseline_models import get_baseline_models, train_baseline_model
# from utils.helpers import save_unified_results

# def main():
#     print("Loading data and creating incremental tasks...")
#     tasks, task_test_sets, global_test_set = create_incremental_tasks()
    
#     baseline_models = get_baseline_models()
#     all_results = {}
    
#     for model_name, model in baseline_models.items():
#         results = train_baseline_model(model, model_name, tasks, task_test_sets, global_test_set)
#         all_results[model_name] = results
    
#     # Save results
#     with open('baseline_results.json', 'w') as f:
#         # Convert numpy values to Python floats for JSON serialization
#         json_results = {}
#         for model_name, results in all_results.items():
#             json_results[model_name] = {}
#             for task_trained, task_results in results.items():
#                 json_results[model_name][task_trained] = {}
#                 for eval_task, metrics in task_results.items():
#                     json_results[model_name][task_trained][eval_task] = {
#                         k: float(v) if isinstance(v, (np.floating, np.integer)) else v.tolist() if hasattr(v, 'tolist') else v
#                         for k, v in metrics.items()
#                     }
#         json.dump(json_results, f, indent=2)
    
#     # Create visualizations
#     task_names = [f'Task {i}' for i in range(len(tasks))]
    
#     # Plot accuracy matrix for each model
#     for model_name in baseline_models.keys():
#         accuracy_matrix = create_accuracy_matrix(all_results[model_name])
#         plt = plot_accuracy_matrix(accuracy_matrix, model_name, task_names)
#         plt.savefig(f'accuracy_matrix_{model_name.replace(" ", "_")}.png')
#         plt.close()
    
#     # Plot comparison plots
#     plt = plot_forgetting_rates(all_results, task_names)
#     plt.savefig('forgetting_comparison.png')
#     plt.close()
    
#     plt = plot_global_accuracy(all_results, task_names)
#     plt.savefig('global_accuracy.png')
#     plt.close()
    
#     # Create and print summary table
#     summary_df = create_summary_table(all_results)
#     print("\n" + "="*60)
#     print("BASELINE MODELS SUMMARY")
#     print("="*60)
#     print(summary_df.to_string(index=False))
#     print("="*60)
    
#     # Save summary table
#     summary_df.to_csv('baseline_summary.csv', index=False)
#     print("Results saved to JSON and CSV files")
#     print("Plots saved as PNG files")

# if __name__ == "__main__":
#     main()

# 01_train_baselines.py
import numpy as np
import json
import pandas as pd
from utils.data_loader import create_incremental_tasks
from utils.metrics import evaluate_model, create_accuracy_matrix
from utils.visualization import plot_accuracy_matrix, plot_forgetting_rates, plot_global_accuracy, create_summary_table
from models.baseline_models import get_baseline_models, train_baseline_model
from utils.helpers import save_unified_results
import matplotlib.pyplot as plt  # Make sure plt is imported for plotting

def main():
    print("Loading data and creating incremental tasks...")
    tasks, task_test_sets, global_test_set = create_incremental_tasks()
    
    baseline_models = get_baseline_models()
    all_results = {}
    
    for model_name, model in baseline_models.items():
        results = train_baseline_model(model, model_name, tasks, task_test_sets, global_test_set)
        all_results[model_name] = results
    
    # Save results locally
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
    
    # Save to unified JSON (NEW)
    save_unified_results("Baseline", all_results)
    
    # Save summary table locally
    summary_df.to_csv('baseline_summary.csv', index=False)
    print("Results saved to JSON and CSV files")
    print("Plots saved as PNG files")

if __name__ == "__main__":
    main()

# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
# import joblib
# import json

# # ==============================
# # Configuration
# # ==============================
# DATA_PATH = "data/Encoded.csv"
# OUTPUT_DIR = "outputs"
# BASELINE_MODEL_PATH = os.path.join(OUTPUT_DIR, "baseline_model.pkl")
# METRICS_PATH = os.path.join(OUTPUT_DIR, "baseline_metrics.json")

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# print("🔧 Configuration Loaded:")
# print(f"🎯 Environment: Local")
# print(f"🎯 Data path: {DATA_PATH}\n")

# # ==============================
# # Load and Prepare Data
# # ==============================
# print("📥 Loading data...")
# data = pd.read_csv(DATA_PATH)

# if 'label' not in data.columns:
#     print("❌ Error: No 'label' column found in dataset.")
#     exit()

# X = data.drop(columns=['label'])
# y = data['label']

# print(f"✅ Data loaded successfully: {data.shape[0]} samples, {data.shape[1]} columns")
# print(f"🧩 Features: {X.shape[1]} | Target: {len(set(y))} classes\n")

# # ==============================
# # Encode and Split Data
# # ==============================
# print("⚙️ Preprocessing data...")
# encoder = LabelEncoder()
# y_encoded = encoder.fit_transform(y)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
# )
# print(f"✅ Split complete: {X_train.shape[0]} train / {X_test.shape[0]} test\n")

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # ==============================
# # Train Logistic Regression Model
# # ==============================
# print("🚀 Training Logistic Regression baseline model...")
# baseline_model = LogisticRegression(max_iter=500)
# baseline_model.fit(X_train_scaled, y_train)
# print("✅ Training complete!\n")

# # ==============================
# # Evaluate Model
# # ==============================
# y_pred = baseline_model.predict(X_test_scaled)
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred, output_dict=True)

# print("📊 Evaluation Results:")
# print(f"Accuracy: {accuracy:.4f}")
# print("Detailed Classification Report:")
# print(json.dumps(report, indent=4))
# print("\n")

# # ==============================
# # Save Model and Metrics
# # ==============================
# print("💾 Saving model and metrics...")
# joblib.dump(baseline_model, BASELINE_MODEL_PATH)

# metrics_data = {
#     "model": "Logistic Regression",
#     "accuracy": accuracy,
#     "classification_report": report
# }
# with open(METRICS_PATH, "w") as f:
#     json.dump(metrics_data, f, indent=4)

# print(f"✅ Model saved to: {BASELINE_MODEL_PATH}")
# print(f"✅ Metrics saved to: {METRICS_PATH}")
# print("🏁 Baseline training complete!\n")
