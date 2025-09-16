# # models/baseline_models.py
# from utils.metrics import evaluate_model
# from sklearn.linear_model import SGDClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# import numpy as np
# import joblib

# def get_baseline_models():
#     """Return dictionary of baseline models"""
#     return {
#         "Perceptron": SGDClassifier(
#             loss='perceptron', eta0=1, learning_rate='constant',
#             penalty=None, random_state=42
#         ),
#         "Logistic Regression": SGDClassifier(
#             loss='log_loss', random_state=42
#         ),
#         "SVM": SVC(
#             kernel='linear', random_state=42, probability=True
#         ),
#         "Random Forest": RandomForestClassifier(
#             n_estimators=100, random_state=42
#         )
#     }

# def train_baseline_model(model, model_name, tasks, task_test_sets, global_test_set):
#     """Train a baseline model incrementally"""
#     X_global_test, y_global_test = global_test_set
#     results = {}
    
#     print(f"\n--- Training {model_name} ---")
    
#     for task_id, ((X_train, y_train, X_val, y_val), (X_task_test, y_task_test)) in enumerate(zip(tasks, task_test_sets)):
#         print(f"  Learning Task {task_id}...")
        
#         # Train the model
#         if hasattr(model, 'partial_fit'):
#             model.partial_fit(X_train, y_train, classes=np.array([0, 1]))
#         else:
#             # For models without partial_fit, we need to retrain from scratch
#             # This is not ideal but works for demonstration
#             model.fit(X_train, y_train)
        
#         # Evaluate on all task test sets and global test set
#         task_results = {}
        
#         for eval_task in range(task_id + 1):
#             X_eval, y_eval = task_test_sets[eval_task]
#             metrics = evaluate_model(model, X_eval, y_eval)
#             task_results[eval_task] = metrics
#             print(f"    Task {eval_task} Accuracy: {metrics['accuracy']:.3f}")
        
#         # Evaluate on global test set
#         global_metrics = evaluate_model(model, X_global_test, y_global_test)
#         task_results['global'] = global_metrics
#         print(f"    Global Accuracy: {global_metrics['accuracy']:.3f}")
        
#         results[task_id] = task_results
    
#     # Save the trained model
#     joblib.dump(model, f'{model_name.replace(" ", "_")}_model.joblib')
    
#     return results

# # models/baseline_models.py
# from utils.metrics import evaluate_model
# from sklearn.linear_model import SGDClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# import numpy as np
# import joblib

# def get_baseline_models():
#     """Return dictionary of baseline models"""
#     return {
#         "Perceptron": SGDClassifier(
#             loss='perceptron', eta0=1, learning_rate='constant',
#             penalty=None, random_state=42
#         ),
#         "Logistic Regression": SGDClassifier(
#             loss='log_loss', random_state=42
#         ),
#         "SVM": SVC(
#             kernel='linear', random_state=42, probability=True
#         ),
#         "Random Forest": RandomForestClassifier(
#             n_estimators=100, random_state=42
#         )
#     }

# def train_baseline_model(model, model_name, tasks, task_test_sets, global_test_set):
#     """Train a baseline model incrementally"""
#     X_global_test, y_global_test = global_test_set
#     results = {}
    
#     print(f"\n--- Training {model_name} ---")
    
#     for task_id, ((X_train, y_train, X_val, y_val), (X_task_test, y_task_test)) in enumerate(zip(tasks, task_test_sets)):
#         print(f"  Learning Task {task_id}...")
        
#         # --- CRITICAL FIX: Check for single-class data ---
#         unique_classes = np.unique(y_train)
#         if len(unique_classes) < 2:
#             print(f"    ⚠️  Only one class present ({unique_classes[0]}). Skipping training for this task.")
#             # Still evaluate on previous tasks to maintain result structure
#             task_results = {}
#             for eval_task in range(task_id + 1):
#                 X_eval, y_eval = task_test_sets[eval_task]
#                 metrics = evaluate_model(model, X_eval, y_eval)
#                 task_results[eval_task] = metrics
#                 print(f"    Task {eval_task} Accuracy: {metrics['accuracy']:.3f}")
            
#             global_metrics = evaluate_model(model, X_global_test, y_global_test)
#             task_results['global'] = global_metrics
#             print(f"    Global Accuracy: {global_metrics['accuracy']:.3f}")
            
#             results[task_id] = task_results
#             continue
#         # --- END FIX ---
        
#         # Train the model (original code)
#         if hasattr(model, 'partial_fit'):
#             model.partial_fit(X_train, y_train, classes=np.array([0, 1]))
#         else:
#             model.fit(X_train, y_train)
        
#         # Evaluate on all task test sets and global test set
#         task_results = {}
        
#         for eval_task in range(task_id + 1):
#             X_eval, y_eval = task_test_sets[eval_task]
#             metrics = evaluate_model(model, X_eval, y_eval)
#             task_results[eval_task] = metrics
#             print(f"    Task {eval_task} Accuracy: {metrics['accuracy']:.3f}")
        
#         # Evaluate on global test set
#         global_metrics = evaluate_model(model, X_global_test, y_global_test)
#         task_results['global'] = global_metrics
#         print(f"    Global Accuracy: {global_metrics['accuracy']:.3f}")
        
#         results[task_id] = task_results
    
#     # Save the trained model
#     joblib.dump(model, f'{model_name.replace(" ", "_")}_model.joblib')
    
#     return results

# models/baseline_models.py
from utils.metrics import evaluate_model
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib

def get_baseline_models():
    """Return dictionary of baseline models"""
    return {
        "Perceptron": SGDClassifier(
            loss='perceptron', eta0=1, learning_rate='constant',
            penalty=None, random_state=42
        ),
        "Logistic Regression": SGDClassifier(
            loss='log_loss', random_state=42
        ),
        "SVM": SVC(
            kernel='linear', random_state=42, probability=True
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42
        )
    }

def train_baseline_model(model, model_name, tasks, task_test_sets, global_test_set):
    """Train a baseline model incrementally"""
    X_global_test, y_global_test = global_test_set
    results = {}
    
    print(f"\n--- Training {model_name} ---")
    
    for task_id, ((X_train, y_train, X_val, y_val), (X_task_test, y_task_test)) in enumerate(zip(tasks, task_test_sets)):
        print(f"  Learning Task {task_id}...")
        
        # --- CRITICAL FIX: Check for single-class data ---
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            print(f"    ⚠️  Only one class present ({unique_classes[0]}). Skipping training for this task.")
            # Still evaluate on previous tasks to maintain result structure
            task_results = {}
            for eval_task in range(task_id + 1):
                X_eval, y_eval = task_test_sets[eval_task]
                metrics = evaluate_model(model, X_eval, y_eval)
                task_results[eval_task] = metrics
                print(f"    Task {eval_task} Accuracy: {metrics['accuracy']:.3f}")
            
            global_metrics = evaluate_model(model, X_global_test, y_global_test)
            task_results['global'] = global_metrics
            print(f"    Global Accuracy: {global_metrics['accuracy']:.3f}")
            
            results[task_id] = task_results
            continue
        # --- END FIX ---
        
        # Train the model (original code)
        if hasattr(model, 'partial_fit'):
            model.partial_fit(X_train, y_train, classes=np.array([0, 1]))
        else:
            model.fit(X_train, y_train)
        
        # Evaluate on all task test sets and global test set
        task_results = {}
        
        for eval_task in range(task_id + 1):
            X_eval, y_eval = task_test_sets[eval_task]
            metrics = evaluate_model(model, X_eval, y_eval)
            task_results[eval_task] = metrics
            print(f"    Task {eval_task} Accuracy: {metrics['accuracy']:.3f}")
        
        # Evaluate on global test set
        global_metrics = evaluate_model(model, X_global_test, y_global_test)
        task_results['global'] = global_metrics
        print(f"    Global Accuracy: {global_metrics['accuracy']:.3f}")
        
        results[task_id] = task_results
    
    # Save the trained model
    joblib.dump(model, f'{model_name.replace(" ", "_")}_model.joblib')
    
    return results