# unified_testing_fixed.py - COMPLETE SELF-CONTAINED SCRIPT
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import sys
import os
import copy

# ====== CONFIGURATION ======
DATA_PATH = "data/Encoded.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-4
SAMPLE_FRACTION = 0.01

TASK_ATTACK_ORDER = [
    ['UDPFlood'],        
    ['HTTPFlood'],       
    ['SlowrateDoS'],
]

ATTACK_NAMES = {
    0: "UDPFlood",
    1: "HTTPFlood", 
    2: "SlowrateDoS",
}

def get_attack_name(task_id):
    return ATTACK_NAMES.get(task_id, f"Task_{task_id}")

print("Configuration loaded")

# ====== MLP MODELS (COPIED FROM YOUR CODE) ======
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss()(inputs, targets.unsqueeze(1).float())
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

def create_adaptive_criterion(train_loader, method='auto'):
    train_labels = train_loader.dataset.tensors[1]
    class_counts = torch.bincount(train_labels.long())
    imbalance_ratio = class_counts[0] / class_counts[1]
    
    print(f"    Data imbalance ratio: {imbalance_ratio:.2f}")
    
    if imbalance_ratio < 2:
        print("    Using standard BCEWithLogitsLoss (balanced data)")
        return nn.BCEWithLogitsLoss()
    elif imbalance_ratio < 10:
        pos_weight = torch.tensor([imbalance_ratio])
        print(f"    Using weighted BCEWithLogitsLoss (pos_weight: {imbalance_ratio:.2f})")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        print("    Using Focal Loss (severely imbalanced data)")
        return FocalLoss(alpha=1, gamma=2)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.layers(x)

class MLPWithLwF(MLP):
    def __init__(self, input_size, hidden_size=128):
        super(MLPWithLwF, self).__init__(input_size, hidden_size)
        self.previous_model = None
    
    def set_previous_model(self, previous_model):
        self.previous_model = copy.deepcopy(previous_model)
        self.previous_model.eval()
    
    def compute_loss(self, outputs, targets, previous_outputs=None, alpha=0.5):
        current_loss = nn.BCEWithLogitsLoss()(outputs.squeeze(), targets.float())
        
        if previous_outputs is not None and self.previous_model is not None:
            previous_probs = torch.sigmoid(previous_outputs).detach()
            distillation_loss = nn.BCEWithLogitsLoss()(outputs.squeeze(), previous_probs.squeeze())
            return alpha * current_loss + (1 - alpha) * distillation_loss
        
        return current_loss

class MLPWithEWC(MLP):
    def __init__(self, input_size, hidden_size=128):
        super(MLPWithEWC, self).__init__(input_size, hidden_size)
        self.importance = {}
        self.fisher = {}
        self.previous_params = {}
        self.task_fisher = {}
    
    def compute_fisher(self, data_loader, device='cpu', task_id=None):
        self.eval()
        fisher_dict = {}
        
        for name, param in self.named_parameters():
            fisher_dict[name] = torch.zeros_like(param.data)
        
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            self.zero_grad()
            outputs = self(batch_X)
            loss = nn.BCEWithLogitsLoss()(outputs.squeeze(), batch_y.float())
            loss.backward()
            
            for name, param in self.named_parameters():
                if param.grad is not None:
                    fisher_dict[name] += param.grad.data ** 2 / len(data_loader)
        
        if task_id is not None:
            self.task_fisher[task_id] = fisher_dict
        
        self.fisher = fisher_dict
        self.previous_params = {name: param.data.clone() for name, param in self.named_parameters()}
    
    def compute_ewc_loss(self, current_loss, lamda=1000):
        ewc_loss = current_loss
        for name, param in self.named_parameters():
            if name in self.fisher and name in self.previous_params:
                ewc_loss += (lamda / 2) * torch.sum(
                    self.fisher[name] * (param - self.previous_params[name]) ** 2
                )
        return ewc_loss

def safe_normalize_features(batch_X, eps=1e-8):
    if batch_X.dim() == 1:
        batch_X = batch_X.unsqueeze(0)
    
    std = batch_X.std(dim=0)
    zero_std_mask = std < eps
    
    if zero_std_mask.any():
        normalized = batch_X - batch_X.mean(dim=0)
        normalized[:, zero_std_mask] = 0.0
        non_zero_mask = ~zero_std_mask
        if non_zero_mask.any():
            normalized[:, non_zero_mask] = normalized[:, non_zero_mask] / std[non_zero_mask]
        return normalized
    else:
        return (batch_X - batch_X.mean(dim=0)) / (std + eps)

def debug_model_stability(model, train_loader, device):
    model.train()
    batch_X, batch_y = next(iter(train_loader))
    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
    
    print("=== STABILITY DEBUG ===")
    print(f"Input stats - Mean: {batch_X.mean():.6f}, Std: {batch_X.std():.6f}")
    print(f"Input range: [{batch_X.min():.6f}, {batch_X.max():.6f}]")
    
    outputs = model(batch_X)
    print(f"Output range: [{outputs.min():.6f}, {outputs.max():.6f}]")
    
    loss = nn.BCEWithLogitsLoss()(outputs.squeeze(), batch_y.float())
    loss.backward()
    
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append((name, param.grad.data.norm(2).item()))
    
    print("Gradient norms (first 5):")
    for name, norm in grad_norms[:5]:
        print(f"  {name}: {norm:.6f}")
    
    model.zero_grad()
    print("======================")
    return outputs

def train_torch_model_adaptive(model, train_loader, val_loader, device='cpu', epochs=10, lr=0.001, model_type='basic', task_id=0, max_grad_norm=1.0):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    print("    Configuring adaptive loss function...")
    criterion = create_adaptive_criterion(train_loader)
    
    best_acc = 0
    patience = 3
    patience_counter = 0
    best_model = copy.deepcopy(model.state_dict())
    
    if task_id == 0:
        debug_model_stability(model, train_loader, device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        grad_norms = []
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            batch_X = safe_normalize_features(batch_X)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            if epoch == 0 and batch_idx == 0:
                print(f"    Output range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
                print(f"    Target range: [{batch_y.min().item()}, {batch_y.max().item()}]")
            
            if model_type == 'lwf' and hasattr(model, 'previous_model') and model.previous_model is not None:
                with torch.no_grad():
                    previous_outputs = model.previous_model(batch_X)
                loss = model.compute_loss(outputs, batch_y, previous_outputs)
            else:
                if isinstance(criterion, (nn.BCEWithLogitsLoss, FocalLoss)):
                    loss = criterion(outputs.squeeze(), batch_y.float())
                else:
                    loss = criterion(outputs, batch_y.unsqueeze(1))
            
            if model_type == 'ewc':
                loss = model.compute_ewc_loss(loss)
            
            if torch.isnan(loss):
                print(f"    WARNING: NaN loss detected at epoch {epoch}, batch {batch_idx}")
                continue
            
            loss.backward()
            
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)
            
            if max_grad_norm is not None and total_norm > max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                if epoch == 0 and batch_idx == 0:
                    print(f"    Gradient clipped: {total_norm:.2f} -> {max_grad_norm}")
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0
        
        model.eval()
        val_acc = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                batch_X = safe_normalize_features(batch_X)
                outputs = model(batch_X)
                
                if outputs.dim() > 1 and outputs.shape[1] > 1:
                    preds = torch.argmax(outputs, dim=1)
                else:
                    preds = (torch.sigmoid(outputs) > 0.5).float().squeeze()
                
                val_acc += (preds == batch_y).float().mean().item()
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        val_acc /= len(val_loader)
        
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        pred_0 = np.sum(val_preds == 0)
        pred_1 = np.sum(val_preds == 1)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.3f}, Grad Norm: {avg_grad_norm:.4f}')
        print(f'    Predictions - Class 0: {pred_0}, Class 1: {pred_1}')
        
        if not np.isfinite(train_loss):
            print("    WARNING: Non-finite loss detected, stopping training")
            break
            
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            best_model = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("    Early stopping triggered")
                break
    
    model.load_state_dict(best_model)
    return model

# ====== BASELINE MODELS ======
def get_baseline_models():
    models = {
        "Perceptron": SGDClassifier(
            loss='perceptron', eta0=1, learning_rate='constant',
            penalty=None, random_state=42, max_iter=1000
        ),
        "Logistic Regression": SGDClassifier(
            loss='log_loss', random_state=42, max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=50, random_state=42
        ),
        "SVM": SVC(kernel='linear', random_state=42, probability=True, max_iter=1000)
    }
    return models

# ====== METRICS FUNCTIONS ======
def evaluate_model(model, X, y):
    try:
        if hasattr(model, 'predict_proba'):
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]
        else:
            predictions = model.predict(X)
            probabilities = predictions
        
        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions, zero_division=0)
        
        try:
            auc = roc_auc_score(y, probabilities)
        except:
            auc = 0.5
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': auc
        }
    except Exception as e:
        return {'accuracy': 0.0, 'f1_score': 0.0, 'roc_auc': 0.5}

def evaluate_torch_model(model, X, y, device='cpu'):
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        outputs = model(X)
        
        if outputs.dim() > 1 and outputs.shape[1] > 1:
            predictions = torch.argmax(outputs, dim=1)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
        else:
            predictions = (torch.sigmoid(outputs) > 0.5).float().squeeze()
            probabilities = torch.sigmoid(outputs).squeeze()
        
        predictions = predictions.cpu().numpy()
        probabilities = probabilities.cpu().numpy()
        y_np = y.numpy()
        
        accuracy = accuracy_score(y_np, predictions)
        f1 = f1_score(y_np, predictions, zero_division=0)
        
        try:
            auc = roc_auc_score(y_np, probabilities)
        except:
            auc = 0.5
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': auc
        }

def create_accuracy_matrix(results):
    num_tasks = len(results)
    accuracy_matrix = np.zeros((num_tasks, num_tasks))
    
    for task_trained in results:
        for task_evaluated in results[task_trained]:
            if task_evaluated != 'global':
                accuracy = results[task_trained][task_evaluated]['accuracy']
                accuracy_matrix[task_trained, task_evaluated] = accuracy
    
    return accuracy_matrix

# ====== DATA LOADING ======
def load_and_preprocess_data():
    print("Loading and preprocessing data...")
    df = pd.read_csv(DATA_PATH)
    
    if SAMPLE_FRACTION < 1.0:
        print(f"Quick test: Using {SAMPLE_FRACTION:.1%} of data")
        df = df.sample(frac=SAMPLE_FRACTION, random_state=RANDOM_STATE)
    
    df['binary_label'] = (df['Label'] != 'Benign').astype(int)
    print(f"Initial dataset shape: {df.shape}")
    
    missing_percentage = df.isnull().sum() / len(df)
    columns_to_drop = missing_percentage[missing_percentage > 0.8].index
    df = df.drop(columns=columns_to_drop)
    print(f"Dropped {len(columns_to_drop)} columns with >80% missing data")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        from sklearn.impute import SimpleImputer
        numeric_imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
    
    df = df.dropna()
    print(f"Final dataset shape: {df.shape}")
    return df

def create_test_splits_with_scaling():
    df = load_and_preprocess_data()
    
    feature_columns = [col for col in df.columns if col not in 
                      ['Label', 'Attack Type', 'Attack Tool', 'binary_label']]
    
    print(f"Using {len(feature_columns)} feature columns")
    
    main_df, global_test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, 
        stratify=df['Attack Type']
    )
    
    scaler = StandardScaler()
    X_main = main_df[feature_columns]
    scaler.fit(X_main)
    
    X_global_test = scaler.transform(global_test_df[feature_columns])
    y_global_test = global_test_df['binary_label'].values
    
    print(f"Data scaled - Global test set: {X_global_test.shape}")
    
    tasks = []
    task_test_sets = []
    
    for i, attack_list in enumerate(TASK_ATTACK_ORDER):
        print(f"Creating task {i} for attacks: {attack_list}")
        
        task_data = main_df[main_df['Attack Type'].isin(['Benign'] + attack_list)]
        X_task = task_data[feature_columns]
        y_task = task_data['binary_label']
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_task, y_task, test_size=VAL_SIZE, random_state=RANDOM_STATE,
            stratify=task_data['Attack Type']
        )
        
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        _, task_test = train_test_split(
            task_data, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=task_data['Attack Type']
        )
        
        X_task_test = scaler.transform(task_test[feature_columns])
        y_task_test = task_test['binary_label'].values
        
        tasks.append((X_train_scaled, y_train.values, X_val_scaled, y_val.values))
        task_test_sets.append((X_task_test, y_task_test))
        
        print(f"  Task {i}: {len(X_train_scaled)} training, {len(X_val_scaled)} validation, {len(X_task_test)} test samples")
        train_dist = f"[{np.sum(y_train == 0)} {np.sum(y_train == 1)}]"
        val_dist = f"[{np.sum(y_val == 0)} {np.sum(y_val == 1)}]"
        print(f"    Class distribution - Train: {train_dist}, Val: {val_dist}")
    
    print(f"Global test set: {len(X_global_test)} samples")
    return tasks, task_test_sets, (X_global_test, y_global_test), feature_columns, scaler

# ====== TESTING FUNCTIONS ======
def test_baseline_models(tasks, task_test_sets, global_test_set):
    print("\n" + "="*60)
    print("TESTING BASELINE MODELS")
    print("="*60)
    
    baseline_models = get_baseline_models()
    all_results = {}
    
    for model_name, model in baseline_models.items():
        print(f"\n--- Testing {model_name} ---")
        results = {}
        X_global_test, y_global_test = global_test_set
        current_model = None
        
        for task_id, ((X_train, y_train, X_val, y_val), (X_task_test, y_task_test)) in enumerate(zip(tasks, task_test_sets)):
            current_attack = get_attack_name(task_id)
            print(f"  Learning {current_attack}...")
            
            unique_classes = np.unique(y_train)
            if len(unique_classes) < 2:
                print(f"    Only one class present. Using dummy classifier.")
                current_model = DummyClassifier(strategy="most_frequent")
                current_model.fit(X_train, y_train)
            else:
                if hasattr(model, 'partial_fit') and current_model is None:
                    current_model = copy.deepcopy(model)
                    current_model.partial_fit(X_train, y_train, classes=np.array([0, 1]))
                elif hasattr(model, 'partial_fit') and current_model is not None:
                    current_model.partial_fit(X_train, y_train)
                else:
                    current_model = copy.deepcopy(model)
                    current_model.fit(X_train, y_train)
            
            task_results = {}
            for eval_task in range(task_id + 1):
                eval_attack = get_attack_name(eval_task)
                X_eval, y_eval = task_test_sets[eval_task]
                metrics = evaluate_model(current_model, X_eval, y_eval)
                task_results[eval_attack] = metrics
                print(f"    {eval_attack} Accuracy: {metrics['accuracy']:.3f}")
            
            global_metrics = evaluate_model(current_model, X_global_test, y_global_test)
            task_results['global'] = global_metrics
            print(f"    Global Accuracy: {global_metrics['accuracy']:.3f}")
            
            for eval_task in range(task_id + 1):
                eval_attack = get_attack_name(eval_task)
                task_results[eval_task] = task_results[eval_attack]
            
            results[task_id] = task_results
        
        all_results[model_name] = results
        
        try:
            model_filename = f'{model_name.replace(" ", "_")}_model.joblib'
            joblib.dump(current_model, model_filename)
            print(f"  Model saved as {model_filename}")
        except Exception as e:
            print(f"  Could not save {model_name} model: {e}")
    
    return all_results

def create_torch_dataloaders(tasks, batch_size=32):
    torch_tasks = []
    
    for i, (X_train, y_train, X_val, y_val) in enumerate(tasks):
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        torch_tasks.append((train_loader, val_loader))
        print(f"  Task {i}: {len(train_dataset)} training, {len(val_dataset)} validation samples")
    
    return torch_tasks

def test_mlp_models(tasks, task_test_sets, global_test_set, feature_columns):
    print("\n" + "="*60)
    print("TESTING MLP MODELS")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    torch_tasks = create_torch_dataloaders(tasks, batch_size=BATCH_SIZE)
    input_dim = tasks[0][0].shape[1]
    print(f"Input dimension: {input_dim}")
    
    models = {
        "MLP (Naive)": MLP(input_size=input_dim, hidden_size=128),
        "MLP + LwF": MLPWithLwF(input_size=input_dim, hidden_size=128),
        "MLP + EWC": MLPWithEWC(input_size=input_dim, hidden_size=128)
    }
    
    all_results = {}
    X_global_test, y_global_test = global_test_set
    
    for model_name, model in models.items():
        print(f"\n--- Testing {model_name} ---")
        model.to(device)
        results = {}
        previous_model = None
        
        for task_id, ((train_loader, val_loader), (X_task_test, y_task_test)) in enumerate(zip(torch_tasks, task_test_sets)):
            current_attack = get_attack_name(task_id)
            print(f"  Learning {current_attack}...")
            
            if model_name == "MLP + LwF" and previous_model is not None:
                model.set_previous_model(previous_model)
            
            try:
                trained_model = train_torch_model_adaptive(
                    model, train_loader, val_loader, device=device,
                    epochs=EPOCHS, lr=LEARNING_RATE,
                    model_type='lwf' if 'LwF' in model_name else 'ewc' if 'EWC' in model_name else 'basic',
                    task_id=task_id, max_grad_norm=1.0
                )
                
                if model_name == "MLP + EWC":
                    trained_model.compute_fisher(train_loader, device=device, task_id=task_id)
                
            except Exception as e:
                print(f"  Error training {model_name} on task {task_id}: {e}")
                continue
            
            task_results = {}
            for eval_task in range(task_id + 1):
                try:
                    X_eval, y_eval = task_test_sets[eval_task]
                    X_eval_tensor = torch.FloatTensor(X_eval)
                    y_eval_tensor = torch.LongTensor(y_eval)
                    
                    metrics = evaluate_torch_model(trained_model, X_eval_tensor, y_eval_tensor, device=device)
                    if 'roc_auc' not in metrics:
                        metrics['roc_auc'] = metrics.get('accuracy', 0)
                    
                    task_results[eval_task] = metrics
                    print(f"    Task {eval_task} Accuracy: {metrics['accuracy']:.3f}")
                    
                except Exception as e:
                    print(f"    Error evaluating task {eval_task}: {e}")
                    task_results[eval_task] = {'accuracy': 0.0, 'f1_score': 0.0, 'roc_auc': 0.5}
            
            try:
                X_global_tensor = torch.FloatTensor(X_global_test)
                y_global_tensor = torch.LongTensor(y_global_test)
                global_metrics = evaluate_torch_model(trained_model, X_global_tensor, y_global_tensor, device=device)
                if 'roc_auc' not in global_metrics:
                    global_metrics['roc_auc'] = global_metrics.get('accuracy', 0)
                task_results['global'] = global_metrics
                print(f"    Global Accuracy: {global_metrics['accuracy']:.3f}")
            except Exception as e:
                print(f"    Error evaluating global: {e}")
                task_results['global'] = {'accuracy': 0.0, 'f1_score': 0.0, 'roc_auc': 0.5}
            
            results[task_id] = task_results
            previous_model = copy.deepcopy(trained_model)
        
        all_results[model_name] = results
        
        try:
            model_filename = f'{model_name.replace(" ", "_").replace("(", "").replace(")", "")}_model.pth'
            torch.save({
                'model_state_dict': trained_model.state_dict(),
                'model_type': model_name,
                'input_dim': input_dim
            }, model_filename)
            print(f"  Model saved as {model_filename}")
        except Exception as e:
            print(f"  Could not save {model_name} model: {e}")
    
    return all_results

# ====== MAIN EXECUTION ======
def main():
    print("UNIFIED TESTING SCRIPT - SELF-CONTAINED")
    print("="*60)
    
    tasks, task_test_sets, global_test_set, feature_columns, scaler = create_test_splits_with_scaling()
    
    all_results_baseline = test_baseline_models(tasks, task_test_sets, global_test_set)
    all_results_mlp = test_mlp_models(tasks, task_test_sets, global_test_set, feature_columns)
    
    all_results = {**all_results_baseline, **all_results_mlp}
    
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    try:
        with open('final_results.json', 'w') as f:
            json_results = {}
            for model_name, results in all_results.items():
                json_results[model_name] = {}
                for task_trained, task_results in results.items():
                    json_results[model_name][task_trained] = {}
                    for eval_task, metrics in task_results.items():
                        json_results[model_name][task_trained][eval_task] = {
                            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                            for k, v in metrics.items()
                        }
            json.dump(json_results, f, indent=2)
        print("Results saved to final_results.json")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    print("\n" + "="*60)
    print("TESTING COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()