# models/mlp_models.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import numpy as np

class MLP(nn.Module):
    """Basic Multilayer Perceptron"""
    def __init__(self, input_size, hidden_size=128):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

class MLPWithLwF(MLP):
    """MLP with Learning without Forgetting"""
    def __init__(self, input_size, hidden_size=128):
        super(MLPWithLwF, self).__init__(input_size, hidden_size)
        self.previous_model = None
        self.previous_optimizer = None
    
    def set_previous_model(self, previous_model):
        """Store the previous model for knowledge distillation"""
        self.previous_model = copy.deepcopy(previous_model)
        self.previous_model.eval()  # Set to evaluation mode
    
    def compute_loss(self, outputs, targets, previous_outputs=None, alpha=0.5):
        """Compute combined loss with knowledge distillation"""
        current_loss = nn.BCELoss()(outputs, targets.unsqueeze(1))
        
        if previous_outputs is not None and self.previous_model is not None:
            distillation_loss = nn.BCELoss()(outputs, previous_outputs.detach())
            return alpha * current_loss + (1 - alpha) * distillation_loss
        
        return current_loss

class MLPWithEWC(MLP):
    """MLP with Elastic Weight Consolidation"""
    def __init__(self, input_size, hidden_size=128):
        super(MLPWithEWC, self).__init__(input_size, hidden_size)
        self.importance = {}
        self.fisher = {}
        self.previous_params = {}
    
    def compute_fisher(self, data_loader, device='cpu'):
        """Compute Fisher Information matrix"""
        self.eval()
        fisher_dict = {}
        
        # Initialize fisher values
        for name, param in self.named_parameters():
            fisher_dict[name] = torch.zeros_like(param.data)
        
        # Compute gradients squared
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            self.zero_grad()
            outputs = self(batch_X)
            loss = nn.BCELoss()(outputs, batch_y.unsqueeze(1))
            loss.backward()
            
            for name, param in self.named_parameters():
                if param.grad is not None:
                    fisher_dict[name] += param.grad.data ** 2 / len(data_loader)
        
        self.fisher = fisher_dict
        self.previous_params = {name: param.data.clone() for name, param in self.named_parameters()}
    
    def compute_ewc_loss(self, current_loss, lamda=1000):
        """Add EWC penalty to the loss"""
        ewc_loss = current_loss
        for name, param in self.named_parameters():
            if name in self.fisher and name in self.previous_params:
                ewc_loss += (lamda / 2) * torch.sum(
                    self.fisher[name] * (param - self.previous_params[name]) ** 2
                )
        return ewc_loss

def train_torch_model(model, train_loader, val_loader, device='cpu', epochs=10, lr=0.001, model_type='basic'):
    """Train a PyTorch model"""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    best_acc = 0
    patience = 3
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            if model_type == 'lwf' and hasattr(model, 'previous_model') and model.previous_model is not None:
                with torch.no_grad():
                    previous_outputs = model.previous_model(batch_X)
                loss = model.compute_loss(outputs, batch_y, previous_outputs)
            else:
                loss = criterion(outputs, batch_y.unsqueeze(1))
            
            if model_type == 'ewc':
                loss = model.compute_ewc_loss(loss)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_acc = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                preds = (outputs > 0.5).float()
                val_acc += (preds.squeeze() == batch_y).float().mean().item()
        
        val_acc /= len(val_loader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.3f}')
        
        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            best_model = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    model.load_state_dict(best_model)
    return model