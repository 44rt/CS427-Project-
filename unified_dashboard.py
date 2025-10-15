# import flask
# import pandas as pd
# import json
# import numpy as np
# import os
# import sys
# import random
# import threading
# from datetime import datetime, timedelta
# from flask import Flask, jsonify, render_template

# # Add the current directory to Python path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# app = Flask(__name__)

# class LiveDataLoader:
#     """Live data loader that can reload data on demand"""
    
#     def __init__(self):
#         self.data_loaded = False
#         self.last_update = None
#         self.load_all_data()
    
#     def load_all_data(self):
#         """Load all model results dynamically"""
#         try:
#             print(" Loading live model results...")
            
#             # MLP Results
#             if os.path.exists('mlp_results.json'):
#                 with open('mlp_results.json', 'r') as f:
#                     self.mlp_results = json.load(f)
#                 print(f" MLP models: {list(self.mlp_results.keys())}")
#             else:
#                 self.mlp_results = {}
#                 print("  mlp_results.json not found")
            
#             # Baseline Results
#             if os.path.exists('baseline_results.json'):
#                 with open('baseline_results.json', 'r') as f:
#                     self.baseline_results = json.load(f)
#                 print(f" Baseline models: {list(self.baseline_results.keys())}")
#             else:
#                 self.baseline_results = {}
#                 print("  baseline_results.json not found")
            
#             # Load summaries
#             self.mlp_summary = pd.read_csv('final_mlp_summary.csv') if os.path.exists('final_mlp_summary.csv') else pd.DataFrame()
#             self.baseline_summary = pd.read_csv('baseline_summary.csv') if os.path.exists('baseline_summary.csv') else pd.DataFrame()
            
#             # Create unified data
#             self.unified_data = self._create_unified_from_results()
#             self.last_update = datetime.now()
#             self.data_loaded = True
            
#             print(f" Live data loaded! Total models: {len(self.unified_data['models'])}")
            
#         except Exception as e:
#             print(f" Error loading live data: {e}")
#             self.data_loaded = False
    
#     def _create_unified_from_results(self):
#         """Create unified data from actual training results"""
#         unified_data = {
#             'models': [],
#             'accuracy': [],
#             'f1_scores': [],
#             'auc_scores': [],
#             'forgetting_rates': [],
#             'model_types': [],
#             'last_update': datetime.now().isoformat()
#         }
        
#         # Process MLP models
#         for model_name, results in self.mlp_results.items():
#             if results:  # Check if results exist
#                 accuracy, f1, auc, forgetting = self._extract_model_metrics(results)
#                 unified_data['models'].append(model_name)
#                 unified_data['accuracy'].append(accuracy)
#                 unified_data['f1_scores'].append(f1)
#                 unified_data['auc_scores'].append(auc)
#                 unified_data['forgetting_rates'].append(forgetting)
#                 unified_data['model_types'].append('mlp')
        
#         # Process baseline models
#         for model_name, results in self.baseline_results.items():
#             if results:  # Check if results exist
#                 accuracy, f1, auc, forgetting = self._extract_model_metrics(results)
#                 unified_data['models'].append(model_name)
#                 unified_data['accuracy'].append(accuracy)
#                 unified_data['f1_scores'].append(f1)
#                 unified_data['auc_scores'].append(auc)
#                 unified_data['forgetting_rates'].append(forgetting)
#                 unified_data['model_types'].append('baseline')
        
#         return unified_data
    
#     def _extract_model_metrics(self, results):
#         """Extract metrics from model results"""
#         try:
#             if not results:
#                 return 0, 0, 0, 0
            
#             # Get final task
#             task_keys = [k for k in results.keys() if isinstance(k, int)]
#             if not task_keys:
#                 return 0, 0, 0, 0
            
#             final_task = max(task_keys)
#             final_results = results[final_task]
            
#             # Calculate average metrics across all tasks
#             accuracies, f1_scores, auc_scores = [], [], []
            
#             for eval_task in [k for k in final_results.keys() if k != 'global']:
#                 metrics = final_results[eval_task]
#                 accuracies.append(metrics.get('accuracy', 0))
#                 f1_scores.append(metrics.get('f1_score', 0))
#                 auc_scores.append(metrics.get('roc_auc', metrics.get('accuracy', 0)))
            
#             avg_accuracy = np.mean(accuracies) if accuracies else 0
#             avg_f1 = np.mean(f1_scores) if f1_scores else 0
#             avg_auc = np.mean(auc_scores) if auc_scores else 0
            
#             # Calculate forgetting
#             forgetting = self._calculate_forgetting(results)
            
#             return avg_accuracy, avg_f1, avg_auc, forgetting
            
#         except Exception as e:
#             print(f"Error extracting metrics: {e}")
#             return 0, 0, 0, 0
    
#     def _calculate_forgetting(self, results):
#         """Calculate actual forgetting from training history"""
#         try:
#             task_keys = [k for k in results.keys() if isinstance(k, int)]
#             if len(task_keys) < 2:
#                 return 0.0
            
#             final_task = max(task_keys)
#             final_results = results[final_task]
            
#             forgetting_scores = []
#             for task_id in task_keys[:-1]:  # All except last task
#                 # Get initial accuracy
#                 if task_id in results and task_id in results[task_id]:
#                     initial_acc = results[task_id][task_id].get('accuracy', 0)
#                 else:
#                     continue
                
#                 # Get final accuracy
#                 final_acc = final_results.get(task_id, {}).get('accuracy', 0)
                
#                 # Calculate forgetting
#                 forgetting = max(0, initial_acc - final_acc)
#                 forgetting_scores.append(forgetting)
            
#             return np.mean(forgetting_scores) if forgetting_scores else 0.0
            
#         except Exception as e:
#             return 0.0
    
#     def reload_data(self):
#         """Reload data from disk"""
#         self.load_all_data()
#         return self.data_loaded

# # Initialize data loader
# data_loader = LiveDataLoader()

# # API Routes
# @app.route('/')
# def index():
#     return render_template('unified_dashboard.html')

# @app.route('/api/unified_comparison')
# def get_unified_comparison():
#     """Get unified comparison data"""
#     if not data_loader.data_loaded:
#         return jsonify({'error': 'Data not loaded', 'models': []}), 500
#     return jsonify(data_loader.unified_data)

# @app.route('/api/combined_summary')
# def get_combined_summary():
#     """Get combined model summary"""
#     try:
#         combined = []
        
#         # Add MLP models
#         if not data_loader.mlp_summary.empty:
#             for _, row in data_loader.mlp_summary.iterrows():
#                 combined.append(dict(row))
        
#         # Add baseline models
#         if not data_loader.baseline_summary.empty:
#             for _, row in data_loader.baseline_summary.iterrows():
#                 combined.append(dict(row))
        
#         return jsonify(combined)
#     except Exception as e:
#         return jsonify([])

# @app.route('/api/stats')
# def get_stats():
#     """Get system statistics"""
#     try:
#         if data_loader.data_loaded:
#             total_models = len(data_loader.unified_data['models'])
#             mlp_count = sum(1 for t in data_loader.unified_data['model_types'] if t == 'mlp')
#             baseline_count = total_models - mlp_count
            
#             # Find best model
#             if data_loader.unified_data['accuracy']:
#                 best_idx = np.argmax(data_loader.unified_data['accuracy'])
#                 best_model = data_loader.unified_data['models'][best_idx]
#                 best_accuracy = data_loader.unified_data['accuracy'][best_idx]
#             else:
#                 best_model = "No models"
#                 best_accuracy = 0
#         else:
#             total_models = mlp_count = baseline_count = 0
#             best_model = "Data loading..."
#             best_accuracy = 0
        
#         return jsonify({
#             'best_model': best_model,
#             'best_accuracy': best_accuracy,
#             'total_models': total_models,
#             'mlp_models': mlp_count,
#             'baseline_models': baseline_count,
#             'features_analyzed': 93,
#             'attack_types': ['UDP Flood', 'HTTP Flood', 'Slow-rate DoS'],
#             'data_source': 'BTW_2.csv',
#             'status': 'Operational',
#             'update_time': datetime.now().strftime('%H:%M:%S'),
#             'last_data_update': data_loader.last_update.isoformat() if data_loader.last_update else None
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)})

# @app.route('/api/reload')
# def reload_data():
#     """Force reload data from disk"""
#     if data_loader.reload_data():
#         return jsonify({'status': 'success', 'message': 'Data reloaded successfully'})
#     else:
#         return jsonify({'status': 'error', 'message': 'Failed to reload data'})

# @app.route('/api/detection')
# def get_detection():
#     """Real-time detection simulation"""
#     attacks = [
#         {'type': 'UDP Flood', 'emoji': '', 'is_attack': True},
#         {'type': 'HTTP Flood', 'emoji': '', 'is_attack': True},
#         {'type': 'Slow-rate DoS', 'emoji': '', 'is_attack': True},
#         {'type': 'Normal Traffic', 'emoji': '', 'is_attack': False}
#     ]
    
#     current_detections = []
#     for _ in range(random.randint(2, 5)):
#         attack = random.choice(attacks)
#         current_detections.append({
#             'traffic_type': attack['type'],
#             'emoji': attack['emoji'],
#             'is_attack': attack['is_attack'],
#             'confidence': f"{random.uniform(0.85, 0.99):.2f}",
#             'timestamp': datetime.now().strftime('%H:%M:%S')
#         })
    
#     return jsonify({'current': current_detections})

# @app.route('/api/network_traffic')
# def get_network_traffic():
#     """Network traffic patterns"""
#     hours = 24
#     traffic_data = {
#         'UDP Flood': {'data': []},
#         'HTTP Flood': {'data': []},
#         'Slow-rate DoS': {'data': []},
#         'Normal': {'data': []}
#     }
    
#     base_time = datetime.now() - timedelta(hours=hours)
    
#     for i in range(hours):
#         timestamp = (base_time + timedelta(hours=i)).strftime('%H:%M')
        
#         traffic_data['UDP Flood']['data'].append({
#             'time': timestamp,
#             'count': random.randint(0, 50) + random.randint(0, 30) * (1 if i % 6 < 2 else 0)
#         })
        
#         traffic_data['HTTP Flood']['data'].append({
#             'time': timestamp,
#             'count': random.randint(0, 40) + random.randint(0, 40) * (1 if i % 8 < 3 else 0)
#         })
        
#         traffic_data['Slow-rate DoS']['data'].append({
#             'time': timestamp,
#             'count': random.randint(0, 30) + random.randint(0, 20) * (1 if i % 12 < 4 else 0)
#         })
        
#         traffic_data['Normal']['data'].append({
#             'time': timestamp,
#             'count': random.randint(100, 200)
#         })
    
#     return jsonify(traffic_data)

# @app.route('/api/model_performance')
# def get_model_performance():
#     """Model performance data for charts"""
#     try:
#         if not data_loader.data_loaded:
#             return jsonify({'error': 'Data not loaded'})
        
#         # Model comparison
#         model_comparison = {
#             'models': data_loader.unified_data['models'],
#             'accuracy': data_loader.unified_data['accuracy'],
#             'f1_scores': data_loader.unified_data['f1_scores']
#         }
        
#         # Task performance (simplified for now)
#         task_performance = {
#             'tasks': ['Task 1', 'Task 2', 'Task 3'],
#             'mlp_naive': [0.8, 0.75, 0.7],
#             'mlp_lwf': [0.82, 0.8, 0.78],
#             'mlp_ewc': [0.81, 0.79, 0.77]
#         }
        
#         # Forgetting
#         forgetting_data = {
#             'models': data_loader.unified_data['models'],
#             'forgetting': data_loader.unified_data['forgetting_rates']
#         }
        
#         return jsonify({
#             'model_comparison': model_comparison,
#             'task_performance': task_performance,
#             'forgetting': forgetting_data
#         })
        
#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     print(" Unified 5G IDS Dashboard Starting...")
#     print("=" * 50)
#     print(" Data Status:", " Loaded" if data_loader.data_loaded else " Not Loaded")
#     if data_loader.data_loaded:
#         print(f"   Total Models: {len(data_loader.unified_data['models'])}")
#         print(f"   Last Update: {data_loader.last_update}")
#     print(" Dashboard: http://localhost:5000")
#     print(" Auto-reload: Visit /api/reload to refresh data")
#     print("=" * 50)
    
#     app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

# unified_dashboard.py
from flask import Flask, render_template, jsonify, request
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import os
from dashboard_data_processor import UnifiedDataProcessor

app = Flask(__name__)

class UnifiedDashboardSystem:
    def __init__(self):
        self.data_processor = UnifiedDataProcessor()
        self.detection_history = []
        self.performance_history = []
        
    def simulate_detection(self):
        """Simulate real-time detection events"""
        traffic_types = [
            ("Normal Browsing", 0.1, "🌐", "Normal", False),
            ("Suspicious UDP", 0.7, "🌊", "UDP Flood", True),
            ("HTTP Flood", 0.9, "🔥", "HTTP Flood", True),
            ("Slow-rate DoS", 0.6, "🐌", "Slow-rate DoS", True),
            ("Mixed Attacks", 0.8, "⚡", "Mixed", True)
        ]
        
        results = []
        for traffic_name, attack_prob, emoji, attack_type, is_attack in traffic_types:
            # Add some randomness to confidence
            confidence = attack_prob + np.random.uniform(-0.2, 0.2)
            confidence = max(0.1, min(0.99, confidence))
            
            results.append({
                'traffic_type': traffic_name,
                'attack_type': attack_type,
                'emoji': emoji,
                'is_attack': is_attack,
                'confidence': round(confidence, 3),
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
        
        return results

# Initialize dashboard
dashboard = UnifiedDashboardSystem()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('unified_dashboard.html')

@app.route('/api/unified_comparison')
def get_unified_comparison():
    """Get unified comparison data for all models"""
    try:
        comparison_data = dashboard.data_processor.get_all_models_comparison()
        return jsonify(comparison_data)
    except Exception as e:
        print(f"Error in unified_comparison: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_performance')
def get_model_performance():
    """Get MLP model performance data"""
    try:
        task_performance = dashboard.data_processor.get_task_performance()
        return jsonify({'task_performance': task_performance})
    except Exception as e:
        print(f"Error in model_performance: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/network_traffic')
def get_network_traffic():
    """Get network traffic patterns"""
    try:
        traffic_patterns = dashboard.data_processor.get_network_traffic_patterns()
        return jsonify(traffic_patterns)
    except Exception as e:
        print(f"Error in network_traffic: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/combined_summary')
def get_combined_summary():
    """Get combined summary for all models"""
    try:
        summary_data = dashboard.data_processor.get_combined_summary()
        return jsonify(summary_data)
    except Exception as e:
        print(f"Error in combined_summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/detection')
def get_detection():
    """Get real-time detection data"""
    try:
        detection_data = dashboard.simulate_detection()
        dashboard.detection_history.extend(detection_data)
        
        # Keep only last 20 detections
        if len(dashboard.detection_history) > 20:
            dashboard.detection_history = dashboard.detection_history[-20:]
        
        return jsonify({
            'current': detection_data,
            'history': dashboard.detection_history[-10:]
        })
    except Exception as e:
        print(f"Error in detection: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Get system statistics"""
    try:
        stats = dashboard.data_processor.get_system_stats()
        return jsonify(stats)
    except Exception as e:
        print(f"Error in stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reload')
def reload_data():
    """Reload data from files"""
    try:
        dashboard.data_processor.load_data()
        return jsonify({'status': 'success', 'message': 'Data reloaded successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Unified 5G IDS Dashboard...")
    print("📊 Loading data from unified testing results...")
    print("🌐 Open: http://localhost:5000")
    print("🔄 Dashboard will show graphs from your actual unified testing results!")
    print("⏹️  Press Ctrl+C to stop")
    
    app.run(debug=True, host='0.0.0.0', port=5000)