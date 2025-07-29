#!/usr/bin/env python3
"""
Training Progress Monitor
Monitors the training progress and shows real-time updates
"""

import time
import json
from pathlib import Path
import os

def monitor_training():
    """Monitor training progress by checking log files and model outputs"""
    
    print("🔍 FaceForensics++ Training Progress Monitor")
    print("=" * 50)
    
    # Paths to monitor
    log_files = list(Path(".").glob("training_*.log"))
    models_dir = Path("models")
    dataset_analysis = Path("dataset/dataset_analysis.json")
    
    # Show dataset info
    if dataset_analysis.exists():
        with open(dataset_analysis) as f:
            data = json.load(f)
        print(f"📊 Dataset: {data['total_real']} real + {data['total_fake']} fake videos")
        print(f"🎭 Methods: {', '.join(data['manipulated_videos'].keys())}")
    
    # Show latest log file
    if log_files:
        latest_log = max(log_files, key=os.path.getctime)
        print(f"📝 Latest log: {latest_log}")
        
        try:
            with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            # Show last few lines
            print("\n📈 Recent Training Output:")
            print("-" * 30)
            for line in lines[-10:]:
                if line.strip():
                    print(line.strip())
        except Exception as e:
            print(f"Could not read log: {e}")
    
    # Check for completed models
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pth"))
        stats_files = list(models_dir.glob("training_stats_*.json"))
        
        if model_files:
            print(f"\n🎯 Trained Models Found: {len(model_files)}")
            for model in model_files:
                print(f"  - {model.name}")
        
        if stats_files:
            print(f"\n📊 Training Stats: {len(stats_files)}")
            latest_stats = max(stats_files, key=os.path.getctime)
            try:
                with open(latest_stats) as f:
                    stats = json.load(f)
                print(f"  - Latest: {latest_stats.name}")
                if 'training_stats' in stats:
                    ts = stats['training_stats']
                    print(f"  - Final Accuracy: {ts.get('best_val_accuracy', 'N/A')}")
                    print(f"  - Training Time: {ts.get('total_training_time', 'N/A')} seconds")
            except Exception as e:
                print(f"  - Could not read stats: {e}")
    
    print(f"\n⏰ Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    monitor_training()
