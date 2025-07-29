#!/usr/bin/env python3
"""
Quick training progress checker
"""

import os
import psutil
import time
from pathlib import Path

def check_training_status():
    """Check current training status"""
    print(f"=== Training Status Check - {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    # Check if training process is running
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent']):
        try:
            if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'train_improved_xception.py' in cmdline:
                    python_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if python_processes:
        print(f"✓ Training process is running (PID: {python_processes[0].pid})")
        print(f"  CPU Usage: {python_processes[0].cpu_percent():.1f}%")
    else:
        print("✗ No training process detected")
    
    # Check for latest model files
    models_dir = Path("models")
    if models_dir.exists():
        print(f"\n📁 Models directory:")
        recent_models = []
        for model_file in models_dir.glob("*.pth"):
            mtime = model_file.stat().st_mtime
            recent_models.append((model_file, mtime))
        
        recent_models.sort(key=lambda x: x[1], reverse=True)
        for model_file, mtime in recent_models[:5]:  # Show last 5 models
            time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
            print(f"  {model_file.name} - {time_str}")
    
    # Check for latest log files
    log_files = list(Path(".").glob("*training*.log"))
    if log_files:
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        print(f"\n📄 Latest training log: {latest_log.name}")
        
        # Read last few lines
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                if lines:
                    print("  Last log entries:")
                    for line in lines[-3:]:
                        print(f"    {line.strip()}")
        except:
            pass
    
    print("\n" + "="*60)

if __name__ == "__main__":
    check_training_status()
