#!/usr/bin/env python3
"""
Simple training progress tracker
"""

import time
import os
from pathlib import Path

def track_training():
    """Track training progress"""
    log_file = "training_20250727_213957.log"
    
    if not os.path.exists(log_file):
        print("Training log not found!")
        return
    
    print("🚀 FaceForensics++ Training Tracker")
    print("="*50)
    
    last_size = 0
    epoch_count = 0
    
    try:
        while True:
            # Check if file has grown
            current_size = os.path.getsize(log_file)
            
            if current_size > last_size:
                # Read new content
                with open(log_file, 'r') as f:
                    f.seek(last_size)
                    new_lines = f.read()
                    
                    # Check for epoch completions
                    for line in new_lines.split('\n'):
                        if 'Epoch' in line and '/20' in line:
                            epoch_info = line.split(' - ')[-1]
                            print(f"📈 {epoch_info}")
                        elif 'Train Loss:' in line:
                            loss_info = line.split(' - ')[-1]
                            print(f"  📊 {loss_info}")
                        elif 'Val Loss:' in line:
                            val_info = line.split(' - ')[-1]
                            print(f"  📋 {val_info}")
                        elif 'Model saved to:' in line:
                            save_info = line.split(' - ')[-1]
                            print(f"  💾 {save_info}")
                        elif 'Training completed' in line:
                            print("🎉 Training completed successfully!")
                            return
                        elif 'ERROR' in line:
                            error_info = line.split(' - ')[-1]
                            print(f"  ❌ {error_info}")
                
                last_size = current_size
            
            time.sleep(5)  # Check every 5 seconds
            
    except KeyboardInterrupt:
        print("\n⏹️ Tracking stopped by user")
    except Exception as e:
        print(f"❌ Error tracking: {e}")

if __name__ == "__main__":
    track_training()
