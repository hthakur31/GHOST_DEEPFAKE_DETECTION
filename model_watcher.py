#!/usr/bin/env python3
"""
Auto-reload enhanced XceptionNet model for Django integration
Monitors for new trained models and automatically reloads the prediction service
"""

import os
import time
import threading
from pathlib import Path
import logging
from enhanced_xception_predictor import reload_xception_model, get_xception_predictor

logger = logging.getLogger(__name__)

class ModelWatcher:
    """Watch for new XceptionNet models and auto-reload the predictor"""
    
    def __init__(self, models_dir="models", check_interval=30):
        self.models_dir = Path(models_dir)
        self.check_interval = check_interval
        self.latest_model_time = 0
        self.watcher_thread = None
        self.stop_watching = threading.Event()
        
        # Get initial model timestamp
        self._update_latest_model_time()
    
    def _update_latest_model_time(self):
        """Update the timestamp of the latest model"""
        if not self.models_dir.exists():
            return
        
        latest_time = 0
        for pattern in ["improved_xception*.pth", "robust_xception*.pth", "xception_best_*.pth"]:
            for model_file in self.models_dir.glob(pattern):
                file_time = model_file.stat().st_mtime
                if file_time > latest_time:
                    latest_time = file_time
        
        self.latest_model_time = latest_time
    
    def _check_for_new_models(self):
        """Check if there are new models and reload if necessary"""
        try:
            current_latest = 0
            latest_model_file = None
            
            if not self.models_dir.exists():
                return
            
            # Check for new improved models first (highest priority)
            for pattern in ["improved_xception*.pth", "robust_xception*.pth", "xception_best_*.pth"]:
                for model_file in self.models_dir.glob(pattern):
                    file_time = model_file.stat().st_mtime
                    if file_time > current_latest:
                        current_latest = file_time
                        latest_model_file = model_file
            
            # If we found a newer model, reload the predictor
            if current_latest > self.latest_model_time:
                logger.info(f"New XceptionNet model detected: {latest_model_file}")
                logger.info("Reloading enhanced XceptionNet predictor...")
                
                success = reload_xception_model()
                if success:
                    logger.info("✓ Enhanced XceptionNet model reloaded successfully!")
                    predictor = get_xception_predictor()
                    if predictor.model is not None:
                        logger.info(f"  Model type: {predictor.model_type}")
                        logger.info(f"  Model file: {latest_model_file.name}")
                        logger.info("  Django application now using the latest trained model")
                else:
                    logger.error("✗ Failed to reload enhanced XceptionNet model")
                
                self.latest_model_time = current_latest
                
        except Exception as e:
            logger.error(f"Error checking for new models: {e}")
    
    def start_watching(self):
        """Start watching for new models in a background thread"""
        if self.watcher_thread and self.watcher_thread.is_alive():
            return
        
        self.stop_watching.clear()
        self.watcher_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.watcher_thread.start()
        logger.info(f"Started XceptionNet model watcher (checking every {self.check_interval}s)")
    
    def stop_watching_models(self):
        """Stop watching for new models"""
        self.stop_watching.set()
        if self.watcher_thread:
            self.watcher_thread.join(timeout=5)
        logger.info("Stopped XceptionNet model watcher")
    
    def _watch_loop(self):
        """Main watching loop"""
        while not self.stop_watching.is_set():
            self._check_for_new_models()
            self.stop_watching.wait(self.check_interval)

# Global model watcher instance
model_watcher = ModelWatcher()

def start_model_watcher():
    """Start the global model watcher (called from Django startup)"""
    model_watcher.start_watching()
    return model_watcher

def stop_model_watcher():
    """Stop the global model watcher"""
    model_watcher.stop_watching_models()

def force_model_reload():
    """Force a manual model reload (useful for API endpoints)"""
    logger.info("Force reloading enhanced XceptionNet model...")
    success = reload_xception_model()
    if success:
        predictor = get_xception_predictor()
        return {
            "success": True,
            "model_loaded": predictor.model is not None,
            "model_type": predictor.model_type if predictor.model else None,
            "message": "Enhanced XceptionNet model reloaded successfully"
        }
    else:
        return {
            "success": False,
            "model_loaded": False,
            "model_type": None,
            "message": "Failed to reload enhanced XceptionNet model"
        }

if __name__ == "__main__":
    # Test the model watcher
    print("Testing Enhanced XceptionNet Model Watcher...")
    watcher = ModelWatcher(check_interval=5)  # Check every 5 seconds for testing
    watcher.start_watching()
    
    try:
        print("Watching for new models... Press Ctrl+C to stop")
        time.sleep(60)  # Watch for 1 minute
    except KeyboardInterrupt:
        print("\nStopping model watcher...")
    finally:
        watcher.stop_watching_models()
        print("Model watcher stopped.")
