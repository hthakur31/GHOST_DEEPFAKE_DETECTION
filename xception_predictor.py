#!/usr/bin/env python3
"""
XceptionNet Prediction Service for Frontend Integration
Provides video deepfake detection using trained XceptionNet model
"""

import os
import sys
import torch
import torch.nn as nn
import cv2
import numpy as np
import face_recognition
from pathlib import Path
import logging
import json
from datetime import datetime
import tempfile
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XceptionNet(nn.Module):
    """XceptionNet for deepfake detection"""
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(XceptionNet, self).__init__()
        
        # Simplified Xception-like architecture (matches robust training)
        self.features = nn.Sequential(
            # Entry flow
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Depthwise separable convolutions
            nn.Conv2d(64, 64, 3, stride=2, padding=1, groups=64),
            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, 3, stride=1, padding=1, groups=128),
            nn.Conv2d(128, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, 3, stride=2, padding=1, groups=256),
            nn.Conv2d(256, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class XceptionNetPredictor:
    """XceptionNet prediction service for frontend integration"""
    
    def __init__(self, model_path=None, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = self._get_transform()
        
        # Auto-load the latest model if no path specified
        if model_path is None:
            model_path = self._find_latest_model()
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            logger.warning("No XceptionNet model found. Please train a model first.")
    
    def _find_latest_model(self):
        """Find the latest trained XceptionNet model"""
        models_dir = Path("models")
        if not models_dir.exists():
            return None
        
        # Look for XceptionNet models (prioritize robust models)
        model_patterns = ["robust_xception*.pth", "xception_best_*.pth", "xception_final_*.pth", "xception_epoch_*.pth"]
        
        latest_model = None
        latest_time = 0
        
        for pattern in model_patterns:
            for model_file in models_dir.glob(pattern):
                file_time = model_file.stat().st_mtime
                if file_time > latest_time:
                    latest_time = file_time
                    latest_model = model_file
        
        if latest_model:
            logger.info(f"Auto-detected XceptionNet model: {latest_model}")
            return str(latest_model)
        
        return None
    
    def _get_transform(self):
        """Get image preprocessing transform"""
        from torchvision import transforms
        
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path):
        """Load trained XceptionNet model"""
        try:
            self.model = XceptionNet(num_classes=2).to(self.device)
            
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                self.model.load_state_dict(checkpoint)
                logger.info("Loaded model state dict")
            
            self.model.eval()
            logger.info(f"XceptionNet model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load XceptionNet model: {e}")
            self.model = None
            return False
    
    def extract_face_from_frame(self, frame):
        """Extract face from video frame"""
        try:
            if frame.shape[0] < 50 or frame.shape[1] < 50:
                return None
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize for faster face detection
            height, width = frame_rgb.shape[:2]
            if height > 480 or width > 640:
                scale = min(480/height, 640/width)
                new_height, new_width = int(height * scale), int(width * scale)
                frame_small = cv2.resize(frame_rgb, (new_width, new_height))
                face_locations = face_recognition.face_locations(frame_small, model='hog')
                # Scale back face locations
                face_locations = [(int(top/scale), int(right/scale), int(bottom/scale), int(left/scale)) 
                                for top, right, bottom, left in face_locations]
            else:
                face_locations = face_recognition.face_locations(frame_rgb, model='hog')
            
            if not face_locations:
                return None
            
            # Get the largest face
            largest_face = max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
            top, right, bottom, left = largest_face
            
            # Validate face region
            if bottom <= top or right <= left:
                return None
            
            # Add padding
            h, w = frame_rgb.shape[:2]
            pad = min(20, (bottom - top) // 4)
            top = max(0, top - pad)
            bottom = min(h, bottom + pad)
            left = max(0, left - pad)
            right = min(w, right + pad)
            
            # Extract and resize face
            face_image = frame_rgb[top:bottom, left:right]
            
            if face_image.shape[0] < 32 or face_image.shape[1] < 32:
                return None
            
            face_image = cv2.resize(face_image, (224, 224))
            return face_image
            
        except Exception as e:
            logger.debug(f"Face extraction error: {e}")
            return None
    
    def predict_frame(self, face_image):
        """Predict if a face image is deepfake"""
        if self.model is None:
            return None, 0.0
        
        try:
            # Preprocess
            face_tensor = self.transform(face_image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence = probabilities[0][1].item()  # Deepfake probability
                prediction = "deepfake" if confidence > 0.5 else "real"
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None, 0.0
    
    def predict_video(self, video_path, max_frames=30):
        """Predict if a video contains deepfake content"""
        if self.model is None:
            return {
                "error": "No XceptionNet model loaded",
                "success": False
            }
        
        try:
            video_path = Path(video_path)
            if not video_path.exists():
                return {
                    "error": f"Video file not found: {video_path}",
                    "success": False
                }
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {
                    "error": "Cannot open video file",
                    "success": False
                }
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Sample frames evenly
            if frame_count > max_frames:
                frame_indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)
            else:
                frame_indices = list(range(0, frame_count, max(1, frame_count // max_frames)))
            
            predictions = []
            confidences = []
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    continue
                
                # Extract face
                face_image = self.extract_face_from_frame(frame)
                if face_image is not None:
                    pred, conf = self.predict_frame(face_image)
                    if pred is not None:
                        predictions.append(pred)
                        confidences.append(conf)
            
            cap.release()
            
            if not predictions:
                return {
                    "error": "No faces detected in video",
                    "success": False
                }
            
            # Calculate overall prediction
            avg_confidence = np.mean(confidences)
            deepfake_ratio = sum(1 for p in predictions if p == "deepfake") / len(predictions)
            
            # Final decision
            is_deepfake = avg_confidence > 0.5
            final_prediction = "deepfake" if is_deepfake else "real"
            
            return {
                "success": True,
                "prediction": final_prediction,
                "confidence": float(avg_confidence),
                "deepfake_ratio": float(deepfake_ratio),
                "frames_analyzed": len(predictions),
                "details": {
                    "video_file": str(video_path.name),
                    "total_frames": frame_count,
                    "fps": fps,
                    "model_type": "XceptionNet",
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Video prediction error: {e}")
            return {
                "error": f"Prediction failed: {str(e)}",
                "success": False
            }
    
    def predict_uploaded_file(self, uploaded_file_path, cleanup=True):
        """Predict uploaded video file (for frontend integration)"""
        try:
            result = self.predict_video(uploaded_file_path)
            
            # Cleanup temporary file if requested
            if cleanup and Path(uploaded_file_path).exists():
                try:
                    os.remove(uploaded_file_path)
                    logger.info(f"Cleaned up temporary file: {uploaded_file_path}")
                except:
                    pass
            
            return result
            
        except Exception as e:
            logger.error(f"Upload prediction error: {e}")
            return {
                "error": f"Upload prediction failed: {str(e)}",
                "success": False
            }

# Global predictor instance for Django views
xception_predictor = XceptionNetPredictor()

def get_xception_predictor():
    """Get the global XceptionNet predictor instance"""
    global xception_predictor
    return xception_predictor

def reload_xception_model():
    """Reload the XceptionNet model (useful after training completes)"""
    global xception_predictor
    xception_predictor = XceptionNetPredictor()
    return xception_predictor.model is not None

# Test function
def test_prediction():
    """Test XceptionNet prediction service"""
    predictor = XceptionNetPredictor()
    
    if predictor.model is None:
        print("No XceptionNet model available for testing")
        return
    
    print("XceptionNet prediction service is ready!")
    print(f"Using device: {predictor.device}")
    print("Service can now be integrated with Django frontend")

if __name__ == "__main__":
    test_prediction()
