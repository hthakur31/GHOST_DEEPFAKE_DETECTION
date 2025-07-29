#!/usr/bin/env python3
"""
Enhanced XceptionNet Pre        # Middle flow with residual connections
        self.middle_blocks = nn.ModuleList([
            self._make_residual_block(512, 0.2) for _ in range(4)
        ])ion Service
Supports both old and new improved XceptionNet architectures
"""

import os
import sys
import time
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

class ImprovedXceptionNet(nn.Module):
    """Enhanced XceptionNet architecture with improved depth and regularization"""
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(ImprovedXceptionNet, self).__init__()
        
        # Enhanced feature extraction with deeper architecture
        self.entry_flow = nn.Sequential(
            # Entry flow - Enhanced
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        
        # Enhanced depthwise separable convolutions
        self.separable_blocks = nn.ModuleList([
            self._make_separable_block(64, 128, stride=2, dropout=0.1),
            self._make_separable_block(128, 256, stride=2, dropout=0.15),
            self._make_separable_block(256, 512, stride=2, dropout=0.2),
            self._make_separable_block(512, 512, stride=1, dropout=0.2),  # Additional block
        ])
        
        # Middle flow with residual connections
        self.middle_blocks = nn.ModuleList([
            self._make_residual_block(512, 0.2) for _ in range(4)
        ])
        
        # Exit flow
        self.exit_flow = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=2, padding=1, groups=512),
            nn.Conv2d(1024, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Enhanced classifier with attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1024),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_separable_block(self, in_channels, out_channels, stride=1, dropout=0.1):
        """Create enhanced separable convolution block"""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
    
    def _make_residual_block(self, channels, dropout=0.2):
        """Create residual block for middle flow"""
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Entry flow
        x = self.entry_flow(x)
        
        # Separable blocks
        for block in self.separable_blocks:
            x = block(x)
        
        # Middle flow with residual connections
        for block in self.middle_blocks:
            residual = x
            x = block(x)
            x = x + residual  # Residual connection
            x = torch.relu(x)
        
        # Exit flow
        x = self.exit_flow(x)
        x = x.view(x.size(0), -1)
        
        # Attention mechanism
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Classification
        x = self.classifier(x)
        return x

class SimpleXceptionNet(nn.Module):
    """Simple but effective XceptionNet for deepfake detection with transfer learning"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(SimpleXceptionNet, self).__init__()
        
        # Import here to avoid circular import issues
        from torchvision import models
        
        # Use pretrained ResNet50 backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Remove the last classification layer
        self.backbone.fc = nn.Identity()
        
        # Add custom classifier for deepfake detection
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class LegacyXceptionNet(nn.Module):
    """Legacy XceptionNet for backward compatibility"""
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(LegacyXceptionNet, self).__init__()
        
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

class EnhancedXceptionNetPredictor:
    """Enhanced XceptionNet prediction service with support for multiple architectures"""
    
    def __init__(self, model_path=None, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_type = None
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
        
        # Look for XceptionNet models (prioritize simple models, then improved, then legacy)
        model_patterns = [
            "simple_xception*.pth",
            "improved_xception*.pth", 
            "robust_xception*.pth", 
            "xception_best_*.pth", 
            "xception_final_*.pth", 
            "xception_epoch_*.pth"
        ]
        
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
        import torchvision.transforms as transforms
        
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _detect_model_architecture(self, checkpoint):
        """Detect which model architecture to use based on checkpoint"""
        try:
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Check for simple architecture (ResNet50 backbone)
            has_backbone = any('backbone' in key for key in state_dict.keys())
            has_resnet_features = any('backbone.layer' in key for key in state_dict.keys())
            
            if has_backbone and has_resnet_features:
                return 'simple'
            
            # Check for improved architecture signatures
            has_attention = any('attention' in key for key in state_dict.keys())
            has_residual_blocks = any('middle_blocks' in key for key in state_dict.keys())
            has_entry_flow = any('entry_flow' in key for key in state_dict.keys())
            
            if has_attention and has_residual_blocks and has_entry_flow:
                return 'improved'
            elif 'features' in str(list(state_dict.keys())):
                return 'legacy'
            else:
                return 'legacy'  # Default fallback
                
        except Exception as e:
            logger.warning(f"Could not detect model architecture: {e}")
            return 'legacy'
    
    def load_model(self, model_path):
        """Load trained XceptionNet model with architecture detection"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Detect architecture
            self.model_type = self._detect_model_architecture(checkpoint)
            logger.info(f"Detected model architecture: {self.model_type}")
            
            # Create appropriate model
            if self.model_type == 'simple':
                self.model = SimpleXceptionNet(num_classes=2, pretrained=False).to(self.device)
                logger.info("Using SimpleXceptionNet architecture (ResNet50 backbone)")
            elif self.model_type == 'improved':
                self.model = ImprovedXceptionNet(num_classes=2).to(self.device)
                logger.info("Using ImprovedXceptionNet architecture")
            else:
                self.model = LegacyXceptionNet(num_classes=2).to(self.device)
                logger.info("Using LegacyXceptionNet architecture")
            
            # Load weights
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
                if 'best_accuracy' in checkpoint:
                    logger.info(f"Model best accuracy: {checkpoint['best_accuracy']:.4f}")
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
        """Enhanced face extraction with multiple fallback methods"""
        try:
            if frame is None or frame.size == 0:
                return None
                
            if frame.shape[0] < 50 or frame.shape[1] < 50:
                return None
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Method 1: Try original HOG method with scaling
            face_locations = []
            try:
                # Resize for faster face detection
                height, width = frame_rgb.shape[:2]
                if height > 480 or width > 640:
                    scale = min(480/height, 640/width)
                    new_height, new_width = int(height * scale), int(width * scale)
                    frame_small = cv2.resize(frame_rgb, (new_width, new_height))
                    face_locations = face_recognition.face_locations(frame_small, model='hog', number_of_times_to_upsample=1)
                    # Scale back face locations
                    face_locations = [(int(top/scale), int(right/scale), int(bottom/scale), int(left/scale)) 
                                    for top, right, bottom, left in face_locations]
                else:
                    face_locations = face_recognition.face_locations(frame_rgb, model='hog', number_of_times_to_upsample=1)
            except Exception as e:
                logger.debug(f"HOG face detection failed: {e}")
            
            # Method 2: If HOG failed, try with more upsampling
            if not face_locations:
                try:
                    face_locations = face_recognition.face_locations(frame_rgb, model='hog', number_of_times_to_upsample=2)
                except Exception as e:
                    logger.debug(f"HOG with upsampling failed: {e}")
            
            # Method 3: If still no faces, try CNN method (slower but more accurate)
            if not face_locations:
                try:
                    # Resize to smaller size for CNN to be faster
                    height, width = frame_rgb.shape[:2]
                    if height > 300 or width > 400:
                        scale = min(300/height, 400/width)
                        new_height, new_width = int(height * scale), int(width * scale)
                        frame_small = cv2.resize(frame_rgb, (new_width, new_height))
                        face_locations = face_recognition.face_locations(frame_small, model='cnn')
                        # Scale back face locations
                        face_locations = [(int(top/scale), int(right/scale), int(bottom/scale), int(left/scale)) 
                                        for top, right, bottom, left in face_locations]
                    else:
                        face_locations = face_recognition.face_locations(frame_rgb, model='cnn')
                except Exception as e:
                    logger.debug(f"CNN face detection failed: {e}")
            
            # Method 4: OpenCV Haar Cascades as last resort
            if not face_locations:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Try different Haar cascade files
                    cascade_files = [
                        'haarcascade_frontalface_default.xml',
                        'haarcascade_frontalface_alt.xml',
                        'haarcascade_frontalface_alt2.xml',
                        'haarcascade_profileface.xml'
                    ]
                    
                    for cascade_file in cascade_files:
                        try:
                            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_file)
                            faces_haar = face_cascade.detectMultiScale(
                                gray, 
                                scaleFactor=1.1, 
                                minNeighbors=3,
                                minSize=(30, 30),
                                flags=cv2.CASCADE_SCALE_IMAGE
                            )
                            
                            if len(faces_haar) > 0:
                                # Convert to face_recognition format (top, right, bottom, left)
                                face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in faces_haar]
                                break
                        except Exception as e:
                            logger.debug(f"Haar cascade {cascade_file} failed: {e}")
                            continue
                except Exception as e:
                    logger.debug(f"OpenCV face detection failed: {e}")
            
            if not face_locations:
                return None
            
            # Get the largest face
            largest_face = max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
            top, right, bottom, left = largest_face
            
            # Validate face region
            if bottom <= top or right <= left:
                return None
            
            # Check if face is reasonable size
            face_width = right - left
            face_height = bottom - top
            frame_area = frame_rgb.shape[0] * frame_rgb.shape[1]
            face_area = face_width * face_height
            
            # Face should be at least 0.1% of frame area but not more than 80%
            if face_area < frame_area * 0.001 or face_area > frame_area * 0.8:
                return None
            
            # Add adaptive padding based on face size
            h, w = frame_rgb.shape[:2]
            pad = max(10, min(face_width, face_height) // 8)  # Adaptive padding
            top = max(0, top - pad)
            bottom = min(h, bottom + pad)
            left = max(0, left - pad)
            right = min(w, right + pad)
            
            # Extract face
            face_image = frame_rgb[top:bottom, left:right]
            
            # Validate extracted face
            if face_image.shape[0] < 50 or face_image.shape[1] < 50:
                return None
            
            # Resize to model input size
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
                
                # Get both probabilities for clarity
                real_prob = probabilities[0][0].item()  # Real probability
                fake_prob = probabilities[0][1].item()  # Deepfake probability
                
                # Use a more conservative threshold for better accuracy
                # Real videos should have high real_prob and low fake_prob
                threshold = 0.6  # Increased threshold for more conservative detection
                prediction = "deepfake" if fake_prob > threshold else "real"
                confidence = fake_prob  # Return fake probability as confidence
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None, 0.0
    
    def predict_video(self, video_path, max_frames=30):
        """Predict if a video contains deepfake content with detailed frame-by-frame analysis"""
        if self.model is None:
            return {
                "error": "No XceptionNet model loaded",
                "success": False
            }
        
        try:
            start_time = time.time()
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
            
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            file_size = video_path.stat().st_size / (1024 * 1024)  # MB
            
            # Sample frames evenly
            if frame_count > max_frames:
                frame_indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)
            else:
                frame_indices = list(range(0, frame_count, max(1, frame_count // max_frames)))
            
            predictions = []
            confidences = []
            frame_analysis = []
            faces_detected_total = 0
            
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    continue
                
                # Calculate timestamp
                timestamp = frame_idx / fps if fps > 0 else 0
                
                # Extract face
                face_image = self.extract_face_from_frame(frame)
                face_detected = face_image is not None
                
                if face_detected:
                    faces_detected_total += 1
                    pred, conf = self.predict_frame(face_image)
                    
                    if pred is not None:
                        predictions.append(pred)
                        confidences.append(conf)
                        
                        # Calculate percentages
                        real_percent = (1 - conf) * 100
                        fake_percent = conf * 100
                        
                        frame_analysis.append({
                            "frame_number": int(frame_idx),
                            "timestamp": f"{timestamp:.2f}s",
                            "prediction": "Fake" if pred == "deepfake" else "Real",
                            "confidence": float(conf),
                            "real_percent": f"{real_percent:.1f}%",
                            "fake_percent": f"{fake_percent:.1f}%",
                            "face_detected": "✓" if face_detected else "✗"
                        })
                else:
                    # No face detected in this frame
                    frame_analysis.append({
                        "frame_number": int(frame_idx),
                        "timestamp": f"{timestamp:.2f}s",
                        "prediction": "No Face",
                        "confidence": 0.0,
                        "real_percent": "N/A",
                        "fake_percent": "N/A",
                        "face_detected": "✗"
                    })
            
            cap.release()
            processing_time = time.time() - start_time
            
            if not predictions:
                return {
                    "error": "No faces detected in video",
                    "success": False,
                    "video_analysis": {
                        "file_size_mb": f"{file_size:.1f} MB",
                        "duration": f"{duration:.2f} seconds",
                        "frame_rate": f"{fps:.1f} FPS",
                        "resolution": f"{width}x{height}",
                        "total_frames": frame_count,
                        "frames_analyzed": len(frame_indices),
                        "faces_detected": 0,
                        "processing_time": f"{processing_time:.2f} seconds"
                    }
                }
            
            # Calculate comprehensive statistics
            avg_confidence = np.mean(confidences)
            deepfake_ratio = sum(1 for p in predictions if p == "deepfake") / len(predictions)
            real_ratio = 1 - deepfake_ratio
            
            # Determine final prediction with enhanced logic
            # Use a more conservative threshold for video-level classification
            video_threshold = 0.6  # Increased from 0.5 for better accuracy
            is_deepfake = avg_confidence > video_threshold
            final_prediction = "deepfake" if is_deepfake else "real"
            
            # Calculate confidence level description
            confidence_level = "High" if abs(avg_confidence - 0.5) > 0.3 else "Medium" if abs(avg_confidence - 0.5) > 0.15 else "Low"
            
            # Generate insights
            if final_prediction == "real":
                if avg_confidence < 0.3:
                    insight = "Strong confidence in authentic content. Video appears genuine with high certainty."
                elif avg_confidence < 0.4:
                    insight = "Good confidence in authentic content. Video likely genuine with reasonable certainty."
                else:
                    insight = "Moderate confidence in authentic content. Some uncertainty remains - manual review recommended."
            else:
                if avg_confidence > 0.7:
                    insight = "Strong evidence of deepfake manipulation. High confidence in synthetic content detection."
                elif avg_confidence > 0.6:
                    insight = "Good evidence of deepfake content. Video likely contains manipulated faces."
                else:
                    insight = "Moderate evidence of deepfake content. Further analysis may be needed for certainty."
            
            return {
                "success": True,
                
                # Main Results
                "prediction": final_prediction,
                "prediction_display": "Real Video" if final_prediction == "real" else "Deepfake Video",
                "confidence": float(avg_confidence),
                "confidence_percent": f"{avg_confidence * 100:.2f}%",
                "confidence_level": confidence_level,
                
                # Probability Breakdown
                "authentic_probability": f"{(1 - avg_confidence) * 100:.1f}%",
                "deepfake_probability": f"{avg_confidence * 100:.1f}%",
                "deepfake_ratio": float(deepfake_ratio),
                "real_ratio": float(real_ratio),
                
                # Video Analysis Details
                "video_analysis": {
                    "file_name": str(video_path.name),
                    "file_size": f"{file_size:.1f} MB",
                    "duration": f"{duration:.2f} seconds",
                    "frame_rate": f"{fps:.1f} FPS",
                    "resolution": f"{width}x{height}",
                    "total_frames": frame_count,
                    "frames_analyzed": len(predictions),
                    "faces_detected": faces_detected_total,
                    "faces_detected_text": f"{faces_detected_total} faces",
                    "processing_time": f"{processing_time:.2f} seconds",
                    "analysis_date": datetime.now().strftime("%b %d, %Y %H:%M")
                },
                
                # Detection Insights
                "insights": {
                    "summary": insight,
                    "confidence_description": f"{confidence_level} confidence indicates {'authentic' if final_prediction == 'real' else 'synthetic'} video content"
                },
                
                # AI Model Information
                "model_info": {
                    "model_name": f"Enhanced XceptionNet ({self.model_type})",
                    "version": "2.0",
                    "method": f"Deep Learning - XceptionNet ({self.model_type} architecture)",
                    "training_data": "FaceForensics++ Dataset + Custom Training",
                    "architecture": "ImprovedXceptionNet with Attention" if self.model_type == "improved" else "LegacyXceptionNet"
                },
                
                # Frame-by-Frame Analysis
                "frame_analysis": frame_analysis,
                "frames_analyzed_count": len(predictions),
                "model_type": self.model_type,
                
                # Technical Details
                "technical_details": {
                    "device": str(self.device),
                    "max_frames_analyzed": max_frames,
                    "face_detection_method": "HOG + face_recognition",
                    "image_preprocessing": "224x224 normalization",
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
enhanced_xception_predictor = EnhancedXceptionNetPredictor()

def get_xception_predictor():
    """Get the global enhanced XceptionNet predictor instance"""
    global enhanced_xception_predictor
    return enhanced_xception_predictor

def reload_xception_model():
    """Reload the XceptionNet model (useful after training completes)"""
    global enhanced_xception_predictor
    enhanced_xception_predictor = EnhancedXceptionNetPredictor()
    return enhanced_xception_predictor.model is not None

# Test function
def test_prediction():
    """Test enhanced XceptionNet prediction service"""
    predictor = EnhancedXceptionNetPredictor()
    
    if predictor.model is None:
        print("No XceptionNet model available for testing")
        return
    
    print(f"Enhanced XceptionNet prediction service is ready!")
    print(f"Using device: {predictor.device}")
    print(f"Model architecture: {predictor.model_type}")
    print("Service can now be integrated with Django frontend")

if __name__ == "__main__":
    test_prediction()
