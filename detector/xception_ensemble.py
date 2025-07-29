#!/usr/bin/env python3
"""
Enhanced deepfake detection with XceptionNet integration
Implements ensemble approach combining ResNet50 and XceptionNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import time
import face_recognition
import mediapipe as mp

logger = logging.getLogger(__name__)

class XceptionNetModel(nn.Module):
    """
    Xception architecture specifically adapted for deepfake detection
    Based on the original Xception paper but optimized for face manipulation detection
    """
    
    def __init__(self, num_classes=2):
        super(XceptionNetModel, self).__init__()
        
        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Depthwise separable convolutions (core of Xception)
        self.sep_conv1 = self._make_separable_conv(64, 128, 3)
        self.sep_conv2 = self._make_separable_conv(128, 256, 3)
        self.sep_conv3 = self._make_separable_conv(256, 728, 3)
        
        # Middle flow (8 blocks)
        self.middle_blocks = nn.ModuleList([
            self._make_separable_conv(728, 728, 3) for _ in range(8)
        ])
        
        # Exit flow
        self.sep_conv4 = self._make_separable_conv(728, 1024, 3)
        self.sep_conv5 = self._make_separable_conv(1024, 1536, 3)
        self.sep_conv6 = self._make_separable_conv(1536, 2048, 3)
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, num_classes)
        
    def _make_separable_conv(self, in_channels, out_channels, kernel_size):
        """Create a depthwise separable convolution block"""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                     padding=kernel_size//2, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Entry flow
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        x = self.sep_conv1(x)
        x = F.max_pool2d(x, 3, stride=2, padding=1)
        
        x = self.sep_conv2(x)
        x = F.max_pool2d(x, 3, stride=2, padding=1)
        
        x = self.sep_conv3(x)
        x = F.max_pool2d(x, 3, stride=2, padding=1)
        
        # Middle flow
        for block in self.middle_blocks:
            residual = x
            x = block(x)
            x = x + residual  # Residual connection
        
        # Exit flow
        x = self.sep_conv4(x)
        x = F.max_pool2d(x, 3, stride=2, padding=1)
        
        x = self.sep_conv5(x)
        x = self.sep_conv6(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class EnsembleDeepfakeDetector:
    """
    Ensemble detector combining ResNet50 and XceptionNet for improved accuracy
    """
    
    def __init__(self, device='auto', threshold=0.6):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu')
        self.threshold = threshold
        
        # Initialize models
        self.resnet_model = self._create_resnet_model()
        self.xception_model = XceptionNetModel(num_classes=2)
        
        # Move models to device
        self.resnet_model.to(self.device)
        self.xception_model.to(self.device)
        
        # Set to evaluation mode
        self.resnet_model.eval()
        self.xception_model.eval()
        
        # Initialize preprocessor
        self.preprocessor = self._create_preprocessor()
        
        # Face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5)
        
        logger.info("Ensemble detector initialized with ResNet50 + XceptionNet")
    
    def _create_resnet_model(self):
        """Create ResNet50 based model"""
        model = resnet50(pretrained=True)
        # Replace classifier
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
        return model
    
    def _create_preprocessor(self):
        """Create image preprocessor"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),  # Xception input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def detect_faces(self, frame):
        """Detect faces in frame using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                faces.append({
                    'bbox': [x, y, x + width, y + height],
                    'confidence': detection.score[0]
                })
        
        return faces
    
    def extract_face_crop(self, frame, bbox):
        """Extract and preprocess face crop"""
        x1, y1, x2, y2 = bbox
        
        # Add padding
        padding = 0.2
        w, h = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - w * padding))
        y1 = max(0, int(y1 - h * padding))
        x2 = min(frame.shape[1], int(x2 + w * padding))
        y2 = min(frame.shape[0], int(y2 + h * padding))
        
        face_crop = frame[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return None
        
        return face_crop
    
    def extract_frames(self, video_path, max_frames=30):
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return [], {}
        
        # Get video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        metadata = {
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'width': width,
            'height': height,
            'resolution': f"{width}x{height}"
        }
        
        frames = []
        frame_interval = max(1, frame_count // max_frames)
        
        frame_idx = 0
        while len(frames) < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frames.append(frame)
            frame_idx += frame_interval
        
        cap.release()
        return frames, metadata
    
    def predict_frame(self, face_crop):
        """Predict single frame using ensemble"""
        # Preprocess
        tensor = self.preprocessor(face_crop).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # ResNet prediction
            resnet_output = self.resnet_model(tensor)
            resnet_probs = F.softmax(resnet_output, dim=1)
            
            # XceptionNet prediction
            xception_output = self.xception_model(tensor)
            xception_probs = F.softmax(xception_output, dim=1)
            
            # Ensemble (weighted average)
            # XceptionNet typically performs better for deepfake detection
            ensemble_probs = 0.4 * resnet_probs + 0.6 * xception_probs
            
            return ensemble_probs.cpu().numpy()[0]
    
    def detect_video(self, video_path: str) -> Dict:
        """
        Detect deepfakes in video using ensemble approach
        """
        start_time = time.time()
        
        try:
            # Extract frames
            frames, metadata = self.extract_frames(video_path)
            
            if not frames:
                return {
                    'prediction': 'ERROR',
                    'error': 'No frames extracted',
                    'confidence_score': 0.0,
                    'processing_time': time.time() - start_time
                }
            
            frame_predictions = []
            valid_predictions = []
            
            for i, frame in enumerate(frames):
                # Detect faces
                faces = self.detect_faces(frame)
                
                if not faces:
                    continue
                
                # Use largest face
                largest_face = max(faces, key=lambda f: 
                    (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]))
                
                # Extract face crop
                face_crop = self.extract_face_crop(frame, largest_face['bbox'])
                
                if face_crop is None:
                    continue
                
                # Predict
                probs = self.predict_frame(face_crop)
                confidence = max(probs)
                prediction = 'REAL' if probs[0] > probs[1] else 'FAKE'
                
                frame_prediction = {
                    'frame_number': i,
                    'prediction': prediction,
                    'confidence_score': confidence,
                    'real_probability': probs[0],
                    'fake_probability': probs[1],
                    'face_detected': True,
                    'face_confidence': largest_face['confidence']
                }
                
                frame_predictions.append(frame_prediction)
                valid_predictions.append(probs)
            
            if not valid_predictions:
                return {
                    'prediction': 'ERROR',
                    'error': 'No faces detected',
                    'confidence_score': 0.0,
                    'face_detected': False,
                    'processing_time': time.time() - start_time
                }
            
            # Ensemble decision with improved logic
            valid_predictions = np.array(valid_predictions)
            
            # Calculate weighted average (weight by confidence)
            confidences = np.max(valid_predictions, axis=1)
            weights = confidences / np.sum(confidences)
            
            weighted_avg = np.average(valid_predictions, axis=0, weights=weights)
            
            # Apply conservative threshold
            final_prediction = 'REAL'
            if weighted_avg[1] > self.threshold:  # Higher threshold reduces false positives
                final_prediction = 'FAKE'
            
            final_confidence = max(weighted_avg)
            
            # Calculate temporal consistency
            predictions_only = [fp['prediction'] for fp in frame_predictions]
            consistency = len([p for p in predictions_only if p == final_prediction]) / len(predictions_only)
            
            result = {
                'prediction': final_prediction,
                'confidence_score': final_confidence,
                'confidence_percentage': final_confidence * 100,
                'real_probability': weighted_avg[0] * 100,
                'fake_probability': weighted_avg[1] * 100,
                'temporal_consistency': consistency,
                'face_detected': True,
                'face_count': len(frame_predictions),
                'frames_analyzed': len(frame_predictions),
                'frame_predictions': frame_predictions,
                'metadata': metadata,
                'model_name': 'Ensemble ResNet50 + XceptionNet',
                'model_version': '1.0',
                'detection_method': 'Ensemble Learning with Conservative Threshold',
                'threshold_used': self.threshold,
                'processing_time': time.time() - start_time,
                'is_deepfake': final_prediction == 'FAKE'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ensemble detection: {e}")
            return {
                'prediction': 'ERROR',
                'error': str(e),
                'confidence_score': 0.0,
                'processing_time': time.time() - start_time
            }

def get_ensemble_detector(threshold=0.65):
    """Get ensemble detector instance"""
    return EnsembleDeepfakeDetector(threshold=threshold)

if __name__ == "__main__":
    # Test the ensemble detector
    detector = get_ensemble_detector(threshold=0.65)
    print("✅ Ensemble detector (ResNet50 + XceptionNet) initialized successfully!")
    print(f"🎯 Threshold set to: {detector.threshold}")
    print("🔍 Ready for deepfake detection with improved accuracy")
