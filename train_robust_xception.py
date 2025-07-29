#!/usr/bin/env python3
"""
Robust XceptionNet Training Script with Corrupted Video Handling
Filters out problematic videos before training to avoid moov atom errors
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import face_recognition
import random
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'robust_xception_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def quick_video_check(video_path, max_timeout=2):
    """
    Ultra-fast check if video is readable
    Returns: True if video seems OK, False if corrupted
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return False
        
        # Check basic properties
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if frame_count <= 0 or fps <= 0:
            cap.release()
            return False
        
        # Quick duration check - skip very long videos
        duration = frame_count / fps if fps > 0 else 0
        if duration > 1200:  # Skip videos longer than 20 minutes
            cap.release()
            return False
        
        # Try to read first frame with strict timeout
        start_time = time.time()
        ret, frame = cap.read()
        elapsed = time.time() - start_time
        
        cap.release()
        
        # If reading took too long or failed, skip this video
        if not ret or elapsed > max_timeout:
            return False
        
        # Check if frame is valid
        if frame is None or frame.shape[0] < 100 or frame.shape[1] < 100:
            return False
        
        return True
        
    except Exception:
        return False

def filter_dataset_videos():
    """Pre-filter all videos to remove corrupted ones"""
    
    dataset_path = Path("G:/Deefake_detection_app/dataset")
    real_path = dataset_path / "original_sequences" / "youtube" / "c23" / "videos"
    fake_path = dataset_path / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
    
    logger.info("Pre-filtering dataset to remove corrupted videos...")
    
    # Get all videos
    all_real_videos = list(real_path.glob("*.mp4")) if real_path.exists() else []
    all_fake_videos = list(fake_path.glob("*.mp4")) if fake_path.exists() else []
    
    logger.info(f"Found {len(all_real_videos)} real videos, {len(all_fake_videos)} fake videos")
    
    # Filter real videos
    valid_real_videos = []
    logger.info("Filtering real videos...")
    for i, video_path in enumerate(all_real_videos):
        if i % 20 == 0:
            logger.info(f"Checking real video {i+1}/{len(all_real_videos)}")
        
        if quick_video_check(video_path):
            valid_real_videos.append(video_path)
        else:
            logger.debug(f"Skipping corrupted real video: {video_path.name}")
    
    # Filter fake videos
    valid_fake_videos = []
    logger.info("Filtering fake videos...")
    for i, video_path in enumerate(all_fake_videos):
        if i % 20 == 0:
            logger.info(f"Checking fake video {i+1}/{len(all_fake_videos)}")
        
        if quick_video_check(video_path):
            valid_fake_videos.append(video_path)
        else:
            logger.debug(f"Skipping corrupted fake video: {video_path.name}")
    
    logger.info(f"After filtering:")
    logger.info(f"  Valid real videos: {len(valid_real_videos)} (removed {len(all_real_videos) - len(valid_real_videos)})")
    logger.info(f"  Valid fake videos: {len(valid_fake_videos)} (removed {len(all_fake_videos) - len(valid_fake_videos)})")
    
    return valid_real_videos, valid_fake_videos

def extract_face_from_video_robust(video_path, target_frames=10, max_attempts=20):
    """
    Ultra-robust face extraction with extensive error handling
    """
    cap = None
    try:
        # Quick pre-check
        if not quick_video_check(video_path):
            return None
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return None
        
        # Get frame indices to sample
        frame_indices = np.linspace(0, total_frames - 1, min(target_frames, total_frames), dtype=int)
        
        attempts = 0
        for frame_idx in frame_indices:
            if attempts >= max_attempts:
                break
            
            attempts += 1
            
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                # Set timeout for frame reading
                start_time = time.time()
                ret, frame = cap.read()
                
                if time.time() - start_time > 3:  # 3 second timeout
                    logger.debug(f"Frame read timeout for {video_path}")
                    continue
                
                if not ret or frame is None:
                    continue
                
                # Validate frame
                if frame.shape[0] < 100 or frame.shape[1] < 100:
                    continue
                
                # Convert BGR to RGB with validation
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize for faster face detection
                height, width = frame_rgb.shape[:2]
                if height > 640 or width > 640:
                    scale = 640 / max(height, width)
                    new_height, new_width = int(height * scale), int(width * scale)
                    frame_small = cv2.resize(frame_rgb, (new_width, new_height))
                else:
                    frame_small = frame_rgb
                    scale = 1.0
                
                # Face detection with timeout
                try:
                    face_start_time = time.time()
                    face_locations = face_recognition.face_locations(frame_small, model='hog')
                    face_time = time.time() - face_start_time
                    
                    if face_time > 5:  # Face detection took too long
                        logger.debug(f"Face detection timeout for {video_path}")
                        continue
                    
                except Exception as face_error:
                    logger.debug(f"Face detection failed for {video_path}: {face_error}")
                    continue
                
                if not face_locations:
                    continue
                
                # Get the largest face and scale back if needed
                largest_face = max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
                
                if scale != 1.0:
                    top, right, bottom, left = largest_face
                    top, right, bottom, left = int(top/scale), int(right/scale), int(bottom/scale), int(left/scale)
                    # Use original frame
                    face_image = frame_rgb[top:bottom, left:right]
                else:
                    top, right, bottom, left = largest_face
                    face_image = frame_small[top:bottom, left:right]
                
                # Validate face region
                if face_image.shape[0] < 50 or face_image.shape[1] < 50:
                    continue
                
                # Resize to target size
                face_image = cv2.resize(face_image, (224, 224))
                
                return face_image
                
            except Exception as e:
                logger.debug(f"Error processing frame {frame_idx} in {video_path}: {e}")
                continue
        
        return None
        
    except Exception as e:
        logger.debug(f"Error extracting face from {video_path}: {e}")
        return None
    finally:
        if cap is not None:
            cap.release()

# Use the existing XceptionNet class from the original file
class XceptionNet(nn.Module):
    """XceptionNet for deepfake detection"""
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(XceptionNet, self).__init__()
        
        # Simplified Xception-like architecture
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

class RobustFaceForensicsDataset(Dataset):
    """Robust dataset class that pre-filters corrupted videos"""
    
    def __init__(self, real_videos, fake_videos, transform=None):
        self.real_videos = real_videos
        self.fake_videos = fake_videos
        self.transform = transform
        
        # Create labels
        self.videos = []
        self.labels = []
        
        for video_path in self.real_videos:
            self.videos.append(video_path)
            self.labels.append(0)  # Real = 0
        
        for video_path in self.fake_videos:
            self.videos.append(video_path)
            self.labels.append(1)  # Fake = 1
        
        logger.info(f"Dataset created with {len(self.videos)} videos ({len(real_videos)} real, {len(fake_videos)} fake)")
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video_path = self.videos[idx]
        label = self.labels[idx]
        
        # Extract face with robust error handling
        face_image = extract_face_from_video_robust(video_path)
        
        if face_image is None:
            # Return a black image if face extraction fails
            face_image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        if self.transform:
            face_image = self.transform(face_image)
        
        return face_image, torch.tensor(label, dtype=torch.long)

def main():
    """Main training function with robust video handling"""
    
    logger.info("Starting robust XceptionNet training...")
    
    # Filter dataset first
    valid_real_videos, valid_fake_videos = filter_dataset_videos()
    
    if len(valid_real_videos) == 0 or len(valid_fake_videos) == 0:
        logger.error("No valid videos found! Check your dataset path and video files.")
        return
    
    # Limit dataset size for testing
    max_videos_per_class = 30
    if len(valid_real_videos) > max_videos_per_class:
        valid_real_videos = random.sample(valid_real_videos, max_videos_per_class)
    if len(valid_fake_videos) > max_videos_per_class:
        valid_fake_videos = random.sample(valid_fake_videos, max_videos_per_class)
    
    logger.info(f"Using {len(valid_real_videos)} real and {len(valid_fake_videos)} fake videos for training")
    
    # Create transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = RobustFaceForensicsDataset(
        real_videos=valid_real_videos,
        fake_videos=valid_fake_videos,
        transform=transform
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=4,  # Smaller batch size for stability
        shuffle=True, 
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=False
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = XceptionNet(num_classes=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    logger.info(f"Training on device: {device}")
    logger.info(f"Dataset size: {len(dataset)} samples")
    
    # Training loop
    num_epochs = 5  # Shorter training for testing
    model.train()
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (data, labels) in enumerate(progress_bar):
            try:
                data, labels = data.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                accuracy = 100. * correct / total
                progress_bar.set_postfix({
                    'Loss': f'{total_loss/(batch_idx+1):.4f}',
                    'Acc': f'{accuracy:.2f}%'
                })
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        epoch_loss = total_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        
        logger.info(f"Epoch {epoch+1} completed - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
        # Save model checkpoint
        if epoch % 2 == 0:  # Save every 2 epochs
            model_path = f"models/robust_xception_epoch_{epoch+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            logger.info(f"Model saved: {model_path}")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
