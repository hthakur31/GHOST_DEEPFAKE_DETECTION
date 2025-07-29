#!/usr/bin/env python3
"""
XceptionNet Training Script for Deepfake Detection
Using FaceForensics++ dataset structure
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
import cv2
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import face_recognition
import random

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'xception_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class XceptionNet(nn.Module):
    """
    XceptionNet architecture for deepfake detection
    Based on the Xception paper but adapted for binary classification
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(XceptionNet, self).__init__()
        
        # Entry flow
        self.entry_conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.entry_bn1 = nn.BatchNorm2d(32)
        self.entry_relu1 = nn.ReLU(inplace=True)
        
        self.entry_conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.entry_bn2 = nn.BatchNorm2d(64)
        self.entry_relu2 = nn.ReLU(inplace=True)
        
        # Depthwise separable convolution blocks
        self.sep_conv1 = self._make_separable_conv(64, 128, stride=2)
        self.sep_conv2 = self._make_separable_conv(128, 256, stride=2)
        self.sep_conv3 = self._make_separable_conv(256, 512, stride=2)
        self.sep_conv4 = self._make_separable_conv(512, 728, stride=2)
        
        # Middle flow (8 repeated blocks)
        self.middle_blocks = nn.ModuleList([
            self._make_separable_conv(728, 728, stride=1) for _ in range(8)
        ])
        
        # Exit flow
        self.exit_conv1 = self._make_separable_conv(728, 1024, stride=2)
        self.exit_conv2 = nn.Conv2d(1024, 1536, 3, padding=1, bias=False)
        self.exit_bn2 = nn.BatchNorm2d(1536)
        self.exit_relu2 = nn.ReLU(inplace=True)
        
        self.exit_conv3 = nn.Conv2d(1536, 2048, 3, padding=1, bias=False)
        self.exit_bn3 = nn.BatchNorm2d(2048)
        self.exit_relu3 = nn.ReLU(inplace=True)
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(2048, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_separable_conv(self, in_channels, out_channels, stride=1):
        """Create a depthwise separable convolution block"""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, 
                     groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Entry flow
        x = self.entry_relu1(self.entry_bn1(self.entry_conv1(x)))
        x = self.entry_relu2(self.entry_bn2(self.entry_conv2(x)))
        
        # Separable convolution blocks
        x = self.sep_conv1(x)
        x = self.sep_conv2(x)
        x = self.sep_conv3(x)
        x = self.sep_conv4(x)
        
        # Middle flow
        for block in self.middle_blocks:
            residual = x
            x = block(x)
            x = x + residual  # Residual connection
        
        # Exit flow
        x = self.exit_conv1(x)
        x = self.exit_relu2(self.exit_bn2(self.exit_conv2(x)))
        x = self.exit_relu3(self.exit_bn3(self.exit_conv3(x)))
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


class FaceForensicsXceptionDataset(Dataset):
    """
    Dataset class for loading FaceForensics++ data for XceptionNet training
    """
    
    def __init__(self, real_videos, fake_videos, transform=None, frames_per_video=10):
        self.real_videos = real_videos
        self.fake_videos = fake_videos
        self.transform = transform
        self.frames_per_video = frames_per_video
        
        # Create label mapping
        self.samples = []
        for video_path in real_videos:
            self.samples.append((video_path, 0))  # 0 = Real
        for video_path in fake_videos:
            self.samples.append((video_path, 1))  # 1 = Fake
        
        logger.info(f"Dataset created with {len(self.samples)} videos")
        logger.info(f"Real videos: {len(real_videos)}, Fake videos: {len(fake_videos)}")
    
    def __len__(self):
        return len(self.samples) * self.frames_per_video
    
    def __getitem__(self, idx):
        video_idx = idx // self.frames_per_video
        frame_idx = idx % self.frames_per_video
        
        video_path, label = self.samples[video_idx]
        
        try:
            # Extract face from video frame
            face_image = self.extract_face_from_video(video_path, frame_idx)
            
            if face_image is None:
                # If no face found, try another frame
                face_image = self.extract_face_from_video(video_path, 0)
            
            if face_image is None:
                # If still no face, create a dummy image
                face_image = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # Apply transforms
            if self.transform:
                face_image = self.transform(face_image)
            
            return face_image, label
            
        except Exception as e:
            logger.warning(f"Error processing {video_path}: {e}")
            # Return dummy data
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, label
    
    def extract_face_from_video(self, video_path, frame_idx):
        """Extract a face from a specific frame of the video"""
        cap = None
        try:
            # Add timeout and error handling for corrupted videos
            cap = cv2.VideoCapture(str(video_path))
            
            # Set timeout to prevent hanging
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0 or total_frames > 10000:  # Skip very long videos
                if cap:
                    cap.release()
                return None
            
            # Calculate actual frame index (limit to reasonable range)
            frames_to_sample = min(self.frames_per_video, total_frames)
            actual_frame_idx = min(frame_idx * (total_frames // frames_to_sample), total_frames - 1)
            
            # Try to set frame position
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_idx):
                # If seeking fails, try reading from beginning
                actual_frame_idx = min(frame_idx * 5, total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_idx)
            
            ret, frame = cap.read()
            
            if not ret or frame is None:
                # Try reading a few more frames
                for _ in range(3):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        break
                
            if cap:
                cap.release()
                
            if not ret or frame is None:
                return None
            
            # Convert BGR to RGB (check if frame is valid)
            if frame.shape[0] < 50 or frame.shape[1] < 50:  # Skip tiny frames
                return None
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use faster face detection with timeout
            try:
                # Resize frame for faster face detection
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
                    
            except Exception as face_error:
                logger.debug(f"Face detection failed for {video_path}: {face_error}")
                return None
            
            if not face_locations:
                return None
            
            # Get the largest face
            largest_face = max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
            top, right, bottom, left = largest_face
            
            # Validate face region
            if bottom <= top or right <= left:
                return None
            
            # Add padding but keep within image bounds
            h, w = frame_rgb.shape[:2]
            pad = min(20, (bottom - top) // 4)  # Dynamic padding
            top = max(0, top - pad)
            bottom = min(h, bottom + pad)
            left = max(0, left - pad)
            right = min(w, right + pad)
            
            # Extract and resize face
            face_image = frame_rgb[top:bottom, left:right]
            
            # Check if face region is valid
            if face_image.shape[0] < 32 or face_image.shape[1] < 32:
                return None
                
            face_image = cv2.resize(face_image, (224, 224))
            
            return face_image
            
        except Exception as e:
            logger.debug(f"Error extracting face from {video_path}: {e}")
            return None
        finally:
            if cap is not None:
                cap.release()


def get_data_transforms():
    """Get data augmentation transforms"""
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def load_dataset(dataset_path, train_split=0.8):
    """Load and split the FaceForensics++ dataset"""
    dataset_path = Path(dataset_path)
    
    # Real videos
    real_path = dataset_path / "original_sequences" / "youtube" / "c23" / "videos"
    real_videos = list(real_path.glob("*.mp4"))
    
    # Fake videos (Deepfakes)
    fake_path = dataset_path / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
    fake_videos = list(fake_path.glob("*.mp4"))
    
    logger.info(f"Found {len(real_videos)} real videos and {len(fake_videos)} fake videos")
    
    # Shuffle and split
    random.shuffle(real_videos)
    random.shuffle(fake_videos)
    
    real_split = int(len(real_videos) * train_split)
    fake_split = int(len(fake_videos) * train_split)
    
    train_real = real_videos[:real_split]
    val_real = real_videos[real_split:]
    train_fake = fake_videos[:fake_split]
    val_fake = fake_videos[fake_split:]
    
    logger.info(f"Train: {len(train_real)} real, {len(train_fake)} fake")
    logger.info(f"Val: {len(val_real)} real, {len(val_fake)} fake")
    
    return train_real, train_fake, val_real, val_fake


def train_xception_model(dataset_path, num_epochs=50, batch_size=16, learning_rate=0.001):
    """Train XceptionNet model"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
    train_real, train_fake, val_real, val_fake = load_dataset(dataset_path)
    
    # Get transforms
    train_transform, val_transform = get_data_transforms()
    
    # Create datasets
    train_dataset = FaceForensicsXceptionDataset(
        real_videos=train_real,
        fake_videos=train_fake,
        transform=train_transform,
        frames_per_video=8
    )
    
    val_dataset = FaceForensicsXceptionDataset(
        real_videos=val_real,
        fake_videos=val_fake,
        transform=val_transform,
        frames_per_video=5
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    model = XceptionNet(num_classes=2, dropout_rate=0.3)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training tracking
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0.0
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            train_acc = 100. * train_correct / train_total
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{train_acc:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                # Update progress bar
                val_acc = 100. * val_correct / val_total
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{val_acc:.2f}%'
                })
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Log results
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        logger.info(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'train_loss': train_loss
            }, f'models/xception_best_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
            logger.info(f'  New best model saved! Val Acc: {val_acc:.2f}%')
        
        # Early stopping check
        if epoch > 10 and val_acc < best_val_acc - 10:
            logger.info("Early stopping triggered")
            break
    
    # Final evaluation
    logger.info(f"\nTraining completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Classification report
    if len(set(all_targets)) > 1:
        report = classification_report(all_targets, all_predictions, 
                                     target_names=['Real', 'Fake'], 
                                     output_dict=True)
        logger.info("Final Classification Report:")
        logger.info(f"Real - Precision: {report['Real']['precision']:.3f}, Recall: {report['Real']['recall']:.3f}")
        logger.info(f"Fake - Precision: {report['Fake']['precision']:.3f}, Recall: {report['Fake']['recall']:.3f}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }
    
    with open(f'models/xception_training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history


if __name__ == "__main__":
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Configuration
    DATASET_PATH = "G:/Deefake_detection_app/dataset"
    NUM_EPOCHS = 30
    BATCH_SIZE = 8  # Reduced for memory efficiency
    LEARNING_RATE = 0.0001
    
    logger.info("Starting XceptionNet training...")
    logger.info(f"Dataset path: {DATASET_PATH}")
    logger.info(f"Epochs: {NUM_EPOCHS}, Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    
    # Train model
    try:
        model, history = train_xception_model(
            dataset_path=DATASET_PATH,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE
        )
        logger.info("XceptionNet training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
