#!/usr/bin/env python3
"""
Improved XceptionNet Training Script
Uses dataset folder for training with enhanced data augmentation and optimization
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import face_recognition
from pathlib import Path
import logging
import json
from datetime import datetime
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time

# Setup enhanced logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"improved_xception_training_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
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
            self._make_residual_block(512, dropout_rate=0.2) for _ in range(4)
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
    
    def _make_residual_block(self, channels, dropout_rate=0.2):
        """Create residual block for middle flow"""
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
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

class EnhancedFaceForensicsDataset(Dataset):
    """Enhanced dataset class with better augmentation and face extraction"""
    
    def __init__(self, real_videos_path, fake_videos_path, transform=None, max_frames_per_video=10, face_size=224):
        self.real_videos_path = Path(real_videos_path)
        self.fake_videos_path = Path(fake_videos_path)
        self.transform = transform
        self.max_frames_per_video = max_frames_per_video
        self.face_size = face_size
        
        # Load video paths
        self.video_paths = []
        self.labels = []
        
        # Real videos (label = 0)
        real_videos = list(self.real_videos_path.glob("*.mp4"))
        for video_path in real_videos:
            self.video_paths.append(video_path)
            self.labels.append(0)
        
        # Fake videos (label = 1)
        fake_videos = list(self.fake_videos_path.glob("*.mp4"))
        for video_path in fake_videos:
            self.video_paths.append(video_path)
            self.labels.append(1)
        
        logger.info(f"Dataset loaded: {len(real_videos)} real videos, {len(fake_videos)} fake videos")
        
        # Pre-extract faces for faster training
        self._preextract_faces()
    
    def _preextract_faces(self):
        """Pre-extract faces from all videos for faster training"""
        logger.info("Pre-extracting faces from videos...")
        self.face_data = []
        
        for i, (video_path, label) in enumerate(tqdm(zip(self.video_paths, self.labels), 
                                                     total=len(self.video_paths), 
                                                     desc="Extracting faces")):
            faces = self._extract_faces_from_video(video_path)
            
            for face in faces:
                self.face_data.append((face, label))
        
        logger.info(f"Extracted {len(self.face_data)} face samples")
    
    def _extract_faces_from_video(self, video_path):
        """Extract faces from a single video"""
        faces = []
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.warning(f"Cannot open video: {video_path}")
            return faces
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames evenly
        if frame_count > self.max_frames_per_video:
            frame_indices = np.linspace(0, frame_count - 1, self.max_frames_per_video, dtype=int)
        else:
            frame_indices = list(range(frame_count))
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                continue
            
            # Extract face
            face = self._extract_face_from_frame(frame)
            if face is not None:
                faces.append(face)
                
                # Stop if we have enough faces
                if len(faces) >= self.max_frames_per_video:
                    break
        
        cap.release()
        return faces
    
    def _extract_face_from_frame(self, frame):
        """Extract and preprocess face from frame"""
        try:
            if frame.shape[0] < 50 or frame.shape[1] < 50:
                return None
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize for faster face detection
            height, width = frame_rgb.shape[:2]
            if height > 640 or width > 640:
                scale = min(640/height, 640/width)
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
            pad = max(20, (bottom - top) // 8)
            top = max(0, top - pad)
            bottom = min(h, bottom + pad)
            left = max(0, left - pad)
            right = min(w, right + pad)
            
            # Extract and resize face
            face_image = frame_rgb[top:bottom, left:right]
            
            if face_image.shape[0] < 64 or face_image.shape[1] < 64:
                return None
            
            face_image = cv2.resize(face_image, (self.face_size, self.face_size))
            return face_image
            
        except Exception as e:
            logger.debug(f"Face extraction error: {e}")
            return None
    
    def __len__(self):
        return len(self.face_data)
    
    def __getitem__(self, idx):
        face, label = self.face_data[idx]
        
        if self.transform:
            # Apply torchvision transforms
            face = self.transform(face)
        else:
            # Convert to tensor manually
            face = torch.from_numpy(face.transpose(2, 0, 1)).float() / 255.0
        
        return face, label

def get_enhanced_transforms():
    """Get enhanced data augmentation transforms using torchvision"""
    
    # Training transforms with enhanced augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ], p=0.3),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5
            )
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
    ])
    
    # Validation transforms
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform

def train_model():
    """Main training function with enhanced features"""
    logger.info("Starting Enhanced XceptionNet Training")
    
    # Configuration
    config = {
        'epochs': 15,
        'batch_size': 16,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'face_size': 224,
        'max_frames_per_video': 15,
        'patience': 5,
        'min_delta': 0.001
    }
    
    logger.info(f"Training configuration: {config}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Dataset paths
    real_videos_path = "dataset/original_sequences/youtube/c23/videos"
    fake_videos_path = "dataset/manipulated_sequences/Deepfakes/c23/videos"
    
    # Check if paths exist
    if not Path(real_videos_path).exists():
        logger.error(f"Real videos path not found: {real_videos_path}")
        return
    
    if not Path(fake_videos_path).exists():
        logger.error(f"Fake videos path not found: {fake_videos_path}")
        return
    
    # Get transforms
    train_transform, val_transform = get_enhanced_transforms()
    
    # Create full dataset
    logger.info("Loading full dataset...")
    full_dataset = EnhancedFaceForensicsDataset(
        real_videos_path=real_videos_path,
        fake_videos_path=fake_videos_path,
        transform=None,  # We'll apply transforms after split
        max_frames_per_video=config['max_frames_per_video'],
        face_size=config['face_size']
    )
    
    if len(full_dataset) == 0:
        logger.error("No data found in dataset!")
        return
    
    # Split dataset into train/validation
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    labels = [full_dataset.face_data[i][1] for i in indices]
    
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Create train and validation datasets
    train_data = [(full_dataset.face_data[i][0], full_dataset.face_data[i][1]) for i in train_indices]
    val_data = [(full_dataset.face_data[i][0], full_dataset.face_data[i][1]) for i in val_indices]
    
    # Custom dataset classes for train/val with different transforms
    class TransformDataset(Dataset):
        def __init__(self, data, transform):
            self.data = data
            self.transform = transform
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            face, label = self.data[idx]
            
            if self.transform:
                face = self.transform(face)
            else:
                face = torch.from_numpy(face.transpose(2, 0, 1)).float() / 255.0
            
            return face, label
    
    train_dataset = TransformDataset(train_data, train_transform)
    val_dataset = TransformDataset(val_data, val_transform)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Data loaders (num_workers=0 to avoid Windows multiprocessing issues)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Model
    model = ImprovedXceptionNet(num_classes=2, dropout_rate=0.5).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training tracking
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    training_history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rates': []
    }
    
    # Training loop
    logger.info("Starting training loop...")
    
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} - Training")
        
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} - Validation")
            
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
                
                val_predictions.extend(predicted.cpu().numpy())
                val_labels.extend(target.cpu().numpy())
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100*val_correct/val_total:.2f}%'
                })
        
        # Calculate metrics
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        
        # Calculate detailed metrics
        precision = precision_score(val_labels, val_predictions, average='binary')
        recall = recall_score(val_labels, val_predictions, average='binary')
        f1 = f1_score(val_labels, val_predictions, average='binary')
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save metrics
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['learning_rates'].append(current_lr)
        
        epoch_time = time.time() - epoch_start_time
        
        # Log epoch results
        logger.info(f"""
        Epoch {epoch+1}/{config['epochs']} - Time: {epoch_time:.2f}s
        Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%
        Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%
        Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}
        Learning Rate: {current_lr:.6f}
        """)
        
        # Save best model
        if val_acc > best_val_acc + config['min_delta']:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_val_acc,
                'config': config,
                'training_history': training_history
            }
            
            model_filename = f"models/improved_xception_best_{timestamp}.pth"
            os.makedirs("models", exist_ok=True)
            torch.save(checkpoint, model_filename)
            logger.info(f"New best model saved: {model_filename} (Val Acc: {val_acc:.2f}%)")
            
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= config['patience']:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save epoch checkpoint
        epoch_checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': val_acc,
            'config': config
        }
        epoch_filename = f"models/improved_xception_epoch_{epoch+1}_{timestamp}.pth"
        torch.save(epoch_checkpoint, epoch_filename)
    
    # Final evaluation
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model for final evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Generate confusion matrix
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    # Plot training history and confusion matrix
    plot_training_results(training_history, all_labels, all_predictions, timestamp)
    
    # Save final model
    final_model_path = f"models/improved_xception_final_{timestamp}.pth"
    final_checkpoint = {
        'model_state_dict': best_model_state or model.state_dict(),
        'best_accuracy': best_val_acc,
        'config': config,
        'training_history': training_history,
        'final_metrics': {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions, average='binary'),
            'recall': recall_score(all_labels, all_predictions, average='binary'),
            'f1': f1_score(all_labels, all_predictions, average='binary')
        }
    }
    torch.save(final_checkpoint, final_model_path)
    logger.info(f"Final model saved: {final_model_path}")
    
    return final_model_path

def plot_training_results(history, true_labels, predictions, timestamp):
    """Plot training results and confusion matrix"""
    try:
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training/Validation Loss
        axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Training/Validation Accuracy
        axes[0, 1].plot(history['train_acc'], label='Train Accuracy', color='blue')
        axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', color='red')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning Rate
        axes[1, 0].plot(history['learning_rates'], color='green')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Confusion Matrix
        cm = confusion_matrix(true_labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title('Confusion Matrix')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('True')
        axes[1, 1].set_xticklabels(['Real', 'Fake'])
        axes[1, 1].set_yticklabels(['Real', 'Fake'])
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"training_results_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        logger.info(f"Training plots saved: {plot_filename}")
        
        plt.close()
        
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")

if __name__ == "__main__":
    try:
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # Train the model
        model_path = train_model()
        
        if model_path:
            logger.info(f"Training completed successfully! Model saved at: {model_path}")
            logger.info("You can now use this improved model for better deepfake detection.")
        else:
            logger.error("Training failed!")
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
