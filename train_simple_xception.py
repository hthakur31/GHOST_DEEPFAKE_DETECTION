#!/usr/bin/env python3
"""
Simple and Effective XceptionNet Training for Deepfake Detection
Uses transfer learning and proven techniques for better results
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
import numpy as np
import face_recognition
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleXceptionNet(nn.Module):
    """Simple but effective XceptionNet for deepfake detection"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(SimpleXceptionNet, self).__init__()
        
        # Use pretrained Xception backbone (we'll use ResNet50 as proxy since Xception is not in torchvision)
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
        
        # Initialize classifier weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class DeepfakeDataset(Dataset):
    """Dataset for deepfake detection with robust preprocessing"""
    
    def __init__(self, real_videos_path, fake_videos_path, transform=None, max_frames_per_video=20, face_size=224):
        self.real_videos_path = Path(real_videos_path)
        self.fake_videos_path = Path(fake_videos_path)
        self.transform = transform
        self.max_frames_per_video = max_frames_per_video
        self.face_size = face_size
        
        self.video_paths = []
        self.labels = []
        
        # Load real videos (label = 0)
        real_videos = list(self.real_videos_path.glob("*.mp4"))
        logger.info(f"Found {len(real_videos)} real videos")
        for video_path in real_videos:
            self.video_paths.append(video_path)
            self.labels.append(0)  # Real = 0
        
        # Load fake videos (label = 1)
        fake_videos = list(self.fake_videos_path.glob("*.mp4"))
        logger.info(f"Found {len(fake_videos)} fake videos")
        for video_path in fake_videos:
            self.video_paths.append(video_path)
            self.labels.append(1)  # Fake = 1
        
        logger.info(f"Dataset: {len(real_videos)} real, {len(fake_videos)} fake videos")
        
        # Pre-extract faces for faster training
        self._preextract_faces()
    
    def _preextract_faces(self):
        """Pre-extract faces from all videos"""
        logger.info("Pre-extracting faces from videos...")
        self.face_data = []
        
        for i, (video_path, label) in enumerate(tqdm(zip(self.video_paths, self.labels), 
                                                     total=len(self.video_paths), 
                                                     desc="Extracting faces")):
            faces = self._extract_faces_from_video(video_path)
            
            # If no faces found, skip this video
            if not faces:
                logger.warning(f"No faces found in {video_path}")
                continue
            
            # Add all faces with the same label
            for face in faces:
                self.face_data.append((face, label))
        
        logger.info(f"Extracted {len(self.face_data)} face samples")
        
        # Check class balance
        real_faces = sum(1 for _, label in self.face_data if label == 0)
        fake_faces = sum(1 for _, label in self.face_data if label == 1)
        logger.info(f"Face distribution: {real_faces} real, {fake_faces} fake")
    
    def _extract_faces_from_video(self, video_path):
        """Extract faces from a single video with improved detection"""
        faces = []
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.warning(f"Cannot open video: {video_path}")
            return faces
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Sample frames more intelligently
        if frame_count > self.max_frames_per_video:
            # Sample frames from beginning, middle, and end
            start_frames = list(range(0, min(frame_count//3, self.max_frames_per_video//3)))
            middle_frames = list(range(frame_count//3, min(2*frame_count//3, frame_count//3 + self.max_frames_per_video//3)))
            end_frames = list(range(2*frame_count//3, min(frame_count, 2*frame_count//3 + self.max_frames_per_video//3)))
            frame_indices = start_frames + middle_frames + end_frames
        else:
            frame_indices = list(range(0, frame_count, max(1, frame_count // self.max_frames_per_video)))
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                continue
            
            # Extract face with better preprocessing
            face = self._extract_face_from_frame(frame)
            if face is not None:
                faces.append(face)
                
                # Stop if we have enough faces
                if len(faces) >= self.max_frames_per_video:
                    break
        
        cap.release()
        return faces
    
    def _extract_face_from_frame(self, frame):
        """Extract and preprocess face from frame with improved quality"""
        try:
            if frame.shape[0] < 64 or frame.shape[1] < 64:
                return None
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Multiple detection attempts for better robustness
            face_locations = []
            
            # Try HOG first (faster)
            try:
                face_locations = face_recognition.face_locations(frame_rgb, model='hog')
            except:
                pass
            
            # If HOG fails, try CNN (more accurate but slower)
            if not face_locations:
                try:
                    # Resize for CNN detection
                    height, width = frame_rgb.shape[:2]
                    if height > 600 or width > 600:
                        scale = min(600/height, 600/width)
                        new_height, new_width = int(height * scale), int(width * scale)
                        frame_small = cv2.resize(frame_rgb, (new_width, new_height))
                        face_locations = face_recognition.face_locations(frame_small, model='cnn')
                        # Scale back face locations
                        face_locations = [(int(top/scale), int(right/scale), int(bottom/scale), int(left/scale)) 
                                        for top, right, bottom, left in face_locations]
                    else:
                        face_locations = face_recognition.face_locations(frame_rgb, model='cnn')
                except:
                    pass
            
            if not face_locations:
                return None
            
            # Get the largest face
            largest_face = max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
            top, right, bottom, left = largest_face
            
            # Validate face region
            if bottom <= top or right <= left:
                return None
            
            # Add padding with better logic
            h, w = frame_rgb.shape[:2]
            face_height = bottom - top
            face_width = right - left
            
            # Dynamic padding based on face size
            pad_h = max(20, face_height // 6)
            pad_w = max(20, face_width // 6)
            
            top = max(0, top - pad_h)
            bottom = min(h, bottom + pad_h)
            left = max(0, left - pad_w)
            right = min(w, right + pad_w)
            
            # Extract face
            face_image = frame_rgb[top:bottom, left:right]
            
            # Validate extracted face
            if face_image.shape[0] < 64 or face_image.shape[1] < 64:
                return None
            
            # Resize to standard size with better interpolation
            face_image = cv2.resize(face_image, (self.face_size, self.face_size), interpolation=cv2.INTER_LANCZOS4)
            
            # Quality check - ensure image is not too dark or too bright
            mean_brightness = np.mean(face_image)
            if mean_brightness < 30 or mean_brightness > 240:
                return None
            
            return face_image
            
        except Exception as e:
            logger.debug(f"Face extraction error: {e}")
            return None
    
    def __len__(self):
        return len(self.face_data)
    
    def __getitem__(self, idx):
        face_image, label = self.face_data[idx]
        
        if self.transform:
            face_image = self.transform(face_image)
        
        return face_image, label

def get_transforms():
    """Get data augmentation transforms"""
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
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

def train_model():
    """Train the improved XceptionNet model"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"simple_xception_training_{timestamp}.log"
    
    # Setup file logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("Starting Simple XceptionNet Training with Transfer Learning")
    
    # Configuration
    config = {
        'epochs': 25,
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'face_size': 224,
        'max_frames_per_video': 25,
        'patience': 8,
        'min_delta': 0.01
    }
    
    logger.info(f"Training configuration: {config}")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Dataset paths
    real_videos_path = "dataset/original_sequences/youtube/c23/videos"
    fake_videos_path = "dataset/manipulated_sequences/Deepfakes/c23/videos"
    
    # Verify paths exist
    if not Path(real_videos_path).exists():
        logger.error(f"Real videos path not found: {real_videos_path}")
        return
    
    if not Path(fake_videos_path).exists():
        logger.error(f"Fake videos path not found: {fake_videos_path}")
        return
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create full dataset
    logger.info("Loading full dataset...")
    full_dataset = DeepfakeDataset(
        real_videos_path=real_videos_path,
        fake_videos_path=fake_videos_path,
        transform=train_transform,
        max_frames_per_video=config['max_frames_per_video'],
        face_size=config['face_size']
    )
    
    if len(full_dataset) == 0:
        logger.error("No faces extracted from dataset!")
        return
    
    # Split dataset
    train_indices, val_indices = train_test_split(
        list(range(len(full_dataset))), 
        test_size=0.2, 
        random_state=42, 
        stratify=[label for _, label in full_dataset.face_data]
    )
    
    # Create training and validation datasets
    train_data = [(full_dataset.face_data[i][0], full_dataset.face_data[i][1]) for i in train_indices]
    val_data = [(full_dataset.face_data[i][0], full_dataset.face_data[i][1]) for i in val_indices]
    
    # Create dataset classes for train and validation
    class SubsetDataset(Dataset):
        def __init__(self, data, transform):
            self.data = data
            self.transform = transform
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            face_image, label = self.data[idx]
            if self.transform:
                face_image = self.transform(face_image)
            return face_image, label
    
    train_dataset = SubsetDataset(train_data, train_transform)
    val_dataset = SubsetDataset(val_data, val_transform)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Model
    model = SimpleXceptionNet(num_classes=2, pretrained=True).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, verbose=True)
    
    # Training tracking
    best_val_accuracy = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    logger.info("Starting training loop...")
    
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate precision, recall, F1
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted', zero_division=0)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"""
        Epoch {epoch+1}/{config['epochs']} - Time: {epoch_time:.2f}s
        Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%
        Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%
        Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}
        Learning Rate: {current_lr:.6f}
        """)
        
        # Learning rate scheduling
        scheduler.step(val_accuracy)
        
        # Save best model
        if val_accuracy > best_val_accuracy + config['min_delta']:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            
            # Save best model
            os.makedirs("models", exist_ok=True)
            best_model_path = f"models/simple_xception_best_{timestamp}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_val_accuracy,
                'config': config,
                'timestamp': timestamp
            }, best_model_path)
            logger.info(f"New best model saved: {best_model_path} (Val Acc: {val_accuracy:.2f}%)")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= config['patience']:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    
    # Save final model
    final_model_path = f"models/simple_xception_final_{timestamp}.pth"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_val_accuracy,
        'config': config,
        'timestamp': timestamp,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }, final_model_path)
    logger.info(f"Final model saved: {final_model_path}")
    
    # Generate training plots
    try:
        plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies, timestamp)
    except Exception as e:
        logger.error(f"Failed to generate plots: {e}")
    
    logger.info(f"Training completed successfully! Best model saved at: {best_model_path}")
    logger.info(f"You can now use this model for better deepfake detection.")

def plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies, timestamp):
    """Generate training result plots"""
    plt.figure(figsize=(15, 10))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(2, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(val_accuracies, label='Validation Accuracy', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Best accuracy indicator
    best_epoch = np.argmax(val_accuracies)
    best_acc = val_accuracies[best_epoch]
    plt.axhline(y=best_acc, color='green', linestyle='--', alpha=0.7, label=f'Best Val Acc: {best_acc:.2f}%')
    plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7)
    
    # Summary statistics
    plt.subplot(2, 2, 3)
    plt.text(0.1, 0.8, f"Best Validation Accuracy: {best_acc:.2f}%", fontsize=14, weight='bold')
    plt.text(0.1, 0.7, f"Best Epoch: {best_epoch + 1}", fontsize=12)
    plt.text(0.1, 0.6, f"Final Training Accuracy: {train_accuracies[-1]:.2f}%", fontsize=12)
    plt.text(0.1, 0.5, f"Final Validation Accuracy: {val_accuracies[-1]:.2f}%", fontsize=12)
    plt.text(0.1, 0.4, f"Total Epochs: {len(train_losses)}", fontsize=12)
    plt.axis('off')
    plt.title('Training Summary')
    
    # Learning curve
    plt.subplot(2, 2, 4)
    improvement = np.array(val_accuracies) - val_accuracies[0]
    plt.plot(improvement, color='purple', linewidth=2)
    plt.title('Validation Accuracy Improvement')
    plt.xlabel('Epoch')
    plt.ylabel('Improvement (%)')
    plt.grid(True)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'simple_xception_training_results_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training plots saved: simple_xception_training_results_{timestamp}.png")

if __name__ == "__main__":
    train_model()
