#!/usr/bin/env python3
"""
Enhanced face detection robustness fix for the deepfake detection app.
This improves the face detection to handle edge cases and reduces "no faces detected" errors.
"""

import cv2
import face_recognition
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def enhanced_extract_face_from_frame(frame):
    """
    Enhanced face extraction with multiple fallback methods and better error handling.
    This replaces the original extract_face_from_frame method in enhanced_xception_predictor.py
    """
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

def apply_enhanced_face_detection():
    """Apply the enhanced face detection to the predictor"""
    
    predictor_file = Path("enhanced_xception_predictor.py")
    
    if not predictor_file.exists():
        print("❌ enhanced_xception_predictor.py not found")
        return False
    
    # Read the current file
    with open(predictor_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the extract_face_from_frame method and replace it
    import re
    
    # Pattern to match the entire extract_face_from_frame method
    pattern = r'(\s*)def extract_face_from_frame\(self, frame\):.*?(?=\n\s*def|\n\s*class|\nclass|\Z)'
    
    # Create the replacement method
    replacement = '''    def extract_face_from_frame(self, frame):
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
            return None'''
    
    # Use re.DOTALL to match newlines
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        # Replace the method
        new_content = content[:match.start()] + replacement + content[match.end():]
        
        # Write back to file
        with open(predictor_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ Enhanced face detection applied successfully!")
        return True
    else:
        print("❌ Could not find extract_face_from_frame method to replace")
        return False

if __name__ == "__main__":
    print("=== Applying Enhanced Face Detection ===")
    
    success = apply_enhanced_face_detection()
    
    if success:
        print("\n🎯 Enhanced face detection features:")
        print("  • Multiple fallback detection methods (HOG → CNN → Haar)")
        print("  • Adaptive upsampling for better small face detection")
        print("  • Multiple Haar cascade classifiers")
        print("  • Adaptive padding based on face size")
        print("  • Better face size validation")
        print("  • Improved error handling")
        
        print("\n📈 This should significantly reduce 'no faces detected' errors!")
        print("💡 Test the system with various video types to verify the improvements.")
    else:
        print("\n❌ Failed to apply enhancements. Please check the file manually.")
