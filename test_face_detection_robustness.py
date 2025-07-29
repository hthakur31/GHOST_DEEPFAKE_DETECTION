#!/usr/bin/env python3
"""
Test script to analyze face detection robustness and identify failure patterns.
This will help us understand why videos are failing with "no faces detected".
"""

import cv2
import face_recognition
import numpy as np
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_face_detection_methods(frame):
    """Test different face detection methods on a frame"""
    results = {}
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Method 1: Original HOG method
    try:
        start_time = time.time()
        faces_hog = face_recognition.face_locations(frame_rgb, model='hog')
        hog_time = time.time() - start_time
        results['hog'] = {
            'faces': len(faces_hog),
            'time': hog_time,
            'locations': faces_hog
        }
    except Exception as e:
        results['hog'] = {'error': str(e)}
    
    # Method 2: CNN method (more accurate but slower)
    try:
        start_time = time.time()
        faces_cnn = face_recognition.face_locations(frame_rgb, model='cnn')
        cnn_time = time.time() - start_time
        results['cnn'] = {
            'faces': len(faces_cnn),
            'time': cnn_time,
            'locations': faces_cnn
        }
    except Exception as e:
        results['cnn'] = {'error': str(e)}
    
    # Method 3: OpenCV Haar Cascades (as backup)
    try:
        start_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces_haar = face_cascade.detectMultiScale(gray, 1.1, 4)
        haar_time = time.time() - start_time
        
        # Convert to face_recognition format (top, right, bottom, left)
        haar_locations = []
        for (x, y, w, h) in faces_haar:
            haar_locations.append((y, x + w, y + h, x))
        
        results['haar'] = {
            'faces': len(haar_locations),
            'time': haar_time,
            'locations': haar_locations
        }
    except Exception as e:
        results['haar'] = {'error': str(e)}
    
    return results

def enhanced_extract_face_from_frame(frame):
    """Enhanced face extraction with multiple fallback methods"""
    try:
        if frame.shape[0] < 50 or frame.shape[1] < 50:
            return None, "Frame too small"
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Try multiple detection methods in order of preference
        methods = ['hog', 'cnn', 'haar']
        face_locations = []
        method_used = None
        
        for method in methods:
            try:
                if method == 'hog':
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
                
                elif method == 'cnn':
                    # Try CNN method (more accurate but slower)
                    face_locations = face_recognition.face_locations(frame_rgb, model='cnn')
                
                elif method == 'haar':
                    # OpenCV Haar Cascades as last resort
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces_haar = face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    # Convert to face_recognition format
                    face_locations = []
                    for (x, y, w, h) in faces_haar:
                        face_locations.append((y, x + w, y + h, x))
                
                if face_locations:
                    method_used = method
                    break
                    
            except Exception as e:
                logger.debug(f"Method {method} failed: {e}")
                continue
        
        if not face_locations:
            return None, "No faces detected with any method"
        
        # Get the largest face
        largest_face = max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
        top, right, bottom, left = largest_face
        
        # Validate face region
        if bottom <= top or right <= left:
            return None, f"Invalid face region using {method_used}"
        
        # Add padding
        h, w = frame_rgb.shape[:2]
        pad = max(20, (bottom - top) // 6)
        top = max(0, top - pad)
        bottom = min(h, bottom + pad)
        left = max(0, left - pad)
        right = min(w, right + pad)
        
        # Extract and resize face
        face_image = frame_rgb[top:bottom, left:right]
        
        if face_image.shape[0] < 64 or face_image.shape[1] < 64:
            return None, f"Extracted face too small ({face_image.shape}) using {method_used}"
        
        face_image = cv2.resize(face_image, (224, 224))
        return face_image, f"Success using {method_used}"
        
    except Exception as e:
        return None, f"Face extraction error: {e}"

def test_video_face_detection(video_path, max_frames=10):
    """Test face detection on a video file"""
    print(f"\nTesting video: {video_path}")
    
    if not Path(video_path).exists():
        print(f"Video file not found: {video_path}")
        return
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height}, {frame_count} frames, {fps:.1f} FPS")
    
    # Sample frames
    if frame_count > max_frames:
        frame_indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)
    else:
        frame_indices = list(range(0, frame_count, max(1, frame_count // max_frames)))
    
    results = {
        'total_frames_tested': 0,
        'faces_detected_original': 0,
        'faces_detected_enhanced': 0,
        'method_success_count': {'hog': 0, 'cnn': 0, 'haar': 0},
        'errors': []
    }
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret or frame is None:
            continue
        
        results['total_frames_tested'] += 1
        
        # Test original method
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            face_locations_orig = face_recognition.face_locations(frame_rgb, model='hog')
            if face_locations_orig:
                results['faces_detected_original'] += 1
        except Exception as e:
            results['errors'].append(f"Frame {frame_idx} original method: {e}")
        
        # Test enhanced method
        face_image, status = enhanced_extract_face_from_frame(frame)
        if face_image is not None:
            results['faces_detected_enhanced'] += 1
            # Extract method from status
            if 'hog' in status:
                results['method_success_count']['hog'] += 1
            elif 'cnn' in status:
                results['method_success_count']['cnn'] += 1
            elif 'haar' in status:
                results['method_success_count']['haar'] += 1
        else:
            results['errors'].append(f"Frame {frame_idx} enhanced method: {status}")
    
    cap.release()
    
    print("\nResults:")
    print(f"  Frames tested: {results['total_frames_tested']}")
    print(f"  Original method success: {results['faces_detected_original']}/{results['total_frames_tested']}")
    print(f"  Enhanced method success: {results['faces_detected_enhanced']}/{results['total_frames_tested']}")
    print(f"  Method breakdown: {results['method_success_count']}")
    
    if results['errors']:
        print(f"  Errors: {len(results['errors'])}")
        for error in results['errors'][:3]:  # Show first 3 errors
            print(f"    {error}")
        if len(results['errors']) > 3:
            print(f"    ... and {len(results['errors']) - 3} more")
    
    return results

def main():
    """Test face detection robustness"""
    print("=== Face Detection Robustness Test ===")
    
    # Look for video files in common locations
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    search_paths = [
        Path('.'),
        Path('./media'),
        Path('./test_videos'),
        Path('./sample_videos')
    ]
    
    video_files = []
    for search_path in search_paths:
        if search_path.exists():
            for ext in video_extensions:
                video_files.extend(search_path.glob(f'*{ext}'))
                video_files.extend(search_path.glob(f'**/*{ext}'))
    
    if not video_files:
        print("No video files found for testing.")
        print("Please place some video files in the current directory or media/ folder.")
        return
    
    # Test up to 5 videos
    test_files = video_files[:5]
    print(f"Found {len(video_files)} video files, testing first {len(test_files)}")
    
    all_results = []
    for video_file in test_files:
        try:
            result = test_video_face_detection(video_file)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"Error testing {video_file}: {e}")
    
    # Summary
    if all_results:
        print("\n=== SUMMARY ===")
        total_frames = sum(r['total_frames_tested'] for r in all_results)
        total_original_success = sum(r['faces_detected_original'] for r in all_results)
        total_enhanced_success = sum(r['faces_detected_enhanced'] for r in all_results)
        
        print(f"Total frames tested: {total_frames}")
        print(f"Original method success rate: {total_original_success}/{total_frames} ({100*total_original_success/total_frames:.1f}%)")
        print(f"Enhanced method success rate: {total_enhanced_success}/{total_frames} ({100*total_enhanced_success/total_frames:.1f}%)")
        
        method_totals = {'hog': 0, 'cnn': 0, 'haar': 0}
        for result in all_results:
            for method, count in result['method_success_count'].items():
                method_totals[method] += count
        
        print(f"Method breakdown:")
        for method, count in method_totals.items():
            print(f"  {method.upper()}: {count} successes")

if __name__ == "__main__":
    main()
