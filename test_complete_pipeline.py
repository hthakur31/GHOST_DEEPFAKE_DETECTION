#!/usr/bin/env python3
"""
Test the complete prediction pipeline to identify why we're getting "no faces detected" errors.
This will simulate the exact workflow that the web app uses.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_xception_predictor import EnhancedXceptionNetPredictor
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_complete_pipeline():
    """Test the complete prediction pipeline"""
    print("=== Complete Pipeline Test ===")
    
    # Initialize predictor
    predictor = EnhancedXceptionNetPredictor()
    
    if predictor.model is None:
        print("❌ Failed to load model")
        return
    
    print("✅ Model loaded successfully")
    
    # Find video files to test
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    search_paths = [
        Path('.'),
        Path('./media'),
        Path('./test_videos'),
        Path('./sample_videos'),
        Path('./dataset')
    ]
    
    video_files = []
    for search_path in search_paths:
        if search_path.exists():
            for ext in video_extensions:
                video_files.extend(search_path.glob(f'*{ext}'))
                # Also check subdirectories for dataset structure
                video_files.extend(search_path.glob(f'**/*{ext}'))
    
    if not video_files:
        print("No video files found for testing.")
        return
    
    # Test a variety of videos
    test_files = video_files[:10]  # Test first 10 videos
    print(f"Testing {len(test_files)} videos...")
    
    success_count = 0
    failure_count = 0
    error_types = {}
    
    for i, video_file in enumerate(test_files):
        print(f"\n[{i+1}/{len(test_files)}] Testing: {video_file.name}")
        
        try:
            # Use the exact same method as the web app
            result = predictor.predict_video(str(video_file))
            
            if result.get("success", False):
                print(f"  ✅ Success: {result.get('prediction', 'Unknown')} "
                      f"({result.get('confidence', 0):.3f} confidence)")
                success_count += 1
                
                # Show some stats
                analysis = result.get('video_analysis', {})
                frames_analyzed = analysis.get('frames_analyzed', 0)
                faces_detected = analysis.get('faces_detected', 0)
                print(f"     Frames: {frames_analyzed}, Faces: {faces_detected}")
                
            else:
                error = result.get("error", "Unknown error")
                print(f"  ❌ Failed: {error}")
                failure_count += 1
                
                # Track error types
                if error not in error_types:
                    error_types[error] = 0
                error_types[error] += 1
                
                # Show video analysis if available
                analysis = result.get('video_analysis', {})
                if analysis:
                    resolution = analysis.get('resolution', 'Unknown')
                    duration = analysis.get('duration', 'Unknown')
                    frames = analysis.get('total_frames', 'Unknown')
                    print(f"     Video info: {resolution}, {duration}s, {frames} frames")
                
        except Exception as e:
            print(f"  💥 Exception: {e}")
            failure_count += 1
            error_type = f"Exception: {type(e).__name__}"
            if error_type not in error_types:
                error_types[error_type] = 0
            error_types[error_type] += 1
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Total videos tested: {len(test_files)}")
    print(f"Successful predictions: {success_count}")
    print(f"Failed predictions: {failure_count}")
    print(f"Success rate: {100*success_count/len(test_files):.1f}%")
    
    if error_types:
        print(f"\nError breakdown:")
        for error, count in error_types.items():
            print(f"  {error}: {count} occurrences")
    
    return success_count, failure_count, error_types

def test_specific_error_case():
    """Test with a potentially problematic video to reproduce the error"""
    print("\n=== Testing Specific Error Cases ===")
    
    predictor = EnhancedXceptionNetPredictor()
    if predictor.model is None:
        print("❌ Failed to load model")
        return
    
    # Test with a very short video or low quality video if available
    video_files = list(Path('.').glob('**/*.mp4'))
    
    if video_files:
        # Try to find a potentially problematic video
        test_video = video_files[0]  # Just test the first one
        print(f"Testing potential problem video: {test_video}")
        
        # Test with reduced max_frames to see if it makes a difference
        result = predictor.predict_video(str(test_video), max_frames=5)
        
        if not result.get("success", False):
            print(f"❌ Failed with fewer frames: {result.get('error', 'Unknown')}")
            
            # Try to debug the face extraction step by step
            print("Debug: Testing individual frames...")
            
            import cv2
            cap = cv2.VideoCapture(str(test_video))
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"Video has {frame_count} frames")
                
                # Test first few frames
                for i in range(min(5, frame_count)):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        face_image = predictor.extract_face_from_frame(frame)
                        print(f"  Frame {i}: {'✅ Face detected' if face_image is not None else '❌ No face'}")
                    else:
                        print(f"  Frame {i}: ❌ Could not read frame")
                
                cap.release()
        else:
            print(f"✅ Success with fewer frames: {result.get('prediction')}")

if __name__ == "__main__":
    test_complete_pipeline()
    test_specific_error_case()
