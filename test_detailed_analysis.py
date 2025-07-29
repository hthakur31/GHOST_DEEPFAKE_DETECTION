#!/usr/bin/env python3
"""
Test Enhanced XceptionNet Detailed Analysis
Demonstrates the comprehensive video analysis features
"""

import json
from enhanced_xception_predictor import get_xception_predictor
from pathlib import Path

def format_analysis_results(results):
    """Format analysis results in a user-friendly way"""
    if not results.get('success', False):
        return f"❌ Error: {results.get('error', 'Unknown error')}"
    
    output = []
    output.append("="*80)
    output.append("🎥 ENHANCED DEEPFAKE DETECTION ANALYSIS")
    output.append("="*80)
    
    # Main Results Section
    output.append(f"\n🔍 DETECTION RESULTS")
    output.append(f"Prediction: {results['prediction_display']}")
    output.append(f"Overall Confidence: {results['confidence_percent']}")
    output.append(f"Confidence Level: {results['confidence_level']}")
    output.append(f"Authentic Probability: {results['authentic_probability']}")
    output.append(f"Deepfake Probability: {results['deepfake_probability']}")
    
    # Video Analysis Details
    va = results['video_analysis']
    output.append(f"\n📊 VIDEO ANALYSIS DETAILS")
    output.append(f"File Name: {va['file_name']}")
    output.append(f"File Size: {va['file_size']}")
    output.append(f"Duration: {va['duration']}")
    output.append(f"Frame Rate: {va['frame_rate']}")
    output.append(f"Resolution: {va['resolution']}")
    output.append(f"Total Frames: {va['total_frames']}")
    output.append(f"Faces Detected: {va['faces_detected']}")
    output.append(f"Frames Analyzed: {va['frames_analyzed']}")
    output.append(f"Processing Time: {va['processing_time']}")
    output.append(f"Analysis Date: {va['analysis_date']}")
    
    # Detection Insights
    insights = results['insights']
    output.append(f"\n💡 DETECTION INSIGHTS")
    output.append(f"Summary: {insights['summary']}")
    output.append(f"Confidence: {insights['confidence_description']}")
    
    # AI Model Information
    model = results['model_info']
    output.append(f"\n🤖 AI MODEL INFORMATION")
    output.append(f"Model: {model['model_name']}")
    output.append(f"Version: {model['version']}")
    output.append(f"Method: {model['method']}")
    output.append(f"Architecture: {model['architecture']}")
    output.append(f"Training Data: {model['training_data']}")
    
    # Frame-by-Frame Analysis
    output.append(f"\n📋 FRAME-BY-FRAME ANALYSIS")
    output.append(f"{'Frame #':<8} {'Timestamp':<10} {'Prediction':<10} {'Confidence':<12} {'Real %':<8} {'Fake %':<8} {'Face':<6}")
    output.append("-" * 70)
    
    for frame in results['frame_analysis']:
        output.append(f"{frame['frame_number']:<8} "
                     f"{frame['timestamp']:<10} "
                     f"{frame['prediction']:<10} "
                     f"{frame['confidence']:<12.3f} "
                     f"{frame['real_percent']:<8} "
                     f"{frame['fake_percent']:<8} "
                     f"{frame['face_detected']:<6}")
    
    output.append("="*80)
    
    return "\n".join(output)

def test_detailed_analysis():
    """Test the enhanced detailed analysis"""
    print("🧪 Testing Enhanced XceptionNet Detailed Analysis")
    print("="*60)
    
    predictor = get_xception_predictor()
    
    if predictor.model is None:
        print("❌ No XceptionNet model loaded for testing")
        return
    
    print(f"✅ Model loaded: {predictor.model_type} architecture")
    
    # Look for test videos in dataset or create a demo
    test_video_paths = [
        "dataset/original_sequences/youtube/c23/videos",
        "dataset/manipulated_sequences/Deepfakes/c23/videos",
        "test_videos"
    ]
    
    found_video = None
    for path in test_video_paths:
        video_dir = Path(path)
        if video_dir.exists():
            for video_file in video_dir.glob("*.mp4"):
                found_video = video_file
                break
        if found_video:
            break
    
    if found_video:
        print(f"🎬 Testing with video: {found_video.name}")
        print("🔄 Running detailed analysis...")
        
        # Run the enhanced analysis
        results = predictor.predict_video(str(found_video), max_frames=15)
        
        # Display formatted results
        print("\n" + format_analysis_results(results))
        
        # Save detailed results to JSON for reference
        output_file = f"detailed_analysis_{found_video.stem}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        
    else:
        print("❌ No test videos found. Please ensure you have videos in:")
        for path in test_video_paths:
            print(f"   📁 {path}")
        
        # Create a sample result structure for demonstration
        print("\n📋 Sample Enhanced Analysis Structure:")
        sample_structure = {
            "prediction_display": "Real Video",
            "confidence_percent": "51.48%",
            "confidence_level": "Medium",
            "authentic_probability": "48.5%",
            "deepfake_probability": "51.5%",
            "video_analysis": {
                "file_size": "1.8 MB",
                "duration": "32.36 seconds",
                "frame_rate": "25.0 FPS",
                "resolution": "854x480",
                "faces_detected": "13 faces",
                "frames_analyzed": 13,
                "processing_time": "28.24 seconds"
            },
            "frame_analysis": [
                {
                    "frame_number": 0,
                    "timestamp": "0.00s",
                    "prediction": "Fake",
                    "confidence": 0.514,
                    "real_percent": "48.6%",
                    "fake_percent": "51.4%",
                    "face_detected": "✓"
                }
            ]
        }
        
        print(json.dumps(sample_structure, indent=2))

if __name__ == "__main__":
    test_detailed_analysis()
