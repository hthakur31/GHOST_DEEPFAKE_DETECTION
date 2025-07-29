#!/usr/bin/env python
"""
Template validation script to check for syntax errors and issues
"""
import os
import sys
import django
from django.conf import settings
from django.template import Template, Context, TemplateSyntaxError
from django.template.loader import get_template

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_detector.settings')
django.setup()

def test_template_syntax():
    """Test template syntax for all detector templates"""
    template_dir = 'detector/templates/detector'
    templates = ['base.html', 'home.html', 'result.html']
    
    print("🔍 Testing Django Template Syntax...")
    
    for template_name in templates:
        try:
            print(f"  Testing {template_name}...", end=' ')
            template = get_template(f'detector/{template_name}')
            print("✅ PASS")
        except TemplateSyntaxError as e:
            print(f"❌ SYNTAX ERROR: {e}")
            return False
        except Exception as e:
            print(f"❌ ERROR: {e}")
            return False
    
    return True

def test_template_rendering():
    """Test template rendering with mock context"""
    print("\n🎨 Testing Template Rendering...")
    
    # Mock context data
    mock_context = {
        'stats': {
            'total_predictions': 100,
            'real_predictions': 60,
            'fake_predictions': 40,
            'fake_percentage': 40.0
        },
        'model_performance': {
            'model_name': 'ResNet50',
            'model_version': '1.0',
            'total_predictions': 100,
            'accuracy': 85.5
        },
        'recent_results': [
            {
                'id': '123e4567-e89b-12d3-a456-426614174000',
                'original_filename': 'test_video.mp4',
                'prediction': 'REAL',
                'confidence_score': 0.95,
                'confidence_percentage': 95.0,
                'processing_time': 2.5,
                'created_at': '2025-07-27 19:00:00'
            }
        ],
        'result': {
            'id': '123e4567-e89b-12d3-a456-426614174000',
            'original_filename': 'test_video.mp4',
            'prediction': 'REAL',
            'confidence_score': 0.95,
            'confidence_percentage': 95.0,
            'real_probability': 95.0,
            'fake_probability': 5.0,
            'file_size': 10485760,
            'video_duration': 30.0,
            'video_fps': 30.0,
            'video_resolution': '1920x1080',
            'face_detected': True,
            'face_count': 1,
            'frames_analyzed': 90,
            'processing_time': 2.5,
            'model_used': 'ResNet50',
            'model_version': '1.0',
            'detection_method': 'Deep Learning',
            'has_error': False,
            'error_message': None
        },
        'frame_analyses': [
            {
                'frame_number': 1,
                'timestamp': 0.033,
                'prediction': 'REAL',
                'confidence_score': 0.95,
                'real_probability': 95.0,
                'fake_probability': 5.0,
                'face_detected': True
            }
        ],
        'chart_data': {
            'frame_numbers': [1, 2, 3],
            'confidence_scores': [0.95, 0.94, 0.96],
            'real_probabilities': [0.95, 0.94, 0.96],
            'fake_probabilities': [0.05, 0.06, 0.04]
        }
    }
    
    try:
        print("  Testing home.html rendering...", end=' ')
        template = get_template('detector/home.html')
        rendered = template.render(mock_context)
        print("✅ PASS")
        
        print("  Testing result.html rendering...", end=' ')
        template = get_template('detector/result.html')
        rendered = template.render(mock_context)
        print("✅ PASS")
        
        return True
    except Exception as e:
        print(f"❌ RENDER ERROR: {e}")
        return False

def validate_html_structure():
    """Basic HTML structure validation"""
    print("\n📋 Validating HTML Structure...")
    
    issues = []
    
    # Read template files directly
    template_files = [
        'detector/templates/detector/home.html',
        'detector/templates/detector/result.html', 
        'detector/templates/detector/base.html'
    ]
    
    for template_path in template_files:
        template_name = os.path.basename(template_path)
        print(f"  Checking {template_name}...", end=' ')
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for unclosed tags (basic check)
            if content.count('<div') != content.count('</div>'):
                issues.append(f"{template_name}: Mismatched div tags")
            
            # Check for required meta tags in base template
            if 'base.html' in template_name:
                if 'charset="UTF-8"' not in content:
                    issues.append(f"{template_name}: Missing charset declaration")
                if 'viewport' not in content:
                    issues.append(f"{template_name}: Missing viewport meta tag")
            
            print("✅ PASS")
            
        except FileNotFoundError:
            issues.append(f"{template_name}: File not found")
            print("❌ FAIL")
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    
    return True

def main():
    """Run all template validation tests"""
    print("🚀 Django Template Validation Suite")
    print("=" * 50)
    
    all_passed = True
    
    # Test syntax
    if not test_template_syntax():
        all_passed = False
    
    # Test rendering
    if not test_template_rendering():
        all_passed = False
    
    # Validate structure
    if not validate_html_structure():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All template tests PASSED! Templates are error-free.")
    else:
        print("❌ Some template tests FAILED. Please fix the issues above.")
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
