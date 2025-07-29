# Copilot Instructions for Deepfake Detection Django Application

<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

## Project Overview
This is a Django web application for deepfake detection that uses machine learning models trained on the FaceForensics++ dataset. The application allows users to upload videos and receive confidence scores indicating whether the video contains deepfake content.

## Key Technologies
- **Django**: Web framework for the application
- **PyTorch/TensorFlow**: Machine learning frameworks for model training and inference
- **OpenCV**: Video processing and computer vision
- **Face Recognition**: Face detection and analysis
- **MediaPipe**: Real-time face landmark detection
- **FaceForensics++ Dataset**: Training data for deepfake detection

## Code Style Guidelines
- Follow Django best practices and PEP 8 style guidelines
- Use class-based views where appropriate
- Implement proper error handling for video uploads and processing
- Add comprehensive logging for model inference and debugging
- Use Django's built-in security features for file uploads
- Implement proper validation for uploaded video files

## Model Integration Guidelines
- Use Django's model system for storing detection results
- Implement asynchronous processing for video analysis
- Cache model predictions to improve performance
- Provide clear confidence scores and accuracy metrics
- Handle different video formats and resolutions gracefully

## Security Considerations
- Validate all uploaded files thoroughly
- Limit file sizes and types for security
- Sanitize user inputs and file names
- Use Django's CSRF protection
- Implement rate limiting for API endpoints

## Testing Guidelines
- Write unit tests for model inference functions
- Test video upload and processing workflows
- Validate confidence score calculations
- Test with various video formats and edge cases
- Include integration tests for the complete pipeline

## Performance Optimization
- Use appropriate video compression and preprocessing
- Implement model caching and optimization
- Consider GPU acceleration for inference when available
- Optimize database queries for result storage and retrieval
