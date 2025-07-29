from rest_framework import serializers
from .models import DetectionResult, FrameAnalysis, ModelPerformance
from django.core.validators import FileExtensionValidator


class VideoUploadSerializer(serializers.Serializer):
    """Serializer for video file uploads"""
    
    video_file = serializers.FileField(
        validators=[FileExtensionValidator(allowed_extensions=['mp4', 'avi', 'mov', 'mkv', 'webm'])],
        help_text="Upload video file (max 100MB). Supported formats: MP4, AVI, MOV, MKV, WEBM"
    )
    
    def validate_video_file(self, value):
        """Validate video file size and format"""
        # Check file size (100MB limit)
        if value.size > 100 * 1024 * 1024:
            raise serializers.ValidationError("File size cannot exceed 100MB")
        
        # Check file format by reading first few bytes
        magic_numbers = {
            b'\x00\x00\x00\x1c\x66\x74\x79\x70': 'mp4',
            b'\x52\x49\x46\x46': 'avi',
            b'\x00\x00\x00\x14\x66\x74\x79\x70\x71\x74': 'mov',
        }
        
        # Read first 20 bytes to check magic numbers
        value.seek(0)
        header = value.read(20)
        value.seek(0)  # Reset file pointer
        
        # Basic validation - in production, use more robust video validation
        if len(header) < 8:
            raise serializers.ValidationError("Invalid video file format")
        
        return value


class FrameAnalysisSerializer(serializers.ModelSerializer):
    """Serializer for frame analysis results"""
    
    confidence_percentage = serializers.SerializerMethodField()
    
    class Meta:
        model = FrameAnalysis
        fields = [
            'frame_number', 'timestamp', 'prediction', 'confidence_score',
            'confidence_percentage', 'fake_probability', 'real_probability',
            'face_detected', 'face_bbox', 'face_landmarks',
            'temporal_consistency', 'spatial_consistency', 'compression_artifacts'
        ]
    
    def get_confidence_percentage(self, obj):
        """Convert confidence score to percentage"""
        return round(obj.confidence_score * 100, 2) if obj.confidence_score else None


class DetectionResultSerializer(serializers.ModelSerializer):
    """Serializer for detection results"""
    
    confidence_percentage = serializers.SerializerMethodField()
    fake_probability_percentage = serializers.SerializerMethodField()
    real_probability_percentage = serializers.SerializerMethodField()
    frame_analyses = FrameAnalysisSerializer(many=True, read_only=True)
    status_display = serializers.SerializerMethodField()
    file_size_mb = serializers.SerializerMethodField()
    
    class Meta:
        model = DetectionResult
        fields = [
            'id', 'original_filename', 'file_size', 'file_size_mb',
            'prediction', 'status_display', 'confidence_score', 'confidence_percentage',
            'fake_probability', 'fake_probability_percentage',
            'real_probability', 'real_probability_percentage',
            'model_used', 'model_version', 'processing_time',
            'video_duration', 'video_fps', 'video_resolution', 'frames_analyzed',
            'face_detected', 'face_count', 'detection_method',
            'error_message', 'created_at', 'updated_at', 'processed_at',
            'frame_analyses'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def get_confidence_percentage(self, obj):
        """Convert confidence score to percentage"""
        return round(obj.confidence_score * 100, 2) if obj.confidence_score else None
    
    def get_fake_probability_percentage(self, obj):
        """Convert fake probability to percentage"""
        return round(obj.fake_probability * 100, 2) if obj.fake_probability else None
    
    def get_real_probability_percentage(self, obj):
        """Convert real probability to percentage"""
        return round(obj.real_probability * 100, 2) if obj.real_probability else None
    
    def get_status_display(self, obj):
        """Get human-readable status"""
        status_map = {
            'PROCESSING': 'Processing...',
            'REAL': 'Real Video',
            'FAKE': 'Deepfake Detected',
            'ERROR': 'Processing Error'
        }
        return status_map.get(obj.prediction, obj.prediction)
    
    def get_file_size_mb(self, obj):
        """Convert file size to MB"""
        return round(obj.file_size / (1024 * 1024), 2) if obj.file_size else None


class ModelPerformanceSerializer(serializers.ModelSerializer):
    """Serializer for model performance metrics"""
    
    accuracy_percentage = serializers.SerializerMethodField()
    precision_percentage = serializers.SerializerMethodField()
    recall_percentage = serializers.SerializerMethodField()
    f1_score_percentage = serializers.SerializerMethodField()
    
    class Meta:
        model = ModelPerformance
        fields = [
            'model_name', 'model_version', 'total_predictions',
            'correct_predictions', 'false_positives', 'false_negatives',
            'true_positives', 'true_negatives',
            'accuracy', 'accuracy_percentage',
            'precision', 'precision_percentage',
            'recall', 'recall_percentage',
            'f1_score', 'f1_score_percentage',
            'avg_processing_time', 'avg_confidence_score',
            'last_updated', 'created_at'
        ]
        read_only_fields = ['created_at', 'last_updated']
    
    def get_accuracy_percentage(self, obj):
        """Convert accuracy to percentage"""
        return round(obj.accuracy * 100, 2) if obj.accuracy else None
    
    def get_precision_percentage(self, obj):
        """Convert precision to percentage"""
        return round(obj.precision * 100, 2) if obj.precision else None
    
    def get_recall_percentage(self, obj):
        """Convert recall to percentage"""
        return round(obj.recall * 100, 2) if obj.recall else None
    
    def get_f1_score_percentage(self, obj):
        """Convert F1 score to percentage"""
        return round(obj.f1_score * 100, 2) if obj.f1_score else None


class DetectionSummarySerializer(serializers.Serializer):
    """Serializer for detection summary statistics"""
    
    total_detections = serializers.IntegerField()
    real_count = serializers.IntegerField()
    fake_count = serializers.IntegerField()
    processing_count = serializers.IntegerField()
    error_count = serializers.IntegerField()
    
    real_percentage = serializers.FloatField()
    fake_percentage = serializers.FloatField()
    
    avg_confidence_score = serializers.FloatField()
    avg_processing_time = serializers.FloatField()
    
    most_recent_detection = DetectionResultSerializer(read_only=True)
    
    def to_representation(self, instance):
        """Calculate summary statistics"""
        from django.db.models import Avg, Count
        from .models import DetectionResult
        
        # Get counts
        total = DetectionResult.objects.count()
        real_count = DetectionResult.objects.filter(prediction='REAL').count()
        fake_count = DetectionResult.objects.filter(prediction='FAKE').count()
        processing_count = DetectionResult.objects.filter(prediction='PROCESSING').count()
        error_count = DetectionResult.objects.filter(prediction='ERROR').count()
        
        # Calculate percentages
        real_percentage = (real_count / total * 100) if total > 0 else 0
        fake_percentage = (fake_count / total * 100) if total > 0 else 0
        
        # Get averages
        completed_results = DetectionResult.objects.exclude(prediction__in=['PROCESSING', 'ERROR'])
        avg_confidence = completed_results.aggregate(Avg('confidence_score'))['confidence_score__avg'] or 0
        avg_processing = completed_results.aggregate(Avg('processing_time'))['processing_time__avg'] or 0
        
        # Get most recent detection
        recent_detection = DetectionResult.objects.order_by('-created_at').first()
        
        return {
            'total_detections': total,
            'real_count': real_count,
            'fake_count': fake_count,
            'processing_count': processing_count,
            'error_count': error_count,
            'real_percentage': round(real_percentage, 2),
            'fake_percentage': round(fake_percentage, 2),
            'avg_confidence_score': round(avg_confidence, 3) if avg_confidence else 0,
            'avg_processing_time': round(avg_processing, 2) if avg_processing else 0,
            'most_recent_detection': DetectionResultSerializer(recent_detection).data if recent_detection else None
        }
