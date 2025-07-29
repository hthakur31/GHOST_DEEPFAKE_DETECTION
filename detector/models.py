from django.db import models
from django.core.validators import FileExtensionValidator
import uuid
import os


def video_upload_path(instance, filename):
    """Generate upload path for video files"""
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('uploads', 'videos', filename)


class DetectionResult(models.Model):
    """Model to store deepfake detection results"""
    
    DETECTION_CHOICES = [
        ('REAL', 'Real'),
        ('FAKE', 'Deepfake'),
        ('PROCESSING', 'Processing'),
        ('ERROR', 'Error'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    video_file = models.FileField(
        upload_to=video_upload_path,
        validators=[FileExtensionValidator(allowed_extensions=['mp4', 'avi', 'mov', 'mkv', 'webm'])],
        help_text="Upload video file (max 100MB)"
    )
    original_filename = models.CharField(max_length=255)
    file_size = models.BigIntegerField(help_text="File size in bytes")
    
    # Detection results
    prediction = models.CharField(max_length=20, choices=DETECTION_CHOICES, default='PROCESSING')
    confidence_score = models.FloatField(null=True, blank=True, help_text="Confidence score (0-1)")
    fake_probability = models.FloatField(null=True, blank=True, help_text="Probability of being fake (0-1)")
    real_probability = models.FloatField(null=True, blank=True, help_text="Probability of being real (0-1)")
    
    # Model metadata
    model_used = models.CharField(max_length=100, blank=True, help_text="Name of the detection model used")
    model_version = models.CharField(max_length=50, blank=True, help_text="Version of the detection model")
    processing_time = models.FloatField(null=True, blank=True, help_text="Processing time in seconds")
    
    # Video metadata
    video_duration = models.FloatField(null=True, blank=True, help_text="Video duration in seconds")
    video_fps = models.FloatField(null=True, blank=True, help_text="Video frames per second")
    video_resolution = models.CharField(max_length=20, blank=True, help_text="Video resolution (e.g., 1920x1080)")
    frames_analyzed = models.IntegerField(null=True, blank=True, help_text="Number of frames analyzed")
    
    # Detection details
    face_detected = models.BooleanField(default=False, help_text="Whether a face was detected")
    face_count = models.IntegerField(default=0, help_text="Number of faces detected")
    detection_method = models.CharField(max_length=100, blank=True, help_text="Detection method used")
    
    # Error handling
    error_message = models.TextField(blank=True, help_text="Error message if processing failed")
    
    # Enhanced metadata for reports
    metadata = models.JSONField(default=dict, blank=True, help_text="Additional detection metadata")
    temporal_consistency = models.FloatField(null=True, blank=True, help_text="Temporal consistency score")
    threshold_used = models.FloatField(null=True, blank=True, help_text="Detection threshold used")
    
    # Report generation
    report_generated = models.BooleanField(default=False, help_text="Whether detailed report was generated")
    report_data = models.JSONField(default=dict, blank=True, help_text="Comprehensive report data")
    
    # User tracking (optional for multi-user scenarios)
    user_ip = models.GenericIPAddressField(null=True, blank=True, help_text="User IP address")
    user_agent = models.TextField(blank=True, help_text="User agent string")
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Detection Result"
        verbose_name_plural = "Detection Results"
    
    def __str__(self):
        return f"{self.original_filename} - {self.prediction} ({self.confidence_score})"
    
    @property
    def is_processing(self):
        return self.prediction == 'PROCESSING'
    
    @property
    def has_error(self):
        return self.prediction == 'ERROR'
    
    @property
    def is_deepfake(self):
        return self.prediction == 'FAKE'
    
    @property
    def is_real(self):
        return self.prediction == 'REAL'
    
    @property
    def confidence_percentage(self):
        if self.confidence_score is not None:
            return round(self.confidence_score * 100, 2)
        return None
    
    @property
    def file_size_mb(self):
        """Return file size in MB"""
        if self.file_size:
            return round(self.file_size / (1024 * 1024), 2)
        return None
    
    def generate_report_filename(self):
        """Generate filename for downloadable report"""
        timestamp = self.created_at.strftime("%Y%m%d_%H%M%S")
        clean_filename = "".join(c for c in self.original_filename if c.isalnum() or c in "._- ")
        name_part = clean_filename.split('.')[0][:30]  # Max 30 chars
        return f"deepfake_report_{name_part}_{timestamp}.pdf"


class FrameAnalysis(models.Model):
    """Model to store frame-by-frame analysis results"""
    
    detection_result = models.ForeignKey(DetectionResult, on_delete=models.CASCADE, related_name='frame_analyses')
    frame_number = models.IntegerField()
    timestamp = models.FloatField(help_text="Timestamp in video (seconds)")
    
    # Frame detection results
    prediction = models.CharField(max_length=20, choices=DetectionResult.DETECTION_CHOICES)
    confidence_score = models.FloatField(help_text="Confidence score for this frame")
    fake_probability = models.FloatField(help_text="Probability of being fake")
    real_probability = models.FloatField(help_text="Probability of being real")
    
    # Face detection in frame
    face_detected = models.BooleanField(default=False)
    face_bbox = models.JSONField(null=True, blank=True, help_text="Face bounding box coordinates")
    face_landmarks = models.JSONField(null=True, blank=True, help_text="Face landmark coordinates")
    
    # Feature scores
    temporal_consistency = models.FloatField(null=True, blank=True)
    spatial_consistency = models.FloatField(null=True, blank=True)
    compression_artifacts = models.FloatField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['frame_number']
        unique_together = ['detection_result', 'frame_number']
        verbose_name = "Frame Analysis"
        verbose_name_plural = "Frame Analyses"
    
    def __str__(self):
        return f"Frame {self.frame_number} - {self.prediction} ({self.confidence_score})"


class ModelPerformance(models.Model):
    """Model to track detection model performance metrics"""
    
    model_name = models.CharField(max_length=100)
    model_version = models.CharField(max_length=50)
    
    # Performance metrics
    total_predictions = models.IntegerField(default=0)
    correct_predictions = models.IntegerField(default=0)
    false_positives = models.IntegerField(default=0)
    false_negatives = models.IntegerField(default=0)
    true_positives = models.IntegerField(default=0)
    true_negatives = models.IntegerField(default=0)
    
    # Calculated metrics
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    
    # Performance statistics
    avg_processing_time = models.FloatField(null=True, blank=True)
    avg_confidence_score = models.FloatField(null=True, blank=True)
    
    last_updated = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['model_name', 'model_version']
        verbose_name = "Model Performance"
        verbose_name_plural = "Model Performance Metrics"
    
    def __str__(self):
        return f"{self.model_name} v{self.model_version} - Accuracy: {self.accuracy}"
    
    def calculate_metrics(self):
        """Calculate performance metrics from confusion matrix values"""
        if self.total_predictions > 0:
            self.accuracy = (self.true_positives + self.true_negatives) / self.total_predictions
        
        if (self.true_positives + self.false_positives) > 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)
        
        if (self.true_positives + self.false_negatives) > 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        
        if self.precision and self.recall:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        
        self.save()
