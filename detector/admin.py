from django.contrib import admin
from .models import DetectionResult, FrameAnalysis, ModelPerformance


@admin.register(DetectionResult)
class DetectionResultAdmin(admin.ModelAdmin):
    list_display = ['original_filename', 'prediction', 'confidence_score', 
                   'face_detected', 'created_at', 'processing_time']
    list_filter = ['prediction', 'face_detected', 'created_at']
    search_fields = ['original_filename', 'prediction']
    readonly_fields = ['created_at', 'updated_at']
    ordering = ['-created_at']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('original_filename', 'video_file', 'file_size')
        }),
        ('Detection Results', {
            'fields': ('prediction', 'confidence_score', 'real_probability', 
                      'fake_probability', 'face_detected', 'face_count')
        }),
        ('Video Properties', {
            'fields': ('video_duration', 'video_fps', 'video_resolution', 
                      'frames_analyzed')
        }),
        ('Model Information', {
            'fields': ('model_used', 'model_version', 'detection_method')
        }),
        ('Processing Information', {
            'fields': ('processing_time', 'error_message')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'processed_at'),
            'classes': ('collapse',)
        })
    )


@admin.register(FrameAnalysis)
class FrameAnalysisAdmin(admin.ModelAdmin):
    list_display = ['detection_result', 'frame_number', 'prediction', 
                   'confidence_score', 'face_detected', 'timestamp']
    list_filter = ['prediction', 'face_detected']
    search_fields = ['detection_result__original_filename']
    readonly_fields = ['created_at']
    ordering = ['detection_result', 'frame_number']
    
    fieldsets = (
        ('Frame Information', {
            'fields': ('detection_result', 'frame_number', 'timestamp')
        }),
        ('Detection Results', {
            'fields': ('prediction', 'confidence_score', 'real_probability', 
                      'fake_probability', 'face_detected')
        }),
        ('Feature Analysis', {
            'fields': ('temporal_consistency', 'spatial_consistency', 
                      'compression_artifacts')
        }),
        ('Face Detection', {
            'fields': ('face_bbox', 'face_landmarks')
        })
    )


@admin.register(ModelPerformance)
class ModelPerformanceAdmin(admin.ModelAdmin):
    list_display = ['model_name', 'model_version', 'accuracy', 
                   'precision', 'recall', 'f1_score', 'created_at']
    list_filter = ['model_name', 'created_at']
    search_fields = ['model_name', 'model_version']
    readonly_fields = ['created_at', 'last_updated']
    ordering = ['-created_at']
    
    fieldsets = (
        ('Model Information', {
            'fields': ('model_name', 'model_version')
        }),
        ('Confusion Matrix', {
            'fields': ('total_predictions', 'correct_predictions', 
                      'true_positives', 'true_negatives', 
                      'false_positives', 'false_negatives')
        }),
        ('Performance Metrics', {
            'fields': ('accuracy', 'precision', 'recall', 'f1_score')
        }),
        ('Performance Statistics', {
            'fields': ('avg_processing_time', 'avg_confidence_score')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'last_updated')
        })
    )
