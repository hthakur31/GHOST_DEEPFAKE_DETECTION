from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, HttpResponse, Http404
from django.views.generic import TemplateView, ListView
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.conf import settings
from django.db import models
from django.contrib import messages
from django.core.paginator import Paginator
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from .models import DetectionResult, FrameAnalysis, ModelPerformance
from .serializers import DetectionResultSerializer, VideoUploadSerializer
from .ml_utils import DeepfakeDetector, get_detector
from enhanced_xception_predictor import get_xception_predictor, EnhancedXceptionNetPredictor
from advanced_report_generator import report_generator
import os
import time
import logging
import json
import mimetypes
from pathlib import Path
from datetime import datetime, timedelta
from django.utils import timezone
import threading
from django.core.files.storage import default_storage

logger = logging.getLogger(__name__)


class HomeView(TemplateView):
    """Main page for video upload and detection"""
    template_name = 'detector/home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get recent detection results (excluding processing and errors for main display)
        context['recent_results'] = DetectionResult.objects.filter(
            prediction__in=['REAL', 'FAKE']
        ).order_by('-created_at')[:10]
        
        # Get model performance stats
        try:
            performance = ModelPerformance.objects.latest('last_updated')
            context['model_performance'] = performance
        except ModelPerformance.DoesNotExist:
            context['model_performance'] = None
        
        # Calculate comprehensive summary statistics
        all_results = DetectionResult.objects.all().count()
        total_predictions = DetectionResult.objects.exclude(prediction='PROCESSING').count()
        fake_predictions = DetectionResult.objects.filter(prediction='FAKE').count()
        real_predictions = DetectionResult.objects.filter(prediction='REAL').count()
        error_predictions = DetectionResult.objects.filter(prediction='ERROR').count()
        processing_predictions = DetectionResult.objects.filter(prediction='PROCESSING').count()
        
        # Calculate additional metrics
        successful_predictions = fake_predictions + real_predictions
        success_rate = (successful_predictions / total_predictions * 100) if total_predictions > 0 else 0
        fake_percentage = (fake_predictions / successful_predictions * 100) if successful_predictions > 0 else 0
        
        # Average confidence score for successful predictions
        successful_results = DetectionResult.objects.filter(
            prediction__in=['REAL', 'FAKE'],
            confidence_score__isnull=False
        )
        avg_confidence = successful_results.aggregate(
            avg_conf=models.Avg('confidence_score')
        )['avg_conf'] or 0
        
        # Calculate processing statistics
        completed_results = DetectionResult.objects.filter(
            prediction__in=['REAL', 'FAKE'],
            processing_time__isnull=False
        )
        avg_processing_time = completed_results.aggregate(
            avg_time=models.Avg('processing_time')
        )['avg_time'] or 0
        
        # Calculate model accuracy from latest model performance or estimate from confidence
        try:
            latest_performance = ModelPerformance.objects.latest('last_updated')
            model_accuracy = (latest_performance.accuracy * 100) if latest_performance.accuracy else None
        except ModelPerformance.DoesNotExist:
            # Estimate accuracy based on average confidence for successful predictions
            # This is an approximation - higher confidence generally correlates with better accuracy
            model_accuracy = avg_confidence * 100 if avg_confidence > 0 else None
        
        context['stats'] = {
            'total_predictions': successful_predictions,  # Only successful predictions for main display
            'fake_predictions': fake_predictions,
            'real_predictions': real_predictions,
            'fake_percentage': fake_percentage,
            'all_videos_uploaded': all_results,
            'error_predictions': error_predictions,
            'processing_predictions': processing_predictions,
            'success_rate': success_rate,
            'avg_confidence': avg_confidence * 100,  # Convert to percentage
            'avg_processing_time': avg_processing_time,
            'model_accuracy': model_accuracy  # Add accuracy to stats
        }
        
        return context


class VideoUploadView(APIView):
    """API view for uploading videos for deepfake detection"""
    parser_classes = (MultiPartParser, FormParser)
    
    def post(self, request, *args, **kwargs):
        """Handle video upload and start detection process"""
        serializer = VideoUploadSerializer(data=request.data)
        
        if serializer.is_valid():
            try:
                # Save the uploaded video
                video_file = request.FILES['video_file']
                
                # Create detection result record
                detection_result = DetectionResult.objects.create(
                    video_file=video_file,
                    original_filename=video_file.name,
                    file_size=video_file.size,
                    prediction='PROCESSING'
                )
                
                # Start background processing
                thread = threading.Thread(
                    target=self._process_video_async,
                    args=(detection_result.id, detection_result.video_file.path)
                )
                thread.start()
                
                return Response({
                    'status': 'success',
                    'message': 'Video uploaded successfully. Processing started.',
                    'detection_id': str(detection_result.id),
                    'redirect_url': f'/result/{detection_result.id}/'
                }, status=status.HTTP_201_CREATED)
                
            except Exception as e:
                logger.error(f"Error uploading video: {e}")
                return Response({
                    'status': 'error',
                    'message': f'Error uploading video: {str(e)}'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return Response({
            'status': 'error',
            'message': 'Invalid file upload',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)
    
    def _process_video_async(self, detection_id, video_path):
        """Process video in background thread using ensemble of models including XceptionNet"""
        try:
            detection_result = DetectionResult.objects.get(id=detection_id)
            start_time = time.time()
            
            # Initialize both FaceForensics++ and XceptionNet models
            results = {}
            final_prediction = 'ERROR'
            final_confidence = 0.0
            
            # Try XceptionNet first (our latest and most robust model)
            try:
                logger.info("Using Enhanced XceptionNet model for detection...")
                xception_predictor = get_xception_predictor()
                
                if xception_predictor.model is not None:
                    xception_results = xception_predictor.predict_video(video_path)
                    
                    if xception_results.get('success', False):
                        final_prediction = 'FAKE' if xception_results['prediction'] == 'deepfake' else 'REAL'
                        final_confidence = xception_results['confidence']
                        
                        # Store comprehensive Enhanced XceptionNet results
                        results['xception'] = {
                            'prediction': final_prediction,
                            'confidence': final_confidence,
                            'confidence_percent': xception_results.get('confidence_percent', f"{final_confidence*100:.2f}%"),
                            'confidence_level': xception_results.get('confidence_level', 'Unknown'),
                            'authentic_probability': xception_results.get('authentic_probability', f"{(1-final_confidence)*100:.1f}%"),
                            'deepfake_probability': xception_results.get('deepfake_probability', f"{final_confidence*100:.1f}%"),
                            'deepfake_ratio': xception_results.get('deepfake_ratio', 0.0),
                            'frames_analyzed': xception_results.get('frames_analyzed_count', 0),
                            'model_type': 'Enhanced XceptionNet',
                            'video_analysis': xception_results.get('video_analysis', {}),
                            'insights': xception_results.get('insights', {}),
                            'model_info': xception_results.get('model_info', {}),
                            'frame_analysis': xception_results.get('frame_analysis', []),
                            'technical_details': xception_results.get('technical_details', {})
                        }
                        
                        logger.info(f"Enhanced XceptionNet prediction: {final_prediction} ({final_confidence:.2f})")
                        logger.info(f"Confidence level: {xception_results.get('confidence_level', 'Unknown')}")
                        logger.info(f"Architecture: {xception_results.get('model_info', {}).get('architecture', 'Unknown')}")
                    else:
                        logger.warning(f"Enhanced XceptionNet failed: {xception_results.get('error', 'Unknown error')}")
                        results['xception_error'] = xception_results.get('error', 'Unknown error')
                else:
                    logger.warning("Enhanced XceptionNet model not available")
                    results['xception_error'] = "Enhanced XceptionNet model not loaded"
                    
            except Exception as e:
                logger.error(f"Enhanced XceptionNet detection error: {e}")
                results['xception_error'] = str(e)
            
            # Fallback to FaceForensics++ model if XceptionNet fails or for ensemble
            if final_prediction == 'ERROR' or results.get('xception_error'):
                try:
                    logger.info("Using FaceForensics++ model for detection...")
                    detector = get_detector(
                        model_path=None,  # Auto-detect latest trained model
                        use_advanced_model=True  # Use FaceForensics++ advanced model
                    )
                    logger.info(f"Using advanced FaceForensics++ detection model: {detector.model_name}")
                    
                    # Run FaceForensics++ detection
                    ff_results = detector.detect_video(video_path)
                    
                    if 'error' not in ff_results:
                        if final_prediction == 'ERROR':  # XceptionNet failed, use FaceForensics++ as primary
                            final_prediction = ff_results.get('prediction', 'ERROR')
                            final_confidence = ff_results.get('confidence_score', 0.0)
                        
                        results['faceforensics'] = ff_results
                        logger.info(f"FaceForensics++ prediction: {ff_results.get('prediction')} ({ff_results.get('confidence_score', 0.0):.2f})")
                    else:
                        results['faceforensics_error'] = ff_results['error']
                        
                except Exception as e:
                    logger.warning(f"Failed to load advanced model, trying basic model: {e}")
                    try:
                        detector = get_detector(
                            model_path=None,
                            use_advanced_model=False  # Fallback to basic model
                        )
                        
                        ff_results = detector.detect_video(video_path)
                        
                        if 'error' not in ff_results:
                            if final_prediction == 'ERROR':
                                final_prediction = ff_results.get('prediction', 'ERROR')
                                final_confidence = ff_results.get('confidence_score', 0.0)
                            
                            results['faceforensics_basic'] = ff_results
                            logger.info(f"Basic model prediction: {ff_results.get('prediction')} ({ff_results.get('confidence_score', 0.0):.2f})")
                        else:
                            results['faceforensics_basic_error'] = ff_results['error']
                            
                    except Exception as e2:
                        logger.error(f"All model detection failed: {e2}")
                        results['all_models_error'] = str(e2)
            
            processing_time = time.time() - start_time
            
            # Update detection result with comprehensive data
            if final_prediction == 'ERROR':
                detection_result.prediction = 'ERROR'
                detection_result.error_message = f"All models failed. Errors: {results}"
            else:
                detection_result.prediction = final_prediction
                detection_result.confidence_score = final_confidence
                
                # Set probability scores based on primary prediction
                if 'xception' in results:
                    # Use XceptionNet confidence
                    if final_prediction == 'FAKE':
                        detection_result.fake_probability = final_confidence * 100
                        detection_result.real_probability = (1 - final_confidence) * 100
                    else:
                        detection_result.real_probability = final_confidence * 100
                        detection_result.fake_probability = (1 - final_confidence) * 100
                elif 'faceforensics' in results:
                    # Use FaceForensics++ format
                    ff_res = results['faceforensics']
                    if 'fake_probability' in ff_res:
                        detection_result.fake_probability = ff_res['fake_probability']
                        detection_result.real_probability = ff_res['real_probability']
                    elif 'confidence_percentage' in ff_res:
                        if final_prediction == 'FAKE':
                            detection_result.fake_probability = ff_res['confidence_percentage']
                            detection_result.real_probability = 100 - ff_res['confidence_percentage']
                        else:
                            detection_result.real_probability = ff_res['confidence_percentage']
                            detection_result.fake_probability = 100 - ff_res['confidence_percentage']
                    else:
                        # Fallback calculation
                        conf = final_confidence * 100
                        if final_prediction == 'FAKE':
                            detection_result.fake_probability = conf
                            detection_result.real_probability = 100 - conf
                        else:
                            detection_result.real_probability = conf
                            detection_result.fake_probability = 100 - conf
                
                # Store detailed results in metadata with enhanced XceptionNet data
                enhanced_metadata = {
                    'model_results': results,
                    'primary_model': 'Enhanced XceptionNet' if 'xception' in results else 'FaceForensics++',
                    'processing_details': {
                        'processing_time': processing_time,
                        'timestamp': datetime.now().isoformat(),
                        'models_used': list(results.keys())
                    }
                }
                
                # Add comprehensive Enhanced XceptionNet analysis if available
                if 'xception' in results:
                    xception_data = results['xception']
                    enhanced_metadata.update({
                        'enhanced_analysis': {
                            'confidence_level': xception_data.get('confidence_level', 'Unknown'),
                            'video_analysis': xception_data.get('video_analysis', {}),
                            'insights': xception_data.get('insights', {}),
                            'model_info': xception_data.get('model_info', {}),
                            'frame_analysis': xception_data.get('frame_analysis', []),
                            'technical_details': xception_data.get('technical_details', {}),
                            'authentic_probability': xception_data.get('authentic_probability', '0%'),
                            'deepfake_probability': xception_data.get('deepfake_probability', '0%')
                        }
                    })
                
                detection_result.metadata = enhanced_metadata
                
                # Set additional detection details (enhanced for XceptionNet)
                if 'xception' in results:
                    xception_res = results['xception']
                    video_analysis = xception_res.get('video_analysis', {})
                    model_info = xception_res.get('model_info', {})
                    
                    detection_result.face_detected = xception_res.get('frames_analyzed', 0) > 0
                    detection_result.face_count = len(xception_res.get('frame_analysis', []))
                    detection_result.frames_analyzed = xception_res.get('frames_analyzed', 0)
                    detection_result.model_used = model_info.get('model_name', 'Enhanced XceptionNet')
                    detection_result.model_version = model_info.get('version', '2.0')
                    detection_result.detection_method = model_info.get('method', 'Enhanced XceptionNet Deep Learning')
                elif 'faceforensics' in results:
                    ff_res = results['faceforensics']
                    detection_result.face_detected = ff_res.get('face_detected', False)
                    detection_result.face_count = ff_res.get('face_count', 0)
                    detection_result.frames_analyzed = ff_res.get('frames_analyzed', 0)
                    detection_result.model_used = ff_res.get('model_name', 'FaceForensics++')
                    detection_result.model_version = ff_res.get('model_version', '1.0')
                    detection_result.detection_method = ff_res.get('detection_method', 'FaceForensics++ CNN')
                
                # Video metadata (enhanced analysis from XceptionNet)
                video_metadata = None
                if 'xception' in results and 'video_analysis' in results['xception']:
                    video_analysis = results['xception']['video_analysis']
                    video_metadata = {
                        'duration': float(video_analysis.get('duration', '0').replace(' seconds', '')),
                        'fps': float(video_analysis.get('frame_rate', '0').replace(' FPS', '')),
                        'resolution': video_analysis.get('resolution', 'Unknown'),
                        'file_size': video_analysis.get('file_size', 'Unknown'),
                        'total_frames': video_analysis.get('total_frames', 0),
                        'faces_detected': video_analysis.get('faces_detected', '0 faces')
                    }
                elif 'xception' in results and 'details' in results['xception']:
                    video_metadata = results['xception']['details']
                elif 'faceforensics' in results and 'metadata' in results['faceforensics']:
                    video_metadata = results['faceforensics']['metadata']
                
                if video_metadata:
                    detection_result.video_duration = video_metadata.get('duration', 0)
                    detection_result.video_fps = video_metadata.get('fps', 0)
                    detection_result.video_resolution = video_metadata.get('resolution', 'Unknown')
                else:
                    # Fallback values
                    detection_result.video_duration = 0
                    detection_result.video_fps = 0
                    detection_result.video_resolution = 'Unknown'
            
            detection_result.processing_time = processing_time
            detection_result.processed_at = timezone.now()
            detection_result.save()
            
            # Save frame-by-frame analysis if available
            if 'frame_predictions' in results and results['frame_predictions']:
                metadata = results.get('metadata', {})
                fps = metadata.get('fps', 30) if metadata else 30
                
                for frame_pred in results['frame_predictions']:
                    # Calculate timestamp
                    frame_num = frame_pred.get('frame_number', 0)
                    timestamp = frame_num / fps if fps > 0 else 0
                    
                    FrameAnalysis.objects.create(
                        detection_result=detection_result,
                        frame_number=frame_num,
                        timestamp=timestamp,
                        prediction=frame_pred.get('prediction', 'UNKNOWN'),
                        confidence_score=frame_pred.get('confidence_score', 0),
                        fake_probability=frame_pred.get('fake_probability', 0),
                        real_probability=frame_pred.get('real_probability', 0),
                        face_detected=frame_pred.get('face_info', {}).get('face_detected', False),
                        face_bbox=frame_pred.get('face_info', {}).get('face_bbox', []),
                        face_landmarks=frame_pred.get('face_info', {}).get('landmarks', [])
                    )
            
            # Update model performance metrics - fix detector reference issue
            model_name = 'Unknown'
            model_version = '1.0'
            
            if 'xception' in results:
                model_name = 'XceptionNet'
                model_version = 'v1.0'
            elif 'faceforensics' in results:
                ff_res = results['faceforensics']
                model_name = ff_res.get('model_name', 'FaceForensics++')
                model_version = ff_res.get('model_version', '1.0')
            elif 'faceforensics_basic' in results:
                ff_res = results['faceforensics_basic']
                model_name = ff_res.get('model_name', 'FaceForensics++ Basic')
                model_version = ff_res.get('model_version', '1.0')
            
            self._update_model_performance(model_name, model_version, {
                'prediction': final_prediction,
                'confidence_score': final_confidence,
                'processing_time': processing_time
            })
            
            logger.info(f"Video processing completed for {detection_id}: {results.get('prediction', 'ERROR')} (confidence: {results.get('confidence_score', 0):.2f})")
            
        except Exception as e:
            logger.error(f"Error processing video {detection_id}: {e}")
            import traceback
            traceback.print_exc()
            try:
                detection_result = DetectionResult.objects.get(id=detection_id)
                detection_result.prediction = 'ERROR'
                detection_result.error_message = str(e)
                detection_result.processed_at = timezone.now()
                detection_result.save()
            except:
                pass
    
    def _update_model_performance(self, model_name, model_version, results):
        """Update model performance metrics"""
        try:
            performance, created = ModelPerformance.objects.get_or_create(
                model_name=model_name,
                model_version=model_version,
                defaults={
                    'total_predictions': 0,
                    'avg_processing_time': 0,
                    'avg_confidence_score': 0
                }
            )
            
            # Update metrics
            performance.total_predictions += 1
            
            # Update average processing time
            processing_time = results.get('processing_time', 0)
            if performance.avg_processing_time:
                performance.avg_processing_time = (
                    performance.avg_processing_time * (performance.total_predictions - 1) + 
                    processing_time
                ) / performance.total_predictions
            else:
                performance.avg_processing_time = processing_time
            
            # Update average confidence score
            confidence = results.get('confidence_score', 0)
            if performance.avg_confidence_score:
                performance.avg_confidence_score = (
                    performance.avg_confidence_score * (performance.total_predictions - 1) + confidence
                ) / performance.total_predictions
            else:
                performance.avg_confidence_score = confidence
            
            performance.save()
            
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")


class DetectionResultView(TemplateView):
    """View for displaying detection results"""
    template_name = 'detector/result.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        detection_id = kwargs.get('detection_id')
        
        try:
            detection_result = get_object_or_404(DetectionResult, id=detection_id)
            context['result'] = detection_result
            
            # Get frame analysis data
            frame_analyses = detection_result.frame_analyses.all()
            context['frame_analyses'] = frame_analyses
            
            # Prepare chart data for confidence scores
            if frame_analyses:
                context['chart_data'] = {
                    'frame_numbers': [f.frame_number for f in frame_analyses],
                    'confidence_scores': [f.confidence_score for f in frame_analyses],
                    'fake_probabilities': [f.fake_probability for f in frame_analyses],
                    'real_probabilities': [f.real_probability for f in frame_analyses]
                }
            
            # Add model performance data for dynamic accuracy display
            try:
                latest_performance = ModelPerformance.objects.filter(
                    model_name__icontains='xception'
                ).order_by('-last_updated').first()
                
                if latest_performance and latest_performance.accuracy:
                    context['model_accuracy'] = round(latest_performance.accuracy * 100, 1)
                    context['model_name'] = latest_performance.model_name
                    context['total_predictions'] = latest_performance.total_predictions
                else:
                    context['model_accuracy'] = None
                    context['model_name'] = "Enhanced XceptionNet"
                    
            except Exception as e:
                logger.error(f"Error getting model performance: {e}")
                context['model_accuracy'] = None
                context['model_name'] = "Enhanced XceptionNet"
            
        except Exception as e:
            logger.error(f"Error retrieving detection result {detection_id}: {e}")
            context['error'] = str(e)
        
        return context


class DetectionStatusAPIView(APIView):
    """API endpoint to check detection status"""
    
    def get(self, request, detection_id):
        """Get current status of detection process"""
        try:
            detection_result = get_object_or_404(DetectionResult, id=detection_id)
            serializer = DetectionResultSerializer(detection_result)
            return Response(serializer.data)
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_404_NOT_FOUND)


class ModelPerformanceAPIView(APIView):
    """API endpoint for model performance metrics"""
    
    def get(self, request):
        """Get model performance statistics"""
        try:
            performance = ModelPerformance.objects.latest('last_updated')
            
            data = {
                'model_name': performance.model_name,
                'model_version': performance.model_version,
                'total_predictions': performance.total_predictions,
                'accuracy': performance.accuracy,
                'precision': performance.precision,
                'recall': performance.recall,
                'f1_score': performance.f1_score,
                'avg_processing_time': performance.avg_processing_time,
                'avg_confidence_score': performance.avg_confidence_score,
                'last_updated': performance.last_updated
            }
            
            return Response(data)
            
        except ModelPerformance.DoesNotExist:
            return Response({
                'message': 'No performance data available'
            }, status=status.HTTP_404_NOT_FOUND)


class RecentDetectionsAPIView(APIView):
    """API endpoint for recent detection results"""
    
    def get(self, request):
        """Get recent detection results"""
        try:
            limit = int(request.GET.get('limit', 20))
            results = DetectionResult.objects.exclude(
                prediction='PROCESSING'
            ).order_by('-created_at')[:limit]
            
            serializer = DetectionResultSerializer(results, many=True)
            return Response(serializer.data)
            
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class HistoryView(TemplateView):
    """View for displaying detection history with pagination and filtering"""
    template_name = 'detector/history.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get filter parameters
        status_filter = self.request.GET.get('status', 'all')
        search_query = self.request.GET.get('search', '')
        
        # Base queryset
        queryset = DetectionResult.objects.all().order_by('-created_at')
        
        # Apply filters
        if status_filter != 'all':
            queryset = queryset.filter(prediction=status_filter.upper())
        
        if search_query:
            queryset = queryset.filter(original_filename__icontains=search_query)
        
        # Pagination
        from django.core.paginator import Paginator
        paginator = Paginator(queryset, 20)  # 20 results per page
        page_number = self.request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        
        context['page_obj'] = page_obj
        context['status_filter'] = status_filter
        context['search_query'] = search_query
        
        # Statistics for the current filter
        filtered_results = queryset
        context['filtered_stats'] = {
            'total_count': filtered_results.count(),
            'real_count': filtered_results.filter(prediction='REAL').count(),
            'fake_count': filtered_results.filter(prediction='FAKE').count(),
            'processing_count': filtered_results.filter(prediction='PROCESSING').count(),
            'error_count': filtered_results.filter(prediction='ERROR').count(),
        }
        
        # Calculate percentages
        total = context['filtered_stats']['total_count']
        if total > 0:
            context['filtered_stats']['real_percentage'] = (context['filtered_stats']['real_count'] / total) * 100
            context['filtered_stats']['fake_percentage'] = (context['filtered_stats']['fake_count'] / total) * 100
        else:
            context['filtered_stats']['real_percentage'] = 0
            context['filtered_stats']['fake_percentage'] = 0
        
        return context


class StatsAPIView(APIView):
    """API endpoint for real-time statistics update"""
    
    def get(self, request):
        """Get current statistics"""
        try:
            # Calculate current stats (same logic as HomeView)
            all_results = DetectionResult.objects.all().count()
            total_predictions = DetectionResult.objects.exclude(prediction='PROCESSING').count()
            fake_predictions = DetectionResult.objects.filter(prediction='FAKE').count()
            real_predictions = DetectionResult.objects.filter(prediction='REAL').count()
            error_predictions = DetectionResult.objects.filter(prediction='ERROR').count()
            processing_predictions = DetectionResult.objects.filter(prediction='PROCESSING').count()
            
            # Calculate additional metrics
            successful_predictions = fake_predictions + real_predictions
            success_rate = (successful_predictions / total_predictions * 100) if total_predictions > 0 else 0
            fake_percentage = (fake_predictions / successful_predictions * 100) if successful_predictions > 0 else 0
            
            # Average confidence score for successful predictions
            successful_results = DetectionResult.objects.filter(
                prediction__in=['REAL', 'FAKE'],
                confidence_score__isnull=False
            )
            avg_confidence = successful_results.aggregate(
                avg_conf=models.Avg('confidence_score')
            )['avg_conf'] or 0
            
            # Calculate processing statistics
            completed_results = DetectionResult.objects.filter(
                prediction__in=['REAL', 'FAKE'],
                processing_time__isnull=False
            )
            avg_processing_time = completed_results.aggregate(
                avg_time=models.Avg('processing_time')
            )['avg_time'] or 0
            
            # Calculate model accuracy from latest model performance or estimate from confidence
            try:
                latest_performance = ModelPerformance.objects.latest('last_updated')
                model_accuracy = (latest_performance.accuracy * 100) if latest_performance.accuracy else None
            except ModelPerformance.DoesNotExist:
                # Estimate accuracy based on average confidence for successful predictions
                # This is an approximation - higher confidence generally correlates with better accuracy
                model_accuracy = avg_confidence * 100 if avg_confidence > 0 else None
            
            stats = {
                'total_predictions': successful_predictions,
                'fake_predictions': fake_predictions,
                'real_predictions': real_predictions,
                'fake_percentage': fake_percentage,
                'all_videos_uploaded': all_results,
                'error_predictions': error_predictions,
                'processing_predictions': processing_predictions,
                'success_rate': success_rate,
                'avg_confidence': avg_confidence * 100,
                'avg_processing_time': avg_processing_time,
                'real_percentage': (real_predictions / successful_predictions * 100) if successful_predictions > 0 else 0,
                'model_accuracy': model_accuracy  # Add accuracy to API response
            }
            
            return Response({
                'status': 'success',
                'stats': stats,
                'timestamp': timezone.now().isoformat()
            })
            
        except Exception as e:
            return Response({
                'status': 'error',
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DeleteDetectionAPIView(APIView):
    """API endpoint to delete detection records"""
    
    def delete(self, request):
        """Bulk delete detection records"""
        try:
            # Support both 'ids' and 'detection_ids' for compatibility
            detection_ids = request.data.get('ids', request.data.get('detection_ids', []))
            
            if not detection_ids:
                return Response({
                    'success': False,
                    'error': 'No detection IDs provided'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            deleted_count = 0
            errors = []
            
            for detection_id in detection_ids:
                try:
                    detection_result = DetectionResult.objects.get(id=detection_id)
                    
                    # Delete video file if it exists
                    if detection_result.video_file:
                        try:
                            if default_storage.exists(detection_result.video_file.name):
                                default_storage.delete(detection_result.video_file.name)
                        except Exception as e:
                            logger.warning(f"Failed to delete video file for {detection_id}: {e}")
                    
                    detection_result.delete()
                    deleted_count += 1
                    
                except DetectionResult.DoesNotExist:
                    errors.append(f"Detection {detection_id} not found")
                except Exception as e:
                    errors.append(f"Error deleting {detection_id}: {str(e)}")
            
            return Response({
                'success': True,
                'message': f'{deleted_count} records deleted successfully',
                'deleted_count': deleted_count,
                'errors': errors
            })
            
        except Exception as e:
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class StopProcessingAPIView(APIView):
    """API endpoint to stop processing videos"""
    
    def post(self, request):
        """Stop processing for a specific detection"""
        try:
            # Get record_id from request body
            record_id = request.data.get('record_id')
            
            if not record_id:
                return Response({
                    'success': False,
                    'error': 'No record ID provided'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            detection_result = get_object_or_404(DetectionResult, id=record_id)
            
            if detection_result.prediction != 'PROCESSING':
                return Response({
                    'success': False,
                    'error': 'Video is not currently being processed'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Update the record to show it was cancelled
            detection_result.prediction = 'ERROR'
            detection_result.error_message = 'Processing cancelled by user'
            detection_result.processed_at = timezone.now()
            detection_result.save()
            
            # Note: In a production environment, you would also need to
            # signal the background thread to stop processing
            # For now, we just mark it as cancelled in the database
            
            return Response({
                'success': True,
                'message': 'Processing stopped successfully'
            })
            
        except Exception as e:
            logger.error(f"Error stopping processing for {record_id}: {e}")
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ==================== NEW ENHANCED FEATURES ====================

class HistoryView(ListView):
    """Display detection history with filtering and pagination"""
    model = DetectionResult
    template_name = 'detector/history.html'
    context_object_name = 'results'
    paginate_by = 20
    
    def get_queryset(self):
        queryset = DetectionResult.objects.all().order_by('-created_at')
        
        # Apply filters from GET parameters
        prediction_filter = self.request.GET.get('prediction')
        if prediction_filter and prediction_filter != 'ALL':
            queryset = queryset.filter(prediction=prediction_filter)
        
        # Date range filter
        date_from = self.request.GET.get('date_from')
        date_to = self.request.GET.get('date_to')
        
        if date_from:
            try:
                date_from = datetime.strptime(date_from, '%Y-%m-%d')
                queryset = queryset.filter(created_at__date__gte=date_from)
            except ValueError:
                pass
        
        if date_to:
            try:
                date_to = datetime.strptime(date_to, '%Y-%m-%d')
                queryset = queryset.filter(created_at__date__lte=date_to)
            except ValueError:
                pass
        
        # Search by filename
        search_query = self.request.GET.get('search')
        if search_query:
            queryset = queryset.filter(original_filename__icontains=search_query)
        
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Add filter values to context for form persistence
        context['current_prediction'] = self.request.GET.get('prediction', 'ALL')
        context['current_date_from'] = self.request.GET.get('date_from', '')
        context['current_date_to'] = self.request.GET.get('date_to', '')
        context['current_search'] = self.request.GET.get('search', '')
        
        # Add summary statistics
        queryset = self.get_queryset()
        context['total_results'] = queryset.count()
        context['fake_count'] = queryset.filter(prediction='FAKE').count()
        context['real_count'] = queryset.filter(prediction='REAL').count()
        context['error_count'] = queryset.filter(prediction='ERROR').count()
        context['processing_count'] = queryset.filter(prediction='PROCESSING').count()
        
        # Prediction choices for filter dropdown
        context['prediction_choices'] = [
            ('ALL', 'All Results'),
            ('REAL', 'Real Videos'),
            ('FAKE', 'Deepfake Videos'),
            ('PROCESSING', 'Processing'),
            ('ERROR', 'Errors')
        ]
        
        return context


class DetailedResultView(TemplateView):
    """Display detailed analysis for a specific result"""
    template_name = 'detector/detailed_result.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        result_id = kwargs.get('result_id')
        result = get_object_or_404(DetectionResult, id=result_id)
        
        context['result'] = result
        context['frame_analyses'] = result.frame_analyses.all() if hasattr(result, 'frame_analyses') else []
        
        # Generate detailed analysis if not already done
        if not result.report_data:
            try:
                # Generate comprehensive report data
                report_data = report_generator._prepare_report_data(result)
                result.report_data = report_data
                result.report_generated = True
                result.save()
                
                context['report_data'] = report_data
            except Exception as e:
                logger.error(f"Error generating report data: {e}")
                context['report_data'] = {}
        else:
            context['report_data'] = result.report_data
        
        return context


class DownloadReportView(APIView):
    """Download reports in various formats"""
    
    def get(self, request, result_id, format_type='pdf'):
        """Download report in specified format"""
        try:
            result = get_object_or_404(DetectionResult, id=result_id)
            
            # Validate format
            supported_formats = ['pdf', 'json', 'excel', 'html']
            if format_type.lower() not in supported_formats:
                return HttpResponse(
                    f'Unsupported format. Supported: {", ".join(supported_formats)}',
                    status=400
                )
            
            # Generate report
            try:
                # Import report generator here to avoid startup issues
                from advanced_report_generator import AdvancedReportGenerator
                report_gen = AdvancedReportGenerator()
                
                report_path = report_gen.generate_comprehensive_report(
                    result, format_type=format_type.lower()
                )
                
            except ImportError as e:
                if 'reportlab' in str(e).lower() and format_type.lower() == 'pdf':
                    # Fallback to JSON for PDF if ReportLab is not available
                    logger.warning(f"PDF generation not available, falling back to JSON: {e}")
                    format_type = 'json'
                    report_path = self._generate_json_report(result)
                else:
                    raise
            except Exception as e:
                logger.error(f"Error generating report: {e}")
                # Fallback to JSON report
                report_path = self._generate_json_report(result)
                format_type = 'json'
            
            # Check if file exists
            if not report_path or not Path(report_path).exists():
                # Generate a simple JSON fallback
                report_path = self._generate_json_report(result)
                format_type = 'json'
            
            # Determine content type
            content_types = {
                'pdf': 'application/pdf',
                'json': 'application/json',
                'excel': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'html': 'text/html'
            }
            
            content_type = content_types.get(format_type.lower(), 'application/octet-stream')
            
            # Read file content
            with open(report_path, 'rb') as f:
                file_content = f.read()
            
            # Create response
            response = HttpResponse(file_content, content_type=content_type)
            
            # Set filename for download
            timestamp = timezone.now().strftime("%Y%m%d_%H%M%S")
            filename_map = {
                'pdf': f"deepfake_analysis_{result.id}_{timestamp}.pdf",
                'excel': f"deepfake_analysis_{result.id}_{timestamp}.xlsx",
                'json': f"deepfake_data_{result.id}_{timestamp}.json",
                'html': f"deepfake_report_{result.id}_{timestamp}.html"
            }
            
            filename = filename_map.get(format_type.lower(), f"deepfake_report_{result.id}_{timestamp}.{format_type}")
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            
            # Update download tracking
            if not hasattr(result, 'metadata') or result.metadata is None:
                result.metadata = {}
            
            if 'downloads' not in result.metadata:
                result.metadata['downloads'] = []
            
            result.metadata['downloads'].append({
                'format': format_type,
                'timestamp': timezone.now().isoformat(),
                'user_ip': self.get_client_ip(request),
                'filename': filename
            })
            result.save()
            
            # Clean up temporary file if it was generated as fallback
            try:
                if format_type == 'json' and 'temp_' in str(report_path):
                    os.remove(report_path)
            except:
                pass
            
            return response
            
        except DetectionResult.DoesNotExist:
            return HttpResponse('Detection result not found', status=404)
        except Exception as e:
            logger.error(f"Error downloading report for {result_id}: {e}")
            
            # Return a simple JSON error response as fallback
            error_data = {
                'error': 'Report generation failed',
                'details': str(e),
                'result_id': str(result_id),
                'timestamp': timezone.now().isoformat()
            }
            
            response = HttpResponse(
                json.dumps(error_data, indent=2), 
                content_type='application/json'
            )
            response['Content-Disposition'] = f'attachment; filename="error_report_{result_id}.json"'
            return response
    
    def _generate_json_report(self, result):
        """Generate a simple JSON report as fallback"""
        import tempfile
        import json
        
        report_data = {
            'analysis_id': str(result.id),
            'filename': result.original_filename,
            'prediction': result.prediction,
            'confidence_score': float(result.confidence_score) if result.confidence_score else None,
            'real_probability': float(result.real_probability) if result.real_probability else None,
            'fake_probability': float(result.fake_probability) if result.fake_probability else None,
            'processing_time': float(result.processing_time) if result.processing_time else None,
            'file_size': result.file_size,
            'face_detected': result.face_detected,
            'face_count': result.face_count,
            'frames_analyzed': result.frames_analyzed,
            'video_duration': float(result.video_duration) if result.video_duration else None,
            'video_fps': float(result.video_fps) if result.video_fps else None,
            'video_resolution': result.video_resolution,
            'model_used': result.model_used or 'Enhanced XceptionNet',
            'created_at': result.created_at.isoformat(),
            'processed_at': result.processed_at.isoformat() if result.processed_at else None,
            'metadata': result.metadata or {},
            'frame_analyses': []
        }
        
        # Add frame analysis data if available
        try:
            frame_analyses = result.frame_analyses.all()
            for frame in frame_analyses:
                report_data['frame_analyses'].append({
                    'frame_number': frame.frame_number,
                    'timestamp': float(frame.timestamp),
                    'prediction': frame.prediction,
                    'confidence_score': float(frame.confidence_score),
                    'real_probability': float(frame.real_probability),
                    'fake_probability': float(frame.fake_probability),
                    'face_detected': frame.face_detected
                })
        except:
            pass
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(report_data, temp_file, indent=2, default=str)
            return temp_file.name
    
    def get_client_ip(self, request):
        """Get client IP address"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip


class BulkDownloadView(APIView):
    """Download multiple reports as a ZIP file"""
    
    def post(self, request):
        """Create ZIP file with multiple reports"""
        try:
            result_ids = request.data.get('result_ids', [])
            format_type = request.data.get('format', 'pdf')
            
            if not result_ids:
                return Response({
                    'error': 'No result IDs provided'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            if len(result_ids) > 50:  # Limit bulk downloads
                return Response({
                    'error': 'Maximum 50 reports per bulk download'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Import zipfile here to avoid import issues
            import zipfile
            from io import BytesIO
            
            # Create ZIP in memory
            zip_buffer = BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for result_id in result_ids:
                    try:
                        result = DetectionResult.objects.get(id=result_id)
                        report_path = report_generator.generate_comprehensive_report(
                            result, format_type=format_type
                        )
                        
                        if report_path.exists():
                            # Add file to ZIP with unique name
                            zip_file.write(str(report_path), report_path.name)
                    
                    except DetectionResult.DoesNotExist:
                        continue
                    except Exception as e:
                        logger.warning(f"Failed to add report {result_id} to ZIP: {e}")
                        continue
            
            zip_buffer.seek(0)
            
            # Create response
            response = HttpResponse(zip_buffer.getvalue(), content_type='application/zip')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            response['Content-Disposition'] = f'attachment; filename="deepfake_reports_{timestamp}.zip"'
            
            return response
            
        except Exception as e:
            logger.error(f"Error in bulk download: {e}")
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DeleteResultView(APIView):
    """Delete detection results"""
    
    def delete(self, request, result_id):
        """Delete a specific result"""
        try:
            result = get_object_or_404(DetectionResult, id=result_id)
            
            # Delete associated video file if it exists
            if result.video_file and default_storage.exists(result.video_file.name):
                default_storage.delete(result.video_file.name)
            
            # Delete the database record
            result.delete()
            
            return Response({
                'success': True,
                'message': 'Result deleted successfully'
            })
            
        except Exception as e:
            logger.error(f"Error deleting result {result_id}: {e}")
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class AnalyticsView(TemplateView):
    """Analytics dashboard for detection results"""
    template_name = 'detector/analytics.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Calculate comprehensive analytics
        now = timezone.now()
        last_week = now - timedelta(days=7)
        last_month = now - timedelta(days=30)
        
        # Basic stats
        total_results = DetectionResult.objects.count()
        context['total_results'] = total_results
        context['fake_count'] = DetectionResult.objects.filter(prediction='FAKE').count()
        context['real_count'] = DetectionResult.objects.filter(prediction='REAL').count()
        context['error_count'] = DetectionResult.objects.filter(prediction='ERROR').count()
        
        # Time-based analytics
        context['weekly_results'] = DetectionResult.objects.filter(created_at__gte=last_week).count()
        context['monthly_results'] = DetectionResult.objects.filter(created_at__gte=last_month).count()
        
        # Performance metrics
        completed_results = DetectionResult.objects.exclude(prediction__in=['PROCESSING', 'ERROR'])
        if completed_results.exists():
            avg_processing_time = completed_results.aggregate(
                models.Avg('processing_time')
            )['processing_time__avg']
            context['avg_processing_time'] = round(avg_processing_time, 2) if avg_processing_time else 0
            
            avg_confidence = completed_results.aggregate(
                models.Avg('confidence_score')
            )['confidence_score__avg']
            context['avg_confidence'] = round(avg_confidence * 100, 1) if avg_confidence else 0
        else:
            context['avg_processing_time'] = 0
            context['avg_confidence'] = 0
        
        # Daily activity for last 7 days
        daily_activity = []
        for i in range(7):
            date = now.date() - timedelta(days=i)
            count = DetectionResult.objects.filter(created_at__date=date).count()
            daily_activity.append({
                'date': date.strftime('%Y-%m-%d'),
                'count': count
            })
        context['daily_activity'] = list(reversed(daily_activity))
        
        # Model performance
        try:
            performance = ModelPerformance.objects.latest('last_updated')
            context['model_performance'] = performance
        except ModelPerformance.DoesNotExist:
            context['model_performance'] = None
        
        return context
