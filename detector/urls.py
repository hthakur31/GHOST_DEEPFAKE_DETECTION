from django.urls import path, include
from . import views

app_name = 'detector'

urlpatterns = [
    # Web views
    path('', views.HomeView.as_view(), name='home'),
    path('history/', views.HistoryView.as_view(), name='history'),
    path('result/<uuid:detection_id>/', views.DetectionResultView.as_view(), name='result'),
    path('detailed/<uuid:result_id>/', views.DetailedResultView.as_view(), name='detailed_result'),
    path('analytics/', views.AnalyticsView.as_view(), name='analytics'),
    
    # API endpoints
    path('api/upload/', views.VideoUploadView.as_view(), name='api_upload'),
    path('api/status/<uuid:detection_id>/', views.DetectionStatusAPIView.as_view(), name='api_status'),
    path('api/performance/', views.ModelPerformanceAPIView.as_view(), name='api_performance'),
    path('api/recent/', views.RecentDetectionsAPIView.as_view(), name='api_recent'),
    path('api/stats/', views.StatsAPIView.as_view(), name='api_stats'),
    path('api/delete/', views.DeleteDetectionAPIView.as_view(), name='api_delete'),
    path('api/stop/', views.StopProcessingAPIView.as_view(), name='api_stop'),
    
    # Report download endpoints
    path('api/download/<uuid:result_id>/', views.DownloadReportView.as_view(), name='download_report'),
    path('api/download/<uuid:result_id>/<str:format_type>/', views.DownloadReportView.as_view(), name='download_report_format'),
    path('api/bulk-download/', views.BulkDownloadView.as_view(), name='bulk_download'),
    path('api/delete-result/<uuid:result_id>/', views.DeleteResultView.as_view(), name='delete_result'),
]
