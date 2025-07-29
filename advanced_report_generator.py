#!/usr/bin/env python3
"""
Advanced Report Generation System for Deepfake Detection
Supports PDF, JSON, and Excel report formats with comprehensive analysis
"""

import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: ReportLab not available. PDF generation will be disabled.")

class AdvancedReportGenerator:
    """
    Advanced report generator for deepfake detection results
    Supports multiple formats: PDF, JSON, Excel, HTML
    """
    
    def __init__(self):
        self.report_dir = Path("reports")
        self.report_dir.mkdir(exist_ok=True)
        
        # Initialize styles for PDF generation
        if REPORTLAB_AVAILABLE:
            self.styles = getSampleStyleSheet()
            self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom styles for PDF reports"""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        # Subtitle style
        self.subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkgreen
        )
        
        # Summary style
        self.summary_style = ParagraphStyle(
            'Summary',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            leftIndent=20
        )
    
    def generate_comprehensive_report(self, detection_result, format_type='pdf'):
        """
        Generate comprehensive analysis report
        
        Args:
            detection_result: DetectionResult model instance
            format_type: 'pdf', 'json', 'excel', 'html'
        
        Returns:
            Path to generated report file or report data
        """
        # Prepare report data
        report_data = self._prepare_report_data(detection_result)
        
        if format_type.lower() == 'pdf':
            return self._generate_pdf_report(report_data, detection_result)
        elif format_type.lower() == 'json':
            return self._generate_json_report(report_data, detection_result)
        elif format_type.lower() == 'excel':
            return self._generate_excel_report(report_data, detection_result)
        elif format_type.lower() == 'html':
            return self._generate_html_report(report_data, detection_result)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _prepare_report_data(self, detection_result):
        """Prepare comprehensive data for report generation"""
        
        # Basic detection info
        report_data = {
            'analysis_summary': {
                'file_name': detection_result.original_filename,
                'file_size_mb': detection_result.file_size_mb,
                'upload_time': detection_result.created_at.isoformat(),
                'processing_time': detection_result.processing_time,
                'final_prediction': detection_result.prediction,
                'confidence_score': detection_result.confidence_score,
                'confidence_percentage': detection_result.confidence_percentage,
                'is_deepfake': detection_result.is_deepfake,
                'model_used': detection_result.model_used,
                'model_version': detection_result.model_version,
                'detection_method': detection_result.detection_method
            },
            
            'video_metadata': {
                'duration_seconds': detection_result.video_duration,
                'fps': detection_result.video_fps,
                'resolution': detection_result.video_resolution,
                'frames_analyzed': detection_result.frames_analyzed,
                'face_detected': detection_result.face_detected,
                'face_count': detection_result.face_count
            },
            
            'technical_details': {
                'threshold_used': detection_result.threshold_used,
                'temporal_consistency': detection_result.temporal_consistency,
                'real_probability': detection_result.real_probability,
                'fake_probability': detection_result.fake_probability,
                'processing_method': detection_result.detection_method or 'Enhanced XceptionNet'
            },
            
            'frame_analysis': [],
            'statistical_analysis': {},
            'risk_assessment': {},
            'recommendations': []
        }
        
        # Add frame-by-frame analysis if available
        if hasattr(detection_result, 'frame_analyses') and detection_result.frame_analyses.exists():
            for frame in detection_result.frame_analyses.all():
                frame_data = {
                    'frame_number': frame.frame_number,
                    'timestamp': frame.timestamp,
                    'prediction': frame.prediction,
                    'confidence': frame.confidence_score,
                    'real_prob': frame.real_probability,
                    'fake_prob': frame.fake_probability,
                    'face_detected': frame.face_detected
                }
                report_data['frame_analysis'].append(frame_data)
        
        # Enhanced metadata from detection results
        if detection_result.metadata:
            report_data['enhanced_metadata'] = detection_result.metadata
        
        if detection_result.report_data:
            report_data.update(detection_result.report_data)
        
        # Generate statistical analysis
        report_data['statistical_analysis'] = self._generate_statistical_analysis(report_data)
        
        # Generate risk assessment
        report_data['risk_assessment'] = self._generate_risk_assessment(detection_result)
        
        # Generate recommendations
        report_data['recommendations'] = self._generate_recommendations(detection_result)
        
        return report_data
    
    def _generate_statistical_analysis(self, report_data):
        """Generate statistical analysis of the detection"""
        stats = {
            'confidence_distribution': {},
            'temporal_patterns': {},
            'detection_stability': {}
        }
        
        if report_data['frame_analysis']:
            confidences = [f['confidence'] for f in report_data['frame_analysis']]
            predictions = [f['prediction'] for f in report_data['frame_analysis']]
            
            stats['confidence_distribution'] = {
                'mean': sum(confidences) / len(confidences) if confidences else 0,
                'max': max(confidences) if confidences else 0,
                'min': min(confidences) if confidences else 0,
                'std_dev': self._calculate_std_dev(confidences)
            }
            
            stats['detection_stability'] = {
                'consistency_score': len([p for p in predictions if p == report_data['analysis_summary']['final_prediction']]) / len(predictions) if predictions else 0,
                'prediction_changes': len(set(predictions)),
                'stable_prediction': len(set(predictions)) == 1
            }
        
        return stats
    
    def _calculate_std_dev(self, values):
        """Calculate standard deviation"""
        if not values:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _generate_risk_assessment(self, detection_result):
        """Generate risk assessment based on detection results"""
        risk_level = "LOW"
        risk_factors = []
        
        if detection_result.is_deepfake:
            if detection_result.confidence_score and detection_result.confidence_score > 0.8:
                risk_level = "HIGH"
                risk_factors.append("High confidence deepfake detection")
            elif detection_result.confidence_score and detection_result.confidence_score > 0.6:
                risk_level = "MEDIUM"
                risk_factors.append("Medium confidence deepfake detection")
            else:
                risk_level = "LOW"
                risk_factors.append("Low confidence deepfake detection - requires manual review")
        
        if detection_result.temporal_consistency and detection_result.temporal_consistency < 0.7:
            risk_factors.append("Low temporal consistency detected")
        
        if not detection_result.face_detected:
            risk_factors.append("No face detected - analysis may be unreliable")
        
        return {
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'reliability_score': self._calculate_reliability_score(detection_result)
        }
    
    def _calculate_reliability_score(self, detection_result):
        """Calculate reliability score for the detection"""
        score = 0.5  # Base score
        
        if detection_result.face_detected:
            score += 0.2
        
        if detection_result.confidence_score:
            if detection_result.confidence_score > 0.8:
                score += 0.2
            elif detection_result.confidence_score > 0.6:
                score += 0.1
        
        if detection_result.temporal_consistency and detection_result.temporal_consistency > 0.8:
            score += 0.1
        
        return min(1.0, score)
    
    def _generate_recommendations(self, detection_result):
        """Generate recommendations based on detection results"""
        recommendations = []
        
        if detection_result.is_deepfake:
            recommendations.append("⚠️ Potential deepfake detected - verify source authenticity")
            recommendations.append("📋 Consider secondary verification with alternative detection methods")
            
            if detection_result.confidence_score and detection_result.confidence_score < 0.7:
                recommendations.append("🔍 Low confidence score - manual expert review recommended")
        
        if not detection_result.face_detected:
            recommendations.append("👤 No face detected - ensure video contains clear facial imagery")
            recommendations.append("📹 Consider reprocessing with higher quality video")
        
        if detection_result.processing_time and detection_result.processing_time > 30:
            recommendations.append("⏱️ Long processing time detected - consider video optimization")
        
        recommendations.append("📊 Save this report for audit trail and documentation")
        recommendations.append("🔄 Regular model updates improve detection accuracy")
        
        return recommendations
    
    def _generate_pdf_report(self, report_data, detection_result):
        """Generate comprehensive PDF report"""
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF generation")
        
        filename = detection_result.generate_report_filename()
        filepath = self.report_dir / filename
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build content
        story = []
        
        # Title
        story.append(Paragraph("🔍 Deepfake Detection Report", self.title_style))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("📋 Executive Summary", self.subtitle_style))
        
        summary_data = [
            ['File Name', report_data['analysis_summary']['file_name']],
            ['Analysis Date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ['Final Prediction', f"<b>{report_data['analysis_summary']['final_prediction']}</b>"],
            ['Confidence Score', f"{report_data['analysis_summary']['confidence_percentage']:.1f}%"],
            ['Risk Level', report_data['risk_assessment']['risk_level']],
            ['Processing Time', f"{report_data['analysis_summary']['processing_time']:.2f}s"]
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Technical Details
        story.append(Paragraph("🔧 Technical Analysis", self.subtitle_style))
        
        tech_data = [
            ['Model Used', report_data['analysis_summary']['model_used']],
            ['Detection Method', report_data['technical_details']['processing_method']],
            ['Threshold Used', f"{report_data['technical_details']['threshold_used']:.2f}" if report_data['technical_details']['threshold_used'] else 'N/A'],
            ['Temporal Consistency', f"{report_data['technical_details']['temporal_consistency']:.2f}" if report_data['technical_details']['temporal_consistency'] else 'N/A'],
            ['Real Probability', f"{report_data['technical_details']['real_probability']:.1f}%" if report_data['technical_details']['real_probability'] else 'N/A'],
            ['Fake Probability', f"{report_data['technical_details']['fake_probability']:.1f}%" if report_data['technical_details']['fake_probability'] else 'N/A']
        ]
        
        tech_table = Table(tech_data, colWidths=[2*inch, 3*inch])
        tech_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(tech_table)
        story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("💡 Recommendations", self.subtitle_style))
        for i, rec in enumerate(report_data['recommendations'], 1):
            story.append(Paragraph(f"{i}. {rec}", self.summary_style))
        
        # Build PDF
        doc.build(story)
        
        return filepath
    
    def _generate_json_report(self, report_data, detection_result):
        """Generate JSON report"""
        filename = f"deepfake_report_{detection_result.id}_detailed.json"
        filepath = self.report_dir / filename
        
        # Add metadata
        report_data['report_metadata'] = {
            'generated_at': datetime.now().isoformat(),
            'report_version': '2.0',
            'generator': 'Advanced Deepfake Detection System',
            'format': 'JSON'
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        return filepath
    
    def _generate_excel_report(self, report_data, detection_result):
        """Generate Excel report with multiple sheets"""
        filename = f"deepfake_report_{detection_result.id}_analysis.xlsx"
        filepath = self.report_dir / filename
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = pd.DataFrame([report_data['analysis_summary']]).T
            summary_df.columns = ['Value']
            summary_df.to_excel(writer, sheet_name='Summary')
            
            # Technical details
            tech_df = pd.DataFrame([report_data['technical_details']]).T
            tech_df.columns = ['Value']
            tech_df.to_excel(writer, sheet_name='Technical Details')
            
            # Frame analysis if available
            if report_data['frame_analysis']:
                frame_df = pd.DataFrame(report_data['frame_analysis'])
                frame_df.to_excel(writer, sheet_name='Frame Analysis', index=False)
            
            # Statistical analysis
            if report_data['statistical_analysis']:
                stats_data = []
                for category, values in report_data['statistical_analysis'].items():
                    if isinstance(values, dict):
                        for key, value in values.items():
                            stats_data.append({'Category': category, 'Metric': key, 'Value': value})
                    else:
                        stats_data.append({'Category': category, 'Metric': 'Value', 'Value': values})
                
                if stats_data:
                    stats_df = pd.DataFrame(stats_data)
                    stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        return filepath
    
    def _generate_html_report(self, report_data, detection_result):
        """Generate HTML report"""
        filename = f"deepfake_report_{detection_result.id}_web.html"
        filepath = self.report_dir / filename
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Deepfake Detection Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .prediction-real {{ background: #d4edda; border-color: #c3e6cb; }}
                .prediction-fake {{ background: #f8d7da; border-color: #f5c6cb; }}
                .table {{ width: 100%; border-collapse: collapse; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .table th {{ background-color: #f2f2f2; }}
                .recommendations {{ background: #e7f3ff; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 Deepfake Detection Report</h1>
                <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="section {'prediction-real' if not detection_result.is_deepfake else 'prediction-fake'}">
                <h2>📋 Analysis Summary</h2>
                <p><strong>File:</strong> {report_data['analysis_summary']['file_name']}</p>
                <p><strong>Prediction:</strong> <strong>{report_data['analysis_summary']['final_prediction']}</strong></p>
                <p><strong>Confidence:</strong> {report_data['analysis_summary']['confidence_percentage']:.1f}%</p>
                <p><strong>Risk Level:</strong> {report_data['risk_assessment']['risk_level']}</p>
            </div>
            
            <div class="section">
                <h2>🔧 Technical Details</h2>
                <table class="table">
                    <tr><th>Model Used</th><td>{report_data['analysis_summary']['model_used']}</td></tr>
                    <tr><th>Processing Time</th><td>{report_data['analysis_summary']['processing_time']:.2f}s</td></tr>
                    <tr><th>Frames Analyzed</th><td>{report_data['video_metadata']['frames_analyzed']}</td></tr>
                    <tr><th>Face Detected</th><td>{'Yes' if report_data['video_metadata']['face_detected'] else 'No'}</td></tr>
                </table>
            </div>
            
            <div class="section recommendations">
                <h2>💡 Recommendations</h2>
                <ul>
                    {''.join(f'<li>{rec}</li>' for rec in report_data['recommendations'])}
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filepath

# Global instance
report_generator = AdvancedReportGenerator()
