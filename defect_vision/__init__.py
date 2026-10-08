"""
DefectVision-QA
Industrial Surface Defect Detection & Quality Assurance Pipeline
Author: Shayan Shah
"""

from defect_vision.preprocessing import SurfacePreprocessor
from defect_vision.defect_engine import DefectDetector, DefectMetrics, InspectionResult
from defect_vision.golden_comparator import GoldenTemplateComparator
from defect_vision.visualizer import QualityVisualizer
from defect_vision.reporter import QualityAuditReporter

__all__ = [
    "SurfacePreprocessor",
    "DefectDetector",
    "DefectMetrics",
    "InspectionResult",
    "GoldenTemplateComparator",
    "QualityVisualizer",
    "QualityAuditReporter",
]
