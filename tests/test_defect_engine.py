"""
Unit and Integration Tests for DefectVision-QA Pipeline.
Verifies preprocessing, reference-free defect detection, golden comparator, and audit reporting.
"""

import os
import json
import unittest
import numpy as np
import cv2
import shutil

from defect_vision.preprocessing import SurfacePreprocessor
from defect_vision.defect_engine import DefectDetector, InspectionResult
from defect_vision.golden_comparator import GoldenTemplateComparator
from defect_vision.visualizer import QualityVisualizer
from defect_vision.reporter import QualityAuditReporter
from samples.generate_synthetic_samples import (
    create_base_metallic_surface,
    add_hairline_crack,
    add_abrasion_scratch,
    add_pitting_voids
)


class TestDefectVisionPipeline(unittest.TestCase):
    """Rigorous verification suite for defect detection algorithms."""

    @classmethod
    def setUpClass(cls):
        cls.test_output_dir = "tests/test_output"
        os.makedirs(cls.test_output_dir, exist_ok=True)
        
        # Pre-synthesize test targets
        cls.clean_surface = create_base_metallic_surface(640, 480)
        cls.crack_surface = add_hairline_crack(create_base_metallic_surface(640, 480), (200, 240), segments=7)
        cls.scratch_surface = add_abrasion_scratch(create_base_metallic_surface(640, 480), (100, 100), (450, 350))
        cls.void_surface = add_pitting_voids(create_base_metallic_surface(640, 480), [(300, 200, 8), (350, 250, 6)])

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_output_dir):
            shutil.rmtree(cls.test_output_dir, ignore_errors=True)

    def test_01_preprocessor(self):
        """Verifies preprocessing grayscale conversion, CLAHE equalization, and dimensions."""
        preprocessor = SurfacePreprocessor(target_resolution=(640, 480))
        color_base, preprocessed_gray = preprocessor.process(self.clean_surface)
        
        self.assertEqual(color_base.shape, (480, 640, 3))
        self.assertEqual(preprocessed_gray.shape, (480, 640))
        self.assertEqual(preprocessed_gray.dtype, np.uint8)
        
        # Mean intensity should be well-centered around dynamic range
        mean_val = np.mean(preprocessed_gray)
        self.assertTrue(100 <= mean_val <= 200)

    def test_02_pristine_surface_passes(self):
        """Verifies that an unblemished nominal substrate evaluates to PASS with 0 defects."""
        detector = DefectDetector(min_defect_area=30.0, sensitivity=0.45, tolerance_max_defects=0)
        result, _, binary_mask = detector.inspect(self.clean_surface)
        
        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.defect_count, 0)
        self.assertEqual(len(result.rejection_reasons), 0)
        self.assertLess(result.total_defect_area_px, 1.0)
        self.assertLess(result.defect_area_ratio_pct, 0.001)

    def test_03_crack_detection_and_classification(self):
        """Verifies detection and classification of hairline fracture as CRACK with FAIL status."""
        detector = DefectDetector(min_defect_area=25.0, sensitivity=0.60, tolerance_max_defects=0)
        result, _, _ = detector.inspect(self.crack_surface)
        
        self.assertEqual(result.status, "FAIL")
        self.assertGreater(result.defect_count, 0)
        
        # Verify that at least one detected defect is classified as CRACK or SCRATCH
        types = [d.defect_type for d in result.defects]
        self.assertTrue("CRACK" in types or "SCRATCH" in types)
        
        # Verify rejection reason recorded
        self.assertGreater(len(result.rejection_reasons), 0)

    def test_04_scratch_detection_and_classification(self):
        """Verifies detection of linear abrasive scratch with high aspect ratio."""
        detector = DefectDetector(min_defect_area=25.0, sensitivity=0.60, tolerance_max_defects=0)
        result, _, _ = detector.inspect(self.scratch_surface)
        
        self.assertEqual(result.status, "FAIL")
        self.assertGreater(result.defect_count, 0)
        
        # Check aspect ratio property of the scratch
        max_ar = max(d.aspect_ratio for d in result.defects)
        self.assertGreaterEqual(max_ar, 2.5)

    def test_05_void_defect_detection(self):
        """Verifies detection of circular pitting cavities / casting voids."""
        detector = DefectDetector(min_defect_area=20.0, sensitivity=0.60, tolerance_max_defects=0)
        result, _, _ = detector.inspect(self.void_surface)
        
        self.assertEqual(result.status, "FAIL")
        self.assertGreater(result.defect_count, 0)

    def test_06_golden_template_comparator(self):
        """Verifies differential comparative inspection against authenticated master reference."""
        golden = self.clean_surface
        comparator = GoldenTemplateComparator(golden, diff_threshold=30, min_defect_area=20.0)
        
        # Comparing clean against clean should pass
        res_clean, _, diff_mask = comparator.compare(self.clean_surface)
        self.assertEqual(res_clean.status, "PASS")
        self.assertEqual(res_clean.defect_count, 0)
        
        # Comparing crack against clean should fail
        res_flaw, _, _ = comparator.compare(self.crack_surface)
        self.assertEqual(res_flaw.status, "FAIL")
        self.assertGreater(res_flaw.defect_count, 0)

    def test_07_visualizer_annotation(self):
        """Verifies HUD overlay and diagnostic quad rendering."""
        detector = DefectDetector()
        result, color_base, binary_mask = detector.inspect(self.crack_surface)
        
        visualizer = QualityVisualizer()
        annotated = visualizer.annotate_frame(color_base, result)
        
        self.assertEqual(annotated.shape, color_base.shape)
        
        # Diagnostic quad
        preprocessed = cv2.cvtColor(color_base, cv2.COLOR_BGR2GRAY)
        quad = visualizer.create_diagnostic_quad(color_base, preprocessed, binary_mask, annotated)
        self.assertEqual(quad.shape, (720, 960, 3))

    def test_08_reporter_logging(self):
        """Verifies JSON and CSV persistent audit trail generation."""
        detector = DefectDetector()
        result, color_base, _ = detector.inspect(self.crack_surface)
        
        reporter = QualityAuditReporter(output_dir=self.test_output_dir)
        record = reporter.export_record("TEST_PART_001", result, annotated_image=color_base)
        
        self.assertTrue(os.path.exists(record["artifacts"]["json_path"]))
        self.assertTrue(os.path.exists(record["artifacts"]["csv_log"]))
        self.assertTrue(os.path.exists(record["artifacts"]["image_path"]))
        
        # Read back json
        with open(record["artifacts"]["json_path"], "r") as f:
            data = json.load(f)
        self.assertEqual(data["serial_id"], "TEST_PART_001")
        self.assertEqual(data["summary"]["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
