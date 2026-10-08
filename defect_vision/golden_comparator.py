"""
Golden Template Differential Inspection Module.
Performs sub-pixel feature alignment and structural subtraction against a nominal master sample.
"""

import cv2
import numpy as np
import time
from typing import Tuple, List, Optional
from defect_vision.defect_engine import DefectMetrics, InspectionResult, DefectDetector
from defect_vision.preprocessing import SurfacePreprocessor


class GoldenTemplateComparator:
    """Detects deviations between an inspection sample and an authenticated reference template."""

    def __init__(
        self,
        golden_template: np.ndarray,
        diff_threshold: int = 35,
        min_defect_area: float = 25.0,
        tolerance_max_defects: int = 0,
        preprocessor: Optional[SurfacePreprocessor] = None
    ):
        """
        Args:
            golden_template: Ideal nominal component image.
            diff_threshold: Intensity difference threshold (0-255) for anomaly classification.
            min_defect_area: Minimum connected component area in pixels to flag.
            tolerance_max_defects: Maximum permitted defects before flagging FAIL.
            preprocessor: Custom preprocessor or default.
        """
        self.preprocessor = preprocessor or SurfacePreprocessor()
        self.diff_threshold = diff_threshold
        self.min_defect_area = min_defect_area
        self.tolerance_max_defects = tolerance_max_defects
        
        # Store preprocessed golden template
        _, self.golden_gray = self.preprocessor.process(golden_template)
        self.golden_color = golden_template.copy()
        
        # Feature detector for homography alignment
        self.orb = cv2.ORB_create(nfeatures=1500)
        self.kp_golden, self.des_golden = self.orb.detectAndCompute(self.golden_gray, None)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def align_sample(self, sample_gray: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Aligns incoming sample to golden template space using ORB feature homography.
        Falls back to direct geometry if feature count is insufficient.
        """
        kp_sample, des_sample = self.orb.detectAndCompute(sample_gray, None)
        if des_sample is None or len(des_sample) < 10 or self.des_golden is None or len(self.des_golden) < 10:
            # Fallback: simple resize/crop alignment
            aligned = cv2.resize(sample_gray, (self.golden_gray.shape[1], self.golden_gray.shape[0]))
            return aligned, False

        matches = self.matcher.match(des_sample, self.des_golden)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 8:
            aligned = cv2.resize(sample_gray, (self.golden_gray.shape[1], self.golden_gray.shape[0]))
            return aligned, False

        pts_sample = np.float32([kp_sample[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
        pts_golden = np.float32([self.kp_golden[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(pts_sample, pts_golden, cv2.RANSAC, 5.0)
        if homography is None:
            aligned = cv2.resize(sample_gray, (self.golden_gray.shape[1], self.golden_gray.shape[0]))
            return aligned, False

        h, w = self.golden_gray.shape
        aligned = cv2.warpPerspective(sample_gray, homography, (w, h))
        return aligned, True

    def compare(self, sample_image: np.ndarray) -> Tuple[InspectionResult, np.ndarray, np.ndarray]:
        """
        Executes golden-sample differential analysis.
        
        Returns:
            Tuple of (InspectionResult, aligned_sample, difference_mask)
        """
        start_time = time.perf_counter()
        color_base, sample_gray = self.preprocessor.process(sample_image)
        aligned_gray, aligned_success = self.align_sample(sample_gray)

        # Absolute structural difference
        diff = cv2.absdiff(aligned_gray, self.golden_gray)
        
        # Binary thresholding on difference map
        _, diff_mask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        
        # Clean small spatial alignment jitter
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        defects: List[DefectMetrics] = []
        defect_id = 1
        detector = DefectDetector(min_defect_area=self.min_defect_area)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_defect_area:
                bbox = cv2.boundingRect(cnt)
                perimeter = cv2.arcLength(cnt, True)
                dtype, severity, ar, solidity, centroid = detector.classify_defect(cnt, area, bbox)
                
                defect = DefectMetrics(
                    defect_id=defect_id,
                    defect_type=dtype,
                    bbox=bbox,
                    area=area,
                    perimeter=perimeter,
                    aspect_ratio=ar,
                    solidity=solidity,
                    severity_score=severity,
                    centroid=centroid,
                    contour=cnt
                )
                defects.append(defect)
                defect_id += 1

        total_pixels = self.golden_gray.shape[0] * self.golden_gray.shape[1]
        status, reasons, total_area, area_pct = detector.evaluate_tolerances(defects, total_pixels)
        
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        result = InspectionResult(
            status=status,
            defects=defects,
            defect_count=len(defects),
            total_defect_area_px=total_area,
            defect_area_ratio_pct=area_pct,
            processing_time_ms=latency_ms,
            rejection_reasons=reasons,
            binary_mask=cleaned_mask
        )

        return result, aligned_gray, cleaned_mask
