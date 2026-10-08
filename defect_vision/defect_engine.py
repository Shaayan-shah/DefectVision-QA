"""
Core Defect Detection, Morphology Segmentation, and Tolerance Evaluation Engine.
Performs geometric contour classification and production QA pass/fail gating.
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
from defect_vision.preprocessing import SurfacePreprocessor


@dataclass
class DefectMetrics:
    """Individual segmented defect geometry and classification metrics."""
    defect_id: int
    defect_type: str
    bbox: Tuple[int, int, int, int]
    area: float
    perimeter: float
    aspect_ratio: float
    solidity: float
    severity_score: float
    centroid: Tuple[int, int]
    contour: np.ndarray

    def to_dict(self) -> dict:
        return {
            "defect_id": self.defect_id,
            "defect_type": self.defect_type,
            "bbox": {
                "x": self.bbox[0],
                "y": self.bbox[1],
                "w": self.bbox[2],
                "h": self.bbox[3]
            },
            "area_px": round(self.area, 2),
            "perimeter_px": round(self.perimeter, 2),
            "aspect_ratio": round(self.aspect_ratio, 2),
            "solidity": round(self.solidity, 3),
            "severity_score": round(self.severity_score, 1),
            "centroid": {"x": self.centroid[0], "y": self.centroid[1]}
        }


@dataclass
class InspectionResult:
    """Consolidated production inspection output and quality determination."""
    status: str
    defects: List[DefectMetrics]
    defect_count: int
    total_defect_area_px: float
    defect_area_ratio_pct: float
    processing_time_ms: float
    rejection_reasons: List[str]
    binary_mask: np.ndarray

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "defect_count": self.defect_count,
            "total_defect_area_px": round(self.total_defect_area_px, 2),
            "defect_area_ratio_pct": round(self.defect_area_ratio_pct, 4),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "rejection_reasons": self.rejection_reasons,
            "defects": [d.to_dict() for d in self.defects]
        }


class DefectDetector:
    """Reference-free morphological and gradient-based surface defect detector."""

    def __init__(
        self,
        min_defect_area: float = 25.0,
        max_defect_area: float = 50000.0,
        morph_kernel_size: int = 3,
        sensitivity: float = 0.5,
        tolerance_max_defects: int = 0,
        tolerance_max_area_pct: float = 0.05,
        border_margin: int = 8,
        critical_classes: Optional[List[str]] = None,
        preprocessor: Optional[SurfacePreprocessor] = None
    ):
        """
        Args:
            min_defect_area: Minimum contour area in pixels to filter out sensor salt noise.
            max_defect_area: Maximum contour area to filter out broad macro boundaries.
            morph_kernel_size: Kernel dimension for gradient and closing operators.
            sensitivity: Detection sensitivity between 0.0 (strict) and 1.0 (sensitive).
            tolerance_max_defects: Maximum permitted defects before flagging FAIL.
            tolerance_max_area_pct: Maximum percentage of surface area occupied by flaws.
            border_margin: Outer margin in pixels excluded from inspection to avoid vignette edge false positives.
            critical_classes: Flaw classes that trigger immediate FAIL regardless of count.
            preprocessor: Custom preprocessor instance or default initialized.
        """
        self.min_defect_area = min_defect_area
        self.max_defect_area = max_defect_area
        self.morph_kernel_size = morph_kernel_size
        self.sensitivity = float(np.clip(sensitivity, 0.05, 0.95))
        self.tolerance_max_defects = tolerance_max_defects
        self.tolerance_max_area_pct = tolerance_max_area_pct
        self.border_margin = border_margin
        self.critical_classes = critical_classes or ["CRACK"]
        self.preprocessor = preprocessor or SurfacePreprocessor()

    def segment_defects(self, preprocessed_gray: np.ndarray) -> np.ndarray:
        """
        Generates a clean binary defect mask using morphological gradient
        with noise-floor thresholding to suppress grain texture.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.morph_kernel_size, self.morph_kernel_size)
        )
        
        # Morphological gradient reveals high-contrast edge discontinuities
        gradient = cv2.morphologyEx(preprocessed_gray, cv2.MORPH_GRADIENT, kernel)
        
        # Calculate statistical properties of the surface gradient
        mean_g = float(np.mean(gradient))
        std_g = float(np.std(gradient))
        
        # Base noise floor: typical metallic grain lies at 15-25 gradient intensity
        # Sensitivity 0.5 -> threshold factor k = 3.5
        # Higher sensitivity lowers k, lower sensitivity increases k
        k = 5.0 - (self.sensitivity * 3.0)
        dynamic_thresh = mean_g + k * std_g
        noise_floor = 28.0 - (self.sensitivity * 10.0)
        effective_thresh = int(max(noise_floor, dynamic_thresh))
        
        # Primary gradient thresholding
        _, thresh_grad = cv2.threshold(gradient, effective_thresh, 255, cv2.THRESH_BINARY)
        
        # Morphological close to bridge fractured hairline fissures
        closed = cv2.morphologyEx(thresh_grad, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Suppress camera sensor boundary margin artifacts
        if self.border_margin > 0:
            m = self.border_margin
            closed[:m, :] = 0
            closed[-m:, :] = 0
            closed[:, :m] = 0
            closed[:, -m:] = 0

        return closed

    def classify_defect(
        self,
        contour: np.ndarray,
        area: float,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[str, float, float, float, Tuple[int, int]]:
        """
        Classifies defect type using rotation-invariant minimum bounding rectangle,
        solidity, and contour tortuosity.
        
        Returns:
            Tuple of (defect_type, severity_score, aspect_ratio, solidity, centroid)
        """
        x, y, w, h = bbox
        perimeter = cv2.arcLength(contour, True)
        
        # Use rotation-invariant minimum area rectangle for true physical elongation
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        if min(rw, rh) > 0:
            aspect_ratio = max(rw, rh) / min(rw, rh)
        else:
            aspect_ratio = max(w, h) / max(1.0, float(min(w, h)))
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(1.0, hull_area)
        
        # Centroid
        m = cv2.moments(contour)
        if m["m00"] != 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        # Classification heuristics
        if aspect_ratio >= 4.0 and solidity < 0.70:
            defect_type = "CRACK"
            base_severity = 85.0
        elif aspect_ratio >= 2.5:
            defect_type = "SCRATCH"
            base_severity = 65.0
        elif aspect_ratio <= 2.0 and solidity >= 0.70:
            defect_type = "SURFACE_VOID"
            base_severity = 70.0
        else:
            defect_type = "BLEMISH"
            base_severity = 40.0

        # Scale severity based on relative area and elongation
        area_weight = min(25.0, (area / 500.0) * 10.0)
        severity = min(100.0, base_severity + area_weight)
        
        return defect_type, severity, aspect_ratio, solidity, (cx, cy)

    def evaluate_tolerances(
        self,
        defects: List[DefectMetrics],
        total_surface_pixels: int
    ) -> Tuple[str, List[str], float, float]:
        """
        Applies automated pass/fail tolerance rules against extracted defects.
        """
        defect_count = len(defects)
        total_defect_area = sum(d.area for d in defects)
        defect_area_ratio_pct = (total_defect_area / max(1, total_surface_pixels)) * 100.0

        rejection_reasons = []

        # Check critical defect classes
        for d in defects:
            if d.defect_type in self.critical_classes:
                rejection_reasons.append(
                    f"Critical defect identified: {d.defect_type} (ID: {d.defect_id}, Severity: {d.severity_score:.1f})"
                )

        # Check total count threshold
        if defect_count > self.tolerance_max_defects:
            rejection_reasons.append(
                f"Defect count {defect_count} exceeds allowable tolerance ({self.tolerance_max_defects})"
            )

        # Check total area percentage threshold
        if defect_area_ratio_pct > self.tolerance_max_area_pct:
            rejection_reasons.append(
                f"Total defect area {defect_area_ratio_pct:.3f}% exceeds allowable limit ({self.tolerance_max_area_pct:.3f}%)"
            )

        if len(rejection_reasons) > 0:
            status = "FAIL"
        elif defect_count > 0:
            status = "WARNING"
        else:
            status = "PASS"

        return status, rejection_reasons, total_defect_area, defect_area_ratio_pct

    def inspect(self, image: np.ndarray) -> Tuple[InspectionResult, np.ndarray, np.ndarray]:
        """
        Executes end-to-end inspection on an input image.
        
        Returns:
            Tuple of (InspectionResult, preprocessed_image, binary_mask)
        """
        start_time = time.perf_counter()
        color_base, preprocessed_gray = self.preprocessor.process(image)
        
        binary_mask = self.segment_defects(preprocessed_gray)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        defects: List[DefectMetrics] = []
        defect_id = 1
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_defect_area <= area <= self.max_defect_area:
                bbox = cv2.boundingRect(cnt)
                perimeter = cv2.arcLength(cnt, True)
                dtype, severity, ar, solidity, centroid = self.classify_defect(cnt, area, bbox)
                
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

        total_pixels = preprocessed_gray.shape[0] * preprocessed_gray.shape[1]
        status, reasons, total_area, area_pct = self.evaluate_tolerances(defects, total_pixels)
        
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        result = InspectionResult(
            status=status,
            defects=defects,
            defect_count=len(defects),
            total_defect_area_px=total_area,
            defect_area_ratio_pct=area_pct,
            processing_time_ms=latency_ms,
            rejection_reasons=reasons,
            binary_mask=binary_mask
        )

        return result, color_base, binary_mask
