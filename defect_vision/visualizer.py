"""
Production QA Visualizer and Heads-Up Display (HUD) Module.
Renders annotated inspection frames, contour overlays, and telemetry dashboards.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from defect_vision.defect_engine import InspectionResult, DefectMetrics


class QualityVisualizer:
    """Renders professional QA overlays and diagnostic inspection telemetry."""

    COLOR_PASS = (46, 204, 113)     # Vibrant Green
    COLOR_FAIL = (41, 41, 231)      # Vibrant Red
    COLOR_WARNING = (39, 174, 245)  # Amber / Yellow
    COLOR_CRACK = (0, 0, 255)       # Red
    COLOR_SCRATCH = (0, 140, 255)   # Orange
    COLOR_VOID = (200, 100, 0)      # Deep Blue/Amber
    COLOR_BLEMISH = (0, 215, 255)   # Gold / Yellow

    def __init__(self, font_scale: float = 0.55, thickness: int = 2):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.thickness = thickness

    def get_defect_color(self, defect_type: str) -> Tuple[int, int, int]:
        """Returns standard BGR color mapping for flaw classification."""
        if defect_type == "CRACK":
            return self.COLOR_CRACK
        elif defect_type == "SCRATCH":
            return self.COLOR_SCRATCH
        elif defect_type == "SURFACE_VOID":
            return self.COLOR_VOID
        return self.COLOR_BLEMISH

    def annotate_frame(
        self,
        base_image: np.ndarray,
        result: InspectionResult,
        show_contours: bool = True,
        show_labels: bool = True,
        show_hud: bool = True
    ) -> np.ndarray:
        """
        Draws defect bounding boxes, contour boundaries, and HUD telemetry onto the image.
        """
        annotated = base_image.copy()
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        # Draw individual defects
        for d in result.defects:
            color = self.get_defect_color(d.defect_type)
            x, y, w, h = d.bbox
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, self.thickness)
            
            # Draw contour
            if show_contours and d.contour is not None:
                cv2.drawContours(annotated, [d.contour], -1, color, 1)

            # Draw defect tag label
            if show_labels:
                label = f"#{d.defect_id} {d.defect_type} ({d.severity_score:.0f}%)"
                (tw, th), baseline = cv2.getTextSize(label, self.font, 0.45, 1)
                
                # Label badge background
                badge_y = max(th + 4, y - 4)
                cv2.rectangle(
                    annotated,
                    (x, badge_y - th - 4),
                    (x + tw + 6, badge_y + 2),
                    color,
                    -1
                )
                cv2.putText(
                    annotated,
                    label,
                    (x + 3, badge_y - 2),
                    self.font,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        if show_hud:
            annotated = self.render_hud(annotated, result)

        return annotated

    def render_hud(self, image: np.ndarray, result: InspectionResult) -> np.ndarray:
        """Renders industrial inspection HUD banner at the top of the frame."""
        h, w = image.shape[:2]
        hud_height = 54
        
        # Create dark overlay bar
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_height), (18, 24, 32), -1)
        image = cv2.addWeighted(overlay, 0.88, image, 0.12, 0)
        
        # Border separator line
        cv2.line(image, (0, hud_height), (w, hud_height), (60, 75, 95), 1)

        # Status badge
        status = result.status
        if status == "PASS":
            status_color = self.COLOR_PASS
        elif status == "FAIL":
            status_color = self.COLOR_FAIL
        else:
            status_color = self.COLOR_WARNING

        # Badge pill
        cv2.rectangle(image, (12, 10), (115, 44), status_color, -1)
        cv2.putText(
            image,
            status,
            (28, 33),
            self.font,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Telemetry metrics
        fps = 1000.0 / max(0.1, result.processing_time_ms)
        text_line = (
            f"Defects: {result.defect_count}  |  "
            f"Flaw Area: {result.defect_area_ratio_pct:.3f}%  |  "
            f"Latency: {result.processing_time_ms:.1f}ms ({fps:.1f} FPS)"
        )
        cv2.putText(
            image,
            text_line,
            (130, 33),
            self.font,
            0.50,
            (240, 245, 250),
            1,
            cv2.LINE_AA
        )

        return image

    def create_diagnostic_quad(
        self,
        original: np.ndarray,
        preprocessed: np.ndarray,
        binary_mask: np.ndarray,
        annotated: np.ndarray
    ) -> np.ndarray:
        """
        Constructs a 2x2 diagnostic quad panel for deep inspection analysis:
        [ Raw Input ]        [ Preprocessed Grayscale ]
        [ Binary Defect Mask ] [ Final Annotated Output ]
        """
        def ensure_bgr(img):
            if len(img.shape) == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img

        p1 = ensure_bgr(original)
        p2 = ensure_bgr(preprocessed)
        p3 = ensure_bgr(binary_mask)
        p4 = ensure_bgr(annotated)

        # Standardize target panel size
        target_h, target_w = 360, 480
        p1 = cv2.resize(p1, (target_w, target_h))
        p2 = cv2.resize(p2, (target_w, target_h))
        p3 = cv2.resize(p3, (target_w, target_h))
        p4 = cv2.resize(p4, (target_w, target_h))

        def label_panel(img, label):
            cv2.rectangle(img, (0, 0), (target_w, 28), (20, 25, 35), -1)
            cv2.putText(img, label, (10, 19), self.font, 0.50, (220, 230, 240), 1, cv2.LINE_AA)
            return img

        p1 = label_panel(p1, "1. Raw Surface Input")
        p2 = label_panel(p2, "2. CLAHE & Bilateral Equalization")
        p3 = label_panel(p3, "3. Morphological Defect Mask")
        p4 = label_panel(p4, "4. Geometric QA Determination")

        top_row = np.hstack([p1, p2])
        bottom_row = np.hstack([p3, p4])
        quad = np.vstack([top_row, bottom_row])
        return quad
