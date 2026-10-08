"""
Quality Audit and Telemetry Reporting Module.
Exports structured JSON logs, CSV batch summaries, and audit trail artifacts.
"""

import os
import json
import csv
import cv2
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from defect_vision.defect_engine import InspectionResult


class QualityAuditReporter:
    """Manages persistent logging of production inspection results and compliance reports."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        self.json_dir = os.path.join(output_dir, "json")
        self.csv_path = os.path.join(output_dir, "production_audit_log.csv")
        self.images_dir = os.path.join(output_dir, "inspected_images")

        os.makedirs(self.json_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        self._init_csv_log()

    def _init_csv_log(self) -> None:
        """Initializes cumulative CSV file with header schema if not already present."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp",
                    "Serial_ID",
                    "Status",
                    "Defect_Count",
                    "Total_Defect_Area_px",
                    "Defect_Area_Ratio_pct",
                    "Processing_Time_ms",
                    "Rejection_Reasons"
                ])

    def export_record(
        self,
        serial_id: str,
        result: InspectionResult,
        annotated_image: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Persists inspection telemetry to JSON, appends to CSV log,
        and optionally saves the annotated proof image.
        """
        now = datetime.now(timezone.utc)
        timestamp_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        timestamp_slug = now.strftime("%Y%m%d_%H%M%S")
        
        record = {
            "serial_id": serial_id,
            "timestamp_utc": timestamp_iso,
            "summary": result.to_dict()
        }

        # 1. Save detailed JSON audit file
        json_filename = f"{serial_id}_{timestamp_slug}.json"
        json_filepath = os.path.join(self.json_dir, json_filename)
        with open(json_filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        # 2. Append to cumulative CSV audit log
        reasons_str = "; ".join(result.rejection_reasons) if result.rejection_reasons else "NONE"
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp_iso,
                serial_id,
                result.status,
                result.defect_count,
                round(result.total_defect_area_px, 2),
                round(result.defect_area_ratio_pct, 4),
                round(result.processing_time_ms, 2),
                reasons_str
            ])

        # 3. Save annotated image if provided
        saved_img_path = None
        if annotated_image is not None:
            img_filename = f"{serial_id}_{result.status}_{timestamp_slug}.png"
            saved_img_path = os.path.join(self.images_dir, img_filename)
            cv2.imwrite(saved_img_path, annotated_image)

        record["artifacts"] = {
            "json_path": json_filepath,
            "csv_log": self.csv_path,
            "image_path": saved_img_path
        }

        return record
