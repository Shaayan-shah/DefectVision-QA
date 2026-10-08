"""
CLI Inspection Runner for DefectVision-QA.
Inspects single image files or batch directories and exports structured QA reports.
"""

import os
import sys
import argparse
import cv2
from typing import List

from defect_vision.preprocessing import SurfacePreprocessor
from defect_vision.defect_engine import DefectDetector
from defect_vision.golden_comparator import GoldenTemplateComparator
from defect_vision.visualizer import QualityVisualizer
from defect_vision.reporter import QualityAuditReporter


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="DefectVision-QA: Automated Surface Defect Inspection Runner"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to a single inspection image file"
    )
    parser.add_argument(
        "--dir",
        type=str,
        help="Directory of images for batch inspection"
    )
    parser.add_argument(
        "--golden",
        type=str,
        default=None,
        help="Optional path to golden template for differential comparison"
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.50,
        help="Detection sensitivity between 0.05 and 0.95 (default: 0.50)"
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=25.0,
        help="Minimum defect contour area in pixels (default: 25.0)"
    )
    parser.add_argument(
        "--max-defects",
        type=int,
        default=0,
        help="Maximum permitted defects before FAIL gating (default: 0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to persist JSON/CSV logs and annotated images"
    )
    parser.add_argument(
        "--diagnostic-quad",
        action="store_true",
        help="Export 2x2 diagnostic quad panel instead of standard overlay"
    )
    return parser.parse_args()


def inspect_file(
    image_path: str,
    detector: DefectDetector,
    comparator: GoldenTemplateComparator,
    visualizer: QualityVisualizer,
    reporter: QualityAuditReporter,
    use_quad: bool = False
):
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to decode image at {image_path}")
        return

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Perform inspection
    if comparator is not None:
        result, processed_ref, mask = comparator.compare(image)
    else:
        result, processed_ref, mask = detector.inspect(image)

    # Visualization
    if use_quad:
        annotated_no_hud = visualizer.annotate_frame(image, result, show_hud=False)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        final_image = visualizer.create_diagnostic_quad(image, gray, mask, annotated_no_hud)
    else:
        final_image = visualizer.annotate_frame(image, result, show_hud=True)

    # Export audit records
    record = reporter.export_record(
        serial_id=base_name,
        result=result,
        annotated_image=final_image
    )

    # Print summary
    print(f"\n--- Inspection: {base_name} ---")
    print(f"Status:          [{result.status}]")
    print(f"Defect Count:    {result.defect_count}")
    print(f"Flaw Area:       {result.total_defect_area_px:.1f} px ({result.defect_area_ratio_pct:.4f}%)")
    print(f"Processing Time: {result.processing_time_ms:.2f} ms")
    if result.rejection_reasons:
        print("Rejection Details:")
        for r in result.rejection_reasons:
            print(f"  * {r}")
    if result.defects:
        print("Defect Manifest:")
        for d in result.defects:
            print(
                f"  - Flaw #{d.defect_id}: {d.defect_type:<12} "
                f"Severity: {d.severity_score:.1f}% | "
                f"Area: {d.area:.1f}px | AR: {d.aspect_ratio:.2f}"
            )
    print(f"Audit Record:    {record['artifacts']['json_path']}")


def main():
    args = parse_arguments()

    if not args.image and not args.dir:
        print("Error: Specify either --image <path> or --dir <folder>")
        sys.exit(1)

    preprocessor = SurfacePreprocessor()
    detector = DefectDetector(
        min_defect_area=args.min_area,
        sensitivity=args.sensitivity,
        tolerance_max_defects=args.max_defects,
        preprocessor=preprocessor
    )

    comparator = None
    if args.golden:
        if not os.path.exists(args.golden):
            print(f"Error: Golden template not found at {args.golden}")
            sys.exit(1)
        golden_img = cv2.imread(args.golden)
        comparator = GoldenTemplateComparator(
            golden_template=golden_img,
            min_defect_area=args.min_area,
            tolerance_max_defects=args.max_defects,
            preprocessor=preprocessor
        )
        print(f"Initialized Golden-Template Comparative Mode with reference: {args.golden}")
    else:
        print("Initialized Reference-Free Morphological Gradient Inspection Mode")

    visualizer = QualityVisualizer()
    reporter = QualityAuditReporter(output_dir=args.output_dir)

    target_files: List[str] = []
    if args.image:
        target_files.append(args.image)
    elif args.dir:
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        for root, _, files in os.walk(args.dir):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in valid_exts:
                    target_files.append(os.path.join(root, f))

    print(f"Processing {len(target_files)} target inspection file(s)...")
    for f in target_files:
        inspect_file(f, detector, comparator, visualizer, reporter, args.diagnostic_quad)

    print(f"\nAll inspections completed. Batch CSV log updated at: {reporter.csv_path}")


if __name__ == "__main__":
    main()
