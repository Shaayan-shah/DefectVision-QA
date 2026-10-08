"""
Real-Time Video Stream and Industrial Camera Inspection Runner.
Processes continuous frames at 30+ FPS with interactive live HUD and keyboard controls.
"""

import sys
import argparse
import cv2
import time

from defect_vision.preprocessing import SurfacePreprocessor
from defect_vision.defect_engine import DefectDetector
from defect_vision.golden_comparator import GoldenTemplateComparator
from defect_vision.visualizer import QualityVisualizer
from defect_vision.reporter import QualityAuditReporter


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="DefectVision-QA: Live Real-Time Surface Inspection"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera device index (default: 0) or path to input video file"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Capture frame width (default: 640)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Capture frame height (default: 480)"
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.50,
        help="Detection sensitivity between 0.05 and 0.95"
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=25.0,
        help="Minimum defect contour area in pixels"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Parse source as int if webcam index
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Unable to open video source: {args.source}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    preprocessor = SurfacePreprocessor(target_resolution=(args.width, args.height))
    detector = DefectDetector(
        min_defect_area=args.min_area,
        sensitivity=args.sensitivity,
        preprocessor=preprocessor
    )
    visualizer = QualityVisualizer()
    reporter = QualityAuditReporter()

    comparator = None
    use_quad = False
    frame_counter = 0

    print("=======================================================")
    print("DefectVision-QA: Live Real-Time Inspection Activated")
    print("Keyboard Controls:")
    print("  [p] - Capture current frame as Golden Reference Template")
    print("  [m] - Reset/Toggle to Reference-Free Morphological Mode")
    print("  [d] - Toggle 2x2 Diagnostic Quad View")
    print("  [s] - Snapshot & log current frame to Quality Audit Trail")
    print("  [q] - Quit Inspection")
    print("=======================================================")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or camera disconnected.")
            break

        frame_counter += 1

        # Perform inspection
        if comparator is not None:
            result, processed_ref, mask = comparator.compare(frame)
        else:
            result, processed_ref, mask = detector.inspect(frame)

        if use_quad:
            annotated_no_hud = visualizer.annotate_frame(frame, result, show_hud=False)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display_frame = visualizer.create_diagnostic_quad(frame, gray, mask, annotated_no_hud)
        else:
            display_frame = visualizer.annotate_frame(frame, result, show_hud=True)

        # Show interactive window
        window_title = "DefectVision-QA Live Industrial Monitor"
        cv2.imshow(window_title, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            use_quad = not use_quad
            print(f"Diagnostic Quad View: {'ON' if use_quad else 'OFF'}")
        elif key == ord('p'):
            # Set golden reference
            comparator = GoldenTemplateComparator(
                golden_template=frame,
                min_defect_area=args.min_area,
                preprocessor=preprocessor
            )
            print("Captured current frame as active Golden Template.")
        elif key == ord('m'):
            comparator = None
            print("Switched to Reference-Free Morphological Inspection Mode.")
        elif key == ord('s'):
            serial = f"LIVE_CYCLE_{frame_counter:05d}"
            record = reporter.export_record(serial, result, annotated_image=display_frame)
            print(f"Logged Snapshot -> Serial: {serial} | Status: [{result.status}] | Audit: {record['artifacts']['json_path']}")

    cap.release()
    cv2.destroyAllWindows()
    print("Live inspection terminated cleanly.")


if __name__ == "__main__":
    main()
