# DefectVision-QA

**Automated surface-defect detection and quality-assurance pipeline**
Real-time computer vision system for micro-crack detection, abrasive scratch isolation, casting void segmentation, and automated production pass/fail tolerance gating.

---

## 1. Problem Statement & Industrial Relevance

In high-throughput manufacturing, manual inspection can be difficult to scale consistently across products, lighting conditions, and production speeds. Automated inspection can support repeatable review when validated for the target process.

Production facilities require deterministic, low-latency, edge-deployable inspection pipelines that can:
1. Normalize non-uniform lighting conditions and diffuse dome glare across textured surfaces.
2. Differentiate authentic micro-defects (fractures, burrs, pitting, voids) from baseline substrate grain texture.
3. Classify detected flaws into functional engineering categories based on invariant geometric metrics.
4. Evaluate configured PASS / FAIL / WARNING tolerances locally, without relying on external cloud APIs.
5. Generate structured JSON inspection records and cumulative CSV telemetry logs.

DefectVision-QA addresses this demand with an open, modular computer vision pipeline featuring both reference-free morphological gradient analysis and golden-sample differential inspection.

---

## 2. Mathematical Formulation & Algorithmic Pipeline

### 2.1 Illumination Normalization & Edge-Preserving Denoising
CLAHE improves local contrast by applying histogram equalization to image tiles, with clipping used to limit contrast amplification.

$$I_{\mathrm{out}}(x,y) = T_{r,c}(I(x,y))$$

where $T_{r,c}$ is the intensity mapping derived from the clipped cumulative histogram for tile $(r,c)$.

To suppress high-frequency camera sensor noise without blurring fine micro-fissure boundaries, bilateral filtering is applied:

$$I^{\text{filtered}}(x) = \frac{1}{W_p} \sum_{x_i \in \Omega} I(x_i) f_r(\|I(x_i) - I(x)\|) g_s(\|x_i - x\|)$$

where $g_s$ is the spatial Gaussian kernel and $f_r$ is the radiometric intensity range kernel.

### 2.2 Morphological Gradient Segmentation
Edge boundaries and surface discontinuities are isolated using symmetrical morphological gradient operations:

$$G(I) = (I \oplus B) - (I \ominus B)$$

where $B$ is a flat structuring element (default: $3 \times 3$ rectangle).

To reject normal background surface roughness (such as brushed metal grain), a dynamic statistical noise-floor threshold is enforced:

$$T = \max\Big(T_{\text{noise\_floor}}, \mu_G + k \cdot \sigma_G\Big)$$

where $\mu_G$ and $\sigma_G$ represent the mean and standard deviation of the gradient field, and $k$ is parameterized by the detection sensitivity scalar.

### 2.3 Rotation-Invariant Geometric Classification
For each detected defect contour $C_i$, the following invariant physical properties are evaluated:

* **Rotation-Invariant Aspect Ratio ($AR$)**: Evaluated via the minimum-area enclosing bounding box:
  $$AR = \frac{\max(w_{\text{rot}}, h_{\text{rot}})}{\min(w_{\text{rot}}, h_{\text{rot}})}$$
* **Convex Hull Solidity ($S$)**: Ratio of contour area to the area of its convex hull:
  $$S = \frac{\text{Area}(C_i)}{\text{Area}(\text{ConvexHull}(C_i))}$$

Defect classification taxonomy:
* **CRACK**: $AR \ge 4.0$ and $S < 0.70$ (highly tortuous, branching, elongated fracture).
* **SCRATCH**: $AR \ge 2.5$ (linear abrasion, scoring mark).
* **SURFACE_VOID**: $AR \le 2.0$ and $S \ge 0.70$ (circular casting porosity, pitting cavity).
* **BLEMISH**: Non-structural surface contamination or discoloration.

### 2.4 Golden-Template Differential Inspection
For assemblies with complex nominal geometries, the system computes an ORB feature set on incoming components and registers them to a nominal master reference using a RANSAC-constrained planar homography matrix $H$:

$$p_{\text{golden}} \sim H \cdot p_{\text{sample}}$$

Structural pixel subtraction generates the differential defect mask:

$$\Delta(x, y) = |I_{\text{sample\_aligned}}(x, y) - I_{\text{golden}}(x, y)|$$

---

## 3. System Architecture

```
DefectVision-QA/
|
+-- defect_vision/
|   +-- __init__.py           # Package exports
|   +-- preprocessing.py      # CLAHE equalization and bilateral filtering
|   +-- defect_engine.py      # Morphological gradient segmentation and classification
|   +-- golden_comparator.py  # ORB homography alignment and differential analysis
|   +-- visualizer.py         # Production HUD and 2x2 diagnostic quad generator
|   +-- reporter.py           # Structured JSON and CSV audit logging
|
+-- samples/
|   +-- generate_synthetic_samples.py  # Programmatic test target generator
|   +-- golden_reference.png           # Pristine master reference
|   +-- sample_pass_pristine.png       # Nominal component (PASS)
|   +-- sample_fail_crack.png          # Hairline fracture (FAIL)
|   +-- sample_fail_scratches.png      # Mechanical abrasions (FAIL)
|   +-- sample_fail_voids.png          # Casting voids and porosity (FAIL)
|   +-- sample_fail_multidefect.png    # Composite multi-flaw sample (FAIL)
|
+-- tests/
|   +-- test_defect_engine.py # Deterministic unittest suite
|
+-- inspect_image.py          # Command-line single and batch inspection runner
+-- live_inspection.py        # Real-time 30+ FPS video stream processor
+-- app.py                    # Interactive Streamlit Web QA Dashboard
+-- requirements.txt          # Python dependencies
+-- LICENSE                   # MIT License
+-- README.md                 # System documentation
```

---

## 4. Installation & Setup

### Prerequisites
* Python 3.10 or higher
* Webcam or industrial USB/GigE camera (optional, for live mode)

### Step 1: Clone Repository
```bash
git clone https://github.com/Shaayan-shah/DefectVision-QA.git
cd DefectVision-QA
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Generate Benchmark Samples
```bash
python samples/generate_synthetic_samples.py
```

---

## 5. Usage & Operations

### 5.1 Command-Line Batch Inspection
To inspect all benchmark samples and export 2x2 diagnostic quad panels:
```bash
python inspect_image.py --dir samples --diagnostic-quad
```

To run a comparative inspection against a golden master reference:
```bash
python inspect_image.py --image samples/sample_fail_crack.png --golden samples/golden_reference.png
```

Sample CLI output:
```
--- Inspection: sample_fail_crack ---
Status:          [FAIL]
Defect Count:    1
Flaw Area:       1311.0 px (0.4268%)
Processing Time: 19.23 ms
Rejection Details:
  * Critical defect identified: CRACK (ID: 1, Severity: 100.0)
  * Defect count 1 exceeds allowable tolerance (0)
  * Total defect area 0.427% exceeds allowable limit (0.050%)
Defect Manifest:
  - Flaw #1: CRACK        Severity: 100.0% | Area: 1311.0px | AR: 7.15
Audit Record:    reports/json/sample_fail_crack_20261008_133159.json
```

### 5.2 Real-Time Live Camera Inspection
Launch the continuous video inspection runner:
```bash
python live_inspection.py --source 0 --width 640 --height 480
```
Interactive keyboard controls:
* `p`: Freeze current frame as the active Golden Reference Template.
* `m`: Reset to Reference-Free Morphological Gradient mode.
* `d`: Toggle 2x2 diagnostic quad display.
* `s`: Export snapshot and write entry to `reports/production_audit_log.csv`.
* `q`: Cleanly terminate inspection feed.

### 5.3 Interactive Streamlit QA Web Dashboard
Launch the browser interface for interactive parameter tuning and audit exports:
```bash
streamlit run app.py
```

---

## 6. Automated Verification Test Suite

The test suite validates:
1. Grayscale normalization and dynamic range bounds.
2. Zero false-alarm pass determination on pristine substrates.
3. Micro-crack isolation and critical classification gating.
4. Linear scratch aspect ratio invariants.
5. Casting void segmentation.
6. Golden template differential alignment.
7. Diagnostic HUD rendering and 2x2 quad assembly.
8. JSON schema and CSV audit file serialization.

Execute tests via:
```bash
python -m unittest tests/test_defect_engine.py
```

Expected output:
```
........
----------------------------------------------------------------------
Ran 8 tests in 0.664s

OK
```

---

## 7. Audit Logging Schema

For each component cycle, telemetry is committed to `reports/json/<serial_id>_<timestamp>.json`:
```json
{
  "serial_id": "PART_001",
  "timestamp_utc": "2026-10-08T13:31:59Z",
  "summary": {
    "status": "FAIL",
    "defect_count": 1,
    "total_defect_area_px": 1311.0,
    "defect_area_ratio_pct": 0.4268,
    "processing_time_ms": 19.23,
    "rejection_reasons": [
      "Critical defect identified: CRACK (ID: 1, Severity: 100.0)",
      "Defect count 1 exceeds allowable tolerance (0)"
    ],
    "defects": [
      {
        "defect_id": 1,
        "defect_type": "CRACK",
        "bbox": {"x": 180, "y": 210, "w": 210, "h": 45},
        "area_px": 1311.0,
        "perimeter_px": 482.3,
        "aspect_ratio": 7.15,
        "solidity": 0.52,
        "severity_score": 100.0,
        "centroid": {"x": 285, "y": 232}
      }
    ]
  }
}
```

---

## 8. License & Author

* **Author**: Shayan Shah
* **Email**: shayanshah.edu.pk@gmail.com
* **GitHub**: [@Shaayan-shah](https://github.com/Shaayan-shah)
* **LinkedIn**: [linkedin.com/in/shaayanshah](https://www.linkedin.com/in/shaayanshah)
* **License**: MIT License. See [LICENSE](LICENSE) for details.
