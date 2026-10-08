"""
Synthetic Industrial Sample Generator.
Creates deterministic, reproducible test imagery representing pristine and defective surfaces.
"""

import os
import cv2
import numpy as np


def create_base_metallic_surface(width: int = 640, height: int = 480, base_val: int = 175) -> np.ndarray:
    """Generates a realistic brushed metal / machined surface texture with subtle grain."""
    np.random.seed(42)
    # Base luminance
    surface = np.full((height, width), base_val, dtype=np.uint8)
    
    # Horizontal machining brush lines
    brush_noise = np.random.normal(0, 4.0, (height, 1)).astype(np.float32)
    brush_pattern = np.tile(brush_noise, (1, width))
    
    # High-frequency grain noise
    grain = np.random.normal(0, 3.5, (height, width)).astype(np.float32)
    
    combined = surface.astype(np.float32) + brush_pattern + grain
    combined = np.clip(combined, 0, 255).astype(np.uint8)
    
    # Slight vignette gradient representing industrial diffuse dome light
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xv, yv = np.meshgrid(x, y)
    vignette = 1.0 - 0.08 * (xv**2 + yv**2)
    vignetted = np.clip(combined.astype(np.float32) * vignette, 0, 255).astype(np.uint8)
    
    return cv2.cvtColor(vignetted, cv2.COLOR_GRAY2BGR)


def add_hairline_crack(image: np.ndarray, start_pt: tuple, segments: int = 6) -> np.ndarray:
    """Simulates a jagged stress-corrosion fracture or hairline crack."""
    img = image.copy()
    np.random.seed(101)
    
    curr = list(start_pt)
    points = [tuple(curr)]
    for _ in range(segments):
        curr[0] += np.random.randint(15, 30)
        curr[1] += np.random.randint(-15, 20)
        points.append(tuple(curr))
        
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], (40, 40, 40), 2, cv2.LINE_AA)
        # Micro-fissures around main crack
        if i % 2 == 0:
            branch_pt = (points[i][0] + np.random.randint(-5, 10), points[i][1] + np.random.randint(-8, 12))
            cv2.line(img, points[i], branch_pt, (50, 50, 50), 1, cv2.LINE_AA)
            
    return img


def add_abrasion_scratch(image: np.ndarray, pt1: tuple, pt2: tuple) -> np.ndarray:
    """Simulates an elongated mechanical handling scratch."""
    img = image.copy()
    # Deep shadow line flanked by slight bright burr reflection
    cv2.line(img, pt1, pt2, (25, 25, 25), 2, cv2.LINE_AA)
    # Bright parallel burr line
    burr_pt1 = (pt1[0], pt1[1] + 1)
    burr_pt2 = (pt2[0], pt2[1] + 1)
    cv2.line(img, burr_pt1, burr_pt2, (230, 230, 230), 1, cv2.LINE_AA)
    return img


def add_pitting_voids(image: np.ndarray, centers: list) -> np.ndarray:
    """Simulates casting cavities, porosity, or pitting voids."""
    img = image.copy()
    for cx, cy, r in centers:
        cv2.circle(img, (cx, cy), r, (30, 30, 30), -1)
        # Irregular perimeter
        cv2.ellipse(img, (cx, cy), (r + 2, r), 30, 0, 360, (50, 50, 50), 1)
    return img


def generate_all_samples(output_dir: str = "samples"):
    """Generates complete sample collection."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Golden Reference
    golden = create_base_metallic_surface()
    golden_path = os.path.join(output_dir, "golden_reference.png")
    cv2.imwrite(golden_path, golden)
    print(f"Generated: {golden_path}")

    # 2. Pristine Component (Pass)
    pristine = create_base_metallic_surface()
    pristine_path = os.path.join(output_dir, "sample_pass_pristine.png")
    cv2.imwrite(pristine_path, pristine)
    print(f"Generated: {pristine_path}")

    # 3. Flawed: Hairline Crack (Fail)
    crack_img = add_hairline_crack(create_base_metallic_surface(), (180, 210), segments=8)
    crack_path = os.path.join(output_dir, "sample_fail_crack.png")
    cv2.imwrite(crack_path, crack_img)
    print(f"Generated: {crack_path}")

    # 4. Flawed: Linear Scratches (Fail)
    scratch_img = create_base_metallic_surface()
    scratch_img = add_abrasion_scratch(scratch_img, (120, 140), (420, 260))
    scratch_img = add_abrasion_scratch(scratch_img, (200, 320), (380, 370))
    scratch_path = os.path.join(output_dir, "sample_fail_scratches.png")
    cv2.imwrite(scratch_path, scratch_img)
    print(f"Generated: {scratch_path}")

    # 5. Flawed: Surface Voids (Fail)
    void_img = create_base_metallic_surface()
    void_img = add_pitting_voids(void_img, [(240, 180, 7), (255, 195, 5), (420, 310, 8)])
    void_path = os.path.join(output_dir, "sample_fail_voids.png")
    cv2.imwrite(void_path, void_img)
    print(f"Generated: {void_path}")

    # 6. Flawed: Multi-Defect Combined (Fail)
    multi_img = create_base_metallic_surface()
    multi_img = add_hairline_crack(multi_img, (150, 160), segments=6)
    multi_img = add_abrasion_scratch(multi_img, (310, 260), (480, 340))
    multi_img = add_pitting_voids(multi_img, [(200, 340, 6)])
    multi_path = os.path.join(output_dir, "sample_fail_multidefect.png")
    cv2.imwrite(multi_path, multi_img)
    print(f"Generated: {multi_path}")

    print("All synthetic industrial samples successfully generated.")


if __name__ == "__main__":
    generate_all_samples()
