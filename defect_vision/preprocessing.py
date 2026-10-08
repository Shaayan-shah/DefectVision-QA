"""
Surface Preprocessing and Illumination Normalization Module.
Provides edge-preserving noise reduction and adaptive contrast equalization.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class SurfacePreprocessor:
    """Preprocesses raw industrial surface imagery for defect detection."""

    def __init__(
        self,
        clip_limit: float = 2.5,
        tile_grid_size: Tuple[int, int] = (8, 8),
        denoise_diameter: int = 7,
        sigma_color: float = 50.0,
        sigma_space: float = 50.0,
        target_resolution: Optional[Tuple[int, int]] = None
    ):
        """
        Args:
            clip_limit: Threshold for contrast limiting in CLAHE.
            tile_grid_size: Grid dimensions for localized histogram equalization.
            denoise_diameter: Pixel neighborhood diameter for bilateral filtering.
            sigma_color: Filter sigma in the color space.
            sigma_space: Filter sigma in the coordinate space.
            target_resolution: Optional (width, height) to resize inputs to standard scale.
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.denoise_diameter = denoise_diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.target_resolution = target_resolution
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        """Resizes input image if a target resolution is defined."""
        if self.target_resolution is not None:
            return cv2.resize(image, self.target_resolution, interpolation=cv2.INTER_AREA)
        return image

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Converts an RGB or BGR image to single-channel 8-bit grayscale."""
        if len(image.shape) == 2:
            return image
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def normalize_illumination(self, gray: np.ndarray) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
        to eliminate non-uniform lighting gradients and shadows across the substrate.
        """
        return self.clahe.apply(gray)

    def denoise(self, gray: np.ndarray) -> np.ndarray:
        """
        Applies bilateral filtering to suppress camera sensor noise while
        strictly preserving high-frequency edge transitions of micro-defects.
        """
        return cv2.bilateralFilter(
            gray,
            d=self.denoise_diameter,
            sigmaColor=self.sigma_color,
            sigmaSpace=self.sigma_space
        )

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes full preprocessing pipeline.
        
        Returns:
            Tuple of (color_base_image, normalized_denoised_grayscale)
        """
        resized = self.resize_if_needed(image)
        if len(resized.shape) == 2:
            color_base = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            gray = resized
        else:
            color_base = resized.copy()
            gray = self.to_grayscale(resized)

        equalized = self.normalize_illumination(gray)
        denoised = self.denoise(equalized)
        return color_base, denoised
