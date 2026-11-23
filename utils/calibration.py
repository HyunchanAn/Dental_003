from typing import Any, Optional, Sequence, Union


class CalibrationManager:
    """
    Maps panoramic pixel distances to physical millimeters (mm) while tracking whether
    calibration came from verified DICOM metadata/reference object (`is_calibrated=True`)
    or nominal fallback (`is_calibrated=False`).
    """

    def __init__(self, pixels_per_mm: float = 1.0, is_calibrated: bool = False):
        if pixels_per_mm <= 0:
            raise ValueError("pixels_per_mm must be positive.")
        self.pixels_per_mm = float(pixels_per_mm)
        self.is_calibrated = bool(is_calibrated)

    def set_scale_from_dicom(self, pixel_spacing: Optional[Sequence[float]]) -> bool:
        """
        Configure scale from DICOM PixelSpacing / ImagerPixelSpacing [row_mm, col_mm].
        """
        if pixel_spacing and len(pixel_spacing) >= 2:
            avg_spacing = (float(pixel_spacing[0]) + float(pixel_spacing[1])) / 2.0
            if avg_spacing > 0:
                self.pixels_per_mm = 1.0 / avg_spacing
                self.is_calibrated = True
                return True
        return False

    def set_scale_from_panorama_meta(
        self,
        image_meta: Union[Any, dict, None],
        nominal_mm_per_px: float = 0.1,
    ) -> bool:
        """
        Configure scale from Dental_Core `PanoramaImageMeta` (or dict).
        Falls back to `nominal_mm_per_px` with `is_calibrated=False` when uncalibrated.
        """
        spacing = None
        calibrated_flag = False
        if image_meta is not None:
            if isinstance(image_meta, dict):
                spacing = image_meta.get("pixel_spacing_mm")
                calibrated_flag = bool(image_meta.get("is_calibrated", spacing is not None))
            else:
                spacing = getattr(image_meta, "pixel_spacing_mm", None)
                calibrated_flag = bool(getattr(image_meta, "is_calibrated", spacing is not None))

        if calibrated_flag and spacing and self.set_scale_from_dicom(spacing):
            return True

        self.pixels_per_mm = 1.0 / max(1e-6, float(nominal_mm_per_px))
        self.is_calibrated = False
        return False

    def set_scale_from_reference(
        self, reference_pixel_length: float, reference_real_length_mm: float
    ) -> bool:
        """
        Configure scale from a known physical reference structure (e.g., fixture or calibration ball).
        """
        if reference_pixel_length > 0 and reference_real_length_mm > 0:
            self.pixels_per_mm = float(reference_pixel_length) / float(reference_real_length_mm)
            self.is_calibrated = True
            return True
        return False

    def pixel_to_mm(self, pixel_distance: float) -> float:
        return float(pixel_distance) / self.pixels_per_mm
