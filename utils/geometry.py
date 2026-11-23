"""
Geometry and Calculation Utilities for Pano_BoneLoss_Measurement.
"""

import math
from typing import Optional, Sequence, Tuple


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calculate 2D Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_rbl(
    cej: Tuple[float, float], crest: Tuple[float, float], apex: Tuple[float, float]
) -> float:
    """
    Compute Radiographic Bone Loss (RBL %) from CEJ, Alveolar Crest, and Root Apex.
    Clamps negative coronal crest positions to 0.0% and caps at 100.0%.
    """
    root_length = calculate_distance(cej, apex)
    if root_length == 0:
        raise ValueError("CEJ and Apex cannot be identical (Root length is 0).")

    bone_loss_distance = calculate_distance(cej, crest)
    vector_loss = (crest[0] - cej[0], crest[1] - cej[1])
    vector_root = (apex[0] - cej[0], apex[1] - cej[1])
    dot_product = vector_loss[0] * vector_root[0] + vector_loss[1] * vector_root[1]
    if dot_product < 0:
        return 0.0

    rbl_percentage = (bone_loss_distance / root_length) * 100.0
    return min(100.0, rbl_percentage)


def calculate_axial_rbl(
    cej: Tuple[float, float],
    crest: Tuple[float, float],
    apex: Tuple[float, float],
    axis_unit_vec: Optional[Sequence[float]] = None,
) -> Tuple[float, float, float]:
    """
    Project CEJ->Crest and CEJ->Apex vectors onto the tooth's principal anatomical axis
    (`crown -> apex` unit vector from Dental_008) so tilted teeth adjacent to extraction
    spaces are not inflated by horizontal interproximal offsets.

    Returns:
        (rbl_percent, axial_loss_px, axial_root_px)
    """
    vec_root = (float(apex[0]) - float(cej[0]), float(apex[1]) - float(cej[1]))
    vec_loss = (float(crest[0]) - float(cej[0]), float(crest[1]) - float(cej[1]))

    if axis_unit_vec is not None and len(axis_unit_vec) >= 2:
        ux, uy = float(axis_unit_vec[0]), float(axis_unit_vec[1])
        norm = math.hypot(ux, uy)
        if norm > 1e-6:
            ux, uy = ux / norm, uy / norm
            # Ensure unit vector points from CEJ toward Apex
            if (vec_root[0] * ux + vec_root[1] * uy) < 0:
                ux, uy = -ux, -uy
        else:
            axis_unit_vec = None

    if axis_unit_vec is None:
        euclid_root = math.hypot(vec_root[0], vec_root[1])
        if euclid_root <= 1e-6:
            raise ValueError("CEJ and Apex cannot be identical (Root length is 0).")
        ux, uy = vec_root[0] / euclid_root, vec_root[1] / euclid_root

    axial_root_px = vec_root[0] * ux + vec_root[1] * uy
    if axial_root_px <= 1e-6:
        raise ValueError("Projected axial root length must be positive.")

    axial_loss_px = max(0.0, vec_loss[0] * ux + vec_loss[1] * uy)
    rbl_pct = min(100.0, max(0.0, (axial_loss_px / axial_root_px) * 100.0))
    return (round(rbl_pct, 2), round(axial_loss_px, 2), round(axial_root_px, 2))


def pixel_to_mm(pixel_distance: float, pixels_per_mm: float) -> float:
    """Convert pixel distance to physical millimeters."""
    if pixels_per_mm <= 0:
        raise ValueError("pixels_per_mm must be positive.")
    return pixel_distance / pixels_per_mm
