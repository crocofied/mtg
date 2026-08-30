"""Computer-vision layer: capture, calibration, detection, recognition."""

from .calibration import (
    CalibrationError,
    MatCalibration,
    calibrate_from_corners,
    calibrate_from_markers,
    detect_markers,
    find_mat_quad,
    generate_marker_sheet,
)
from .capture import CameraSource, FrameSource, ImageDirSource, ListSource, VideoSource, open_source
from .detect import CARD_H, CARD_W, CardDetection, CardDetector, DetectorConfig, warp_card
from .mat import MatLayout, MatRegion, default_layout
from .pipeline import Observation, ObservedCard, PipelineConfig, VisionPipeline
from .recognize import CardIndex, IndexEntry, RecognitionResult, describe, phash

__all__ = [
    "CARD_H",
    "CARD_W",
    "CalibrationError",
    "CameraSource",
    "CardDetection",
    "CardDetector",
    "CardIndex",
    "DetectorConfig",
    "FrameSource",
    "ImageDirSource",
    "IndexEntry",
    "ListSource",
    "MatCalibration",
    "MatLayout",
    "MatRegion",
    "Observation",
    "ObservedCard",
    "PipelineConfig",
    "RecognitionResult",
    "VideoSource",
    "VisionPipeline",
    "calibrate_from_corners",
    "calibrate_from_markers",
    "default_layout",
    "describe",
    "detect_markers",
    "find_mat_quad",
    "generate_marker_sheet",
    "open_source",
    "phash",
    "warp_card",
]
