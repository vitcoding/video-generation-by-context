"""
B-Roll Video Generation Package

This package provides tools for generating b-roll videos from audio transcripts.
It includes modules for transcript analysis, image generation, and video creation.
"""

__version__ = "1.0.0"
__author__ = "Video Generation Team"

from b_roll.logger_config import logger

from .b_roll_generation import (
    BRollAnalyzer,
    ImageGenerator,
    KlingImageToVideoGenerator,
)
from .config import config
from .constants import *

# Import main components for easy access
from .workflow import UnifiedWorkflow

__all__ = [
    "UnifiedWorkflow",
    "BRollAnalyzer",
    "ImageGenerator",
    "KlingImageToVideoGenerator",
    "config",
    "logger",
]
