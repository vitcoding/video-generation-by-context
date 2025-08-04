"""
B-roll Generation Module

This module contains components for generating b-roll content from audio transcripts:
- BRollAnalyzer: Analyzes transcripts and generates prompts
- ImageGenerator: Generates images from prompts
- KlingImageToVideoGenerator: Converts images to videos
- prompts: Centralized prompts for all generation processes
"""

from . import prompts
from .broll_image_generation import ImageGenerator
from .broll_prompts import BRollAnalyzer
from .kling_image_to_video import KlingImageToVideoGenerator
from .video_editing_cloudinary import process_video_with_broll

__all__ = [
    "BRollAnalyzer",
    "ImageGenerator",
    "KlingImageToVideoGenerator",
    "prompts",
    "process_video_with_broll",
]
