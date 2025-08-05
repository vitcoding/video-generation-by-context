"""
B-roll Generation Module

This module contains components for generating b-roll content from audio transcripts:
- VideoTranscriber: Transcribes video files with word-level timestamps
- BRollAnalyzer: Analyzes transcripts and generates prompts
- ImageGenerator: Generates images from prompts
- KlingImageToVideoGenerator: Converts images to videos
- process_video_with_broll: Edits final video with b-roll overlays
- prompts: Centralized prompts for all generation processes
"""

from . import prompts
from .broll_image_generation import ImageGenerator
from .broll_prompts import BRollAnalyzer
from .kling_image_to_video import KlingImageToVideoGenerator
from .video_editing_cloudinary import process_video_with_broll
from .word_level_transcriber import VideoTranscriber, transcribe_video_to_json

__all__ = [
    "VideoTranscriber",
    "transcribe_video_to_json",
    "BRollAnalyzer",
    "ImageGenerator",
    "KlingImageToVideoGenerator",
    "process_video_with_broll",
    "prompts",
]
