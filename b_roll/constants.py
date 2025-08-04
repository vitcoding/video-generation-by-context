#!/usr/bin/env python3
"""
Constants for video generation project
"""

# Video generation constants
# VIDEO_RESOLUTION = "1080x1920"
VIDEO_RESOLUTION = "720x1280"
# VIDEO_RESOLUTION = "1920x1080"  # Full HD resolution for video generation
# VIDEO_RESOLUTION = "1280x720"  # HD resolution
# VIDEO_RESOLUTION = "854x480"    # SD resolution
# VIDEO_RESOLUTION = "3840x2160"  # 4K resolution

# Model endpoints
FAL_MODEL_ENDPOINT = "fal-ai/imagen4/preview/fast"
KLING_MODEL_ENDPOINT = "fal-ai/kling-video/v1.6/pro/image-to-video"

# Image aspect ratios
# IMAGE_ASPECT_RATIO = "16:9"  # Default aspect ratio for video content
# IMAGE_ASPECT_RATIO = "1:1"   # Square format, good for social media
IMAGE_ASPECT_RATIO = "9:16"  # Vertical format, good for mobile/vertical video
# IMAGE_ASPECT_RATIO = "4:3"   # Traditional format
# IMAGE_ASPECT_RATIO = "3:2"   # Photography standard
# IMAGE_ASPECT_RATIO = "21:9"  # Ultra-wide format

# Video generation parameters
VIDEO_FPS = 24  # Frames per second for video generation
# VIDEO_FPS = 30  # Frames per second for video generation
VIDEO_DURATION = 5.0  # Default video duration in seconds
DEFAULT_CFG_SCALE = 0.7  # Default CFG scale for video generation

# File paths - dynamically set based on mock mode
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from config import config

# Use data_mock directory when mock mode is enabled
base_data_dir = "data_mock" if config.is_mock_enabled else "data"

DEFAULT_PROMPTS_FILE = f"./b_roll/{base_data_dir}/video_generation/broll_prompts_api_generated.json"
DEFAULT_IMAGES_OUTPUT_DIR = (
    f"./b_roll/{base_data_dir}/video_generation/images_input"
)
DEFAULT_VIDEOS_OUTPUT_DIR = (
    f"./b_roll/{base_data_dir}/video_generation/videos_output"
)

# API configuration
DEFAULT_SEED = 42  # Default random seed for reproducible results
DEFAULT_NUM_INFERENCE_STEPS = (
    15  # Default number of inference steps for image generation
    # 20  # Default number of inference steps for image generation
)

# Workflow configuration
ENABLE_VIDEO_GENERATION = True  # Flag to enable/disable video generation
# ENABLE_VIDEO_GENERATION = False  # Set to False to skip video generation

# Content policy replacements (for API compliance)
CONTENT_REPLACEMENTS = {
    "AI-dominated world": "world of advanced technology",
    "AI projection": "digital projection",
    "holographic AI projection": "holographic digital display",
    "AI hologram": "digital hologram",
    "humanoid shape": "abstract geometric shape",
    "AI avatar generation": "digital character design",
    "avatar generation": "character design",
    "avatar being digitally rendered": "character being designed",
    "avatar": "digital character",
    "digital avatar": "digital character",
    "finished avatar blinking": "finished character animation",
    "avatar's digital eye": "character's digital features",
    "AI generation": "digital creation",
    "AI-generated": "digitally created",
    "deepfake": "digital effect",
    "synthetic": "digital",
    "role of AI": "role of intelligent systems",
    " AI ": " intelligent system ",
}
