#!/usr/bin/env python3
"""
Constants for video generation project
"""

# Video generation constants
# VIDEO_RESOLUTION = "1080x1920"
# VIDEO_RESOLUTION = "720x1280"
# VIDEO_RESOLUTION = "1920x1080"  # Full HD resolution for video generation
VIDEO_RESOLUTION = "1280x720"  # HD resolution
# VIDEO_RESOLUTION = "854x480"    # SD resolution
# VIDEO_RESOLUTION = "3840x2160"  # 4K resolution

# Model endpoints
FAL_MODEL_ENDPOINT = "fal-ai/imagen4/preview/fast"
KLING_MODEL_ENDPOINT = "fal-ai/kling-video/v1.6/pro/image-to-video"

# Image aspect ratios
IMAGE_ASPECT_RATIO = "16:9"  # Default aspect ratio for video content
# IMAGE_ASPECT_RATIO = "9:16"  # Vertical format, good for mobile/vertical video
# IMAGE_ASPECT_RATIO = "1:1"   # Square format, good for social media
# IMAGE_ASPECT_RATIO = "4:3"   # Traditional format
# IMAGE_ASPECT_RATIO = "3:2"   # Photography standard
# IMAGE_ASPECT_RATIO = "21:9"  # Ultra-wide format

# Video generation parameters
VIDEO_FPS = 24  # Frames per second for video generation
# VIDEO_FPS = 30  # Frames per second for video generation
VIDEO_DURATION = 5.0  # Default video duration in seconds
DEFAULT_CFG_SCALE = 0.7  # Default CFG scale for video generation

# API configuration
DEFAULT_SEED = 42  # Default random seed for reproducible results
DEFAULT_NUM_INFERENCE_STEPS = (
    15  # Default number of inference steps for image generation
    # 20  # Default number of inference steps for image generation
)

# OpenAI models configuration
OPENAI_MODELS = {
    "gpt-4": {
        "name": "gpt-4",
        "max_context_tokens": 8192,
        "max_output_tokens": 4096,
        "cost_per_1k_input": 0.03,
        "cost_per_1k_output": 0.06,
    },
    "gpt-4o-mini": {
        "name": "gpt-4o-mini",
        "max_context_tokens": 128000,
        "max_output_tokens": 16384,
        "cost_per_1k_input": 0.00015,
        "cost_per_1k_output": 0.0006,
    },
    "gpt-4o": {
        "name": "gpt-4o",
        "max_context_tokens": 128000,
        "max_output_tokens": 16384,
        "cost_per_1k_input": 0.0025,
        "cost_per_1k_output": 0.01,
    },
}

# Default models for different tasks
DEFAULT_OPENAI_MODEL = "gpt-4"
FALLBACK_OPENAI_MODEL = (
    "gpt-4o-mini"  # Модель с большим контекстом для длинных промптов
)
PREMIUM_OPENAI_MODEL = "gpt-4o"  # Самая мощная модель

# Workflow configuration
# ENABLE_VIDEO_GENERATION = True  # Flag to enable/disable video generation
ENABLE_VIDEO_GENERATION = False  # Set to False to skip video generation

# B-roll generation configuration
# MAX_SEGMENTS = 1  # Maximum number of b-roll segments to generate
MAX_SEGMENTS = 20  # Maximum number of b-roll segments to generate

# B-roll distribution configuration
# Controls how b-roll segments are distributed across video timeline
EARLY_SEGMENT_RATIO = 0.4  # 40% of segments go in early portion
EARLY_DURATION_RATIO = 0.25  # First 25% of video duration for early segments
REMAINING_SEGMENT_RATIO = (
    1.0 - EARLY_SEGMENT_RATIO
)  # 60% of segments go in remaining portion
REMAINING_DURATION_RATIO = (
    1.0 - EARLY_DURATION_RATIO
)  # Remaining 75% of video duration

# File paths - dynamically set based on mock mode
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from config import config

# Use data_mock directory when mock mode is enabled
base_data_dir = "data_mock" if config.is_mock_enabled else "data"

# Directory structure constants
B_ROLL_MODULE_NAME = "b_roll"
VIDEO_GENERATION_DIR_NAME = "video_generation"
AUDIO_TRANSCRIPT_DIR_NAME = "audio_transcript"
BROLL_PROMPTS_DIR_NAME = "broll_prompts"
INPUT_VIDEO_DIR_NAME = "input_video"
VIDEO_OUTPUT_DIR_NAME = "video_output"
IMAGES_INPUT_DIR_NAME = "images_input"
VIDEOS_OUTPUT_DIR_NAME = "videos_output"

# Base directory paths
BASE_DATA_DIR_PATH = (
    f"./{B_ROLL_MODULE_NAME}/{base_data_dir}/{VIDEO_GENERATION_DIR_NAME}"
)
BASE_RELATIVE_PATH = f"{base_data_dir}/{VIDEO_GENERATION_DIR_NAME}"

# Main directory paths
AUDIO_TRANSCRIPT_DIR = f"{BASE_DATA_DIR_PATH}/{AUDIO_TRANSCRIPT_DIR_NAME}"
BROLL_PROMPTS_DIR = f"{BASE_DATA_DIR_PATH}/{BROLL_PROMPTS_DIR_NAME}"
INPUT_VIDEO_DIR = f"{BASE_DATA_DIR_PATH}/{INPUT_VIDEO_DIR_NAME}"
VIDEO_OUTPUT_DIR = f"{BASE_DATA_DIR_PATH}/{VIDEO_OUTPUT_DIR_NAME}"
IMAGES_INPUT_DIR = f"{BASE_DATA_DIR_PATH}/{IMAGES_INPUT_DIR_NAME}"
VIDEOS_OUTPUT_DIR_PATH = f"{BASE_DATA_DIR_PATH}/{VIDEOS_OUTPUT_DIR_NAME}"

# Existing paths for backward compatibility
DEFAULT_IMAGES_OUTPUT_DIR = IMAGES_INPUT_DIR
DEFAULT_VIDEOS_OUTPUT_DIR = VIDEOS_OUTPUT_DIR_PATH
DEFAULT_PROMPTS_FILE = (
    f"{BASE_DATA_DIR_PATH}/{BROLL_PROMPTS_DIR_NAME}_api_generated.json"
)

# Standard file names
TRANSCRIPTION_JSON_FILENAME = "transcription_verbose_to_json.json"
WORKFLOW_PROMPTS_FILENAME = "workflow_generated_prompts.json"
API_PROMPTS_FILENAME = "broll_prompts_api_generated.json"
WORKFLOW_REPORT_FILENAME = "workflow_complete_report.json"
DEFAULT_VIDEO_FILENAME = "video.mp4"
ENV_FILENAME = ".env"

# Full file paths
TRANSCRIPTION_JSON_PATH = (
    f"{AUDIO_TRANSCRIPT_DIR}/{TRANSCRIPTION_JSON_FILENAME}"
)
WORKFLOW_PROMPTS_PATH = f"{BROLL_PROMPTS_DIR}/{WORKFLOW_PROMPTS_FILENAME}"
API_PROMPTS_PATH = f"{BROLL_PROMPTS_DIR}/{API_PROMPTS_FILENAME}"
WORKFLOW_REPORT_PATH = f"{BASE_DATA_DIR_PATH}/{WORKFLOW_REPORT_FILENAME}"
DEFAULT_VIDEO_PATH = f"{INPUT_VIDEO_DIR}/{DEFAULT_VIDEO_FILENAME}"

# Environment and configuration paths
PROJECT_ROOT_RELATIVE_PATH = ".."  # Relative to b_roll_generation modules
ENV_FILE_PATH = f"{PROJECT_ROOT_RELATIVE_PATH}/{ENV_FILENAME}"
