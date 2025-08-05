import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Add path for importing constants and logger
sys.path.append(str(Path(__file__).parent.parent))
from config import config
from constants import (
    BROLL_PROMPTS_DIR_NAME,
    DEFAULT_VIDEO_FILENAME,
    DEFAULT_VIDEOS_OUTPUT_DIR,
    INPUT_VIDEO_DIR_NAME,
    VIDEO_FPS,
    VIDEO_GENERATION_DIR_NAME,
    VIDEO_OUTPUT_DIR_NAME,
    VIDEO_RESOLUTION,
    WORKFLOW_PROMPTS_FILENAME,
    base_data_dir,
)
from logger_config import logger

# Import real modules only if API is enabled
if config.is_api_enabled:
    import cloudinary
    import cloudinary.api
    import cloudinary.uploader
    import requests
else:
    # Use mock implementations for testing
    from unittest.mock import MagicMock

    # Mock cloudinary
    cloudinary = MagicMock()
    cloudinary.config = MagicMock()
    cloudinary.uploader = MagicMock()
    cloudinary.CloudinaryVideo = MagicMock()

    # Mock requests
    requests = MagicMock()


class MockVideoEditingResponse:
    """Mock response for video editing"""

    def __init__(self, status_code=200):
        self.status_code = status_code
        self.headers = {"content-type": "video/mp4"}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=8192):
        # Return mock video data
        mock_data = b"mock_video_data_for_testing" * 100  # Make it bigger
        for i in range(0, len(mock_data), chunk_size):
            yield mock_data[i : i + chunk_size]


def process_video_with_broll(
    heygen_video_path: str,
    segment_ids: Optional[List[int]] = None,
    output_folder: Optional[str] = None,
    cloudinary_folder: str = "video_editing_test",
    cloud_name: Optional[str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
) -> str:
    """
    Process video by adding B-roll overlays using Cloudinary.
    B-roll data and timing are automatically loaded from workflow_generated_prompts.json.

    Args:
        heygen_video_path: Path to main video file
        segment_ids: List of B-roll segment IDs to process. If None, processes all segments
        output_folder: Folder to save the final video (optional, uses constants if not provided)
        cloudinary_folder: Cloudinary folder for organization
        cloud_name: Cloudinary cloud name (optional, will use env if not provided)
        api_key: Cloudinary API key (optional, will use env if not provided)
        api_secret: Cloudinary API secret (optional, will use env if not provided)

    Returns:
        str: Path to the saved final video file

    Raises:
        ValueError: If required parameters are missing
        FileNotFoundError: If input files don't exist
        Exception: If video processing fails
    """
    # Step 1: Load environment variables and configure Cloudinary
    load_dotenv()

    if config.is_mock_enabled:
        logger.info("🔧 [MOCK] Running video editing in mock mode")
        cloud_name = "mock_cloud_name"
        api_key = "mock_api_key"
        api_secret = "mock_api_secret"
    else:
        cloud_name = (
            cloud_name
            or os.getenv("CLOUDINARY_CLOUD_NAME")
            or os.getenv("CLOUD_NAME")
        )
        api_key = (
            api_key or os.getenv("CLOUDINARY_API_KEY") or os.getenv("API_KEY")
        )
        api_secret = (
            api_secret
            or os.getenv("CLOUDINARY_API_SECRET")
            or os.getenv("API_SECRET")
        )

    if not config.is_mock_enabled and not all(
        [cloud_name, api_key, api_secret]
    ):
        raise ValueError(
            "Cloudinary credentials not found. Please check your .env file or provide them as parameters."
        )

    # Configure cloudinary
    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        secure=True,
    )
    logger.info("Cloudinary configured successfully")

    # Step 2: Set paths using constants
    workflow_json_path = f"b_roll/{base_data_dir}/{VIDEO_GENERATION_DIR_NAME}/{BROLL_PROMPTS_DIR_NAME}/{WORKFLOW_PROMPTS_FILENAME}"
    broll_videos_dir = Path(DEFAULT_VIDEOS_OUTPUT_DIR)

    if output_folder is None:
        output_folder = f"b_roll/{base_data_dir}/{VIDEO_GENERATION_DIR_NAME}/{VIDEO_OUTPUT_DIR_NAME}"

    # Step 3: Validate input files
    if not os.path.exists(heygen_video_path):
        raise FileNotFoundError(
            f"Main video file not found: {heygen_video_path}"
        )
    if not os.path.exists(workflow_json_path):
        raise FileNotFoundError(
            f"Workflow JSON file not found: {workflow_json_path}"
        )

    # Step 4: Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder created/verified: {output_path}")

    # Step 5: Read B-roll segments from workflow JSON
    try:
        with open(workflow_json_path, "r", encoding="utf-8") as f:
            workflow_data = json.load(f)

        all_segments = workflow_data.get("broll_segments", [])

        # Filter segments if specific IDs are requested
        if segment_ids:
            segments_to_process = [
                s for s in all_segments if s["segment_id"] in segment_ids
            ]
            if not segments_to_process:
                raise ValueError(f"No segments found with IDs: {segment_ids}")
        else:
            segments_to_process = all_segments

        if not segments_to_process:
            raise ValueError("No B-roll segments found to process")

        logger.info(
            f"Found {len(segments_to_process)} B-roll segments to process"
        )

    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error processing workflow JSON file: {e}")
        raise

    # Step 6: Upload main video to Cloudinary (or mock)
    timestamp = int(time.time())
    try:
        logger.info("Uploading main video to Cloudinary...")

        if config.is_mock_enabled:
            # Mock upload
            logger.info("🔧 [MOCK] Uploading main video...")
            main_video_public_id = f"mock_main_video_{timestamp}"
            main_video_upload = {"public_id": main_video_public_id}
        else:
            main_video_upload = cloudinary.uploader.upload(
                heygen_video_path,
                resource_type="video",
                folder=cloudinary_folder,
                public_id=f"heygen_main_video_{timestamp}",
            )
            main_video_public_id = main_video_upload.get("public_id")

        logger.info(
            f"Main video uploaded successfully. Public ID: {main_video_public_id}"
        )

    except Exception as e:
        logger.error(f"Error uploading main video to Cloudinary: {e}")
        raise

    # Step 7: Upload B-roll videos and build transformations
    uploaded_broll_ids = []
    transformations = []

    try:
        for segment in segments_to_process:
            segment_id = segment["segment_id"]
            start_time = segment["start_time"]
            end_time = segment["end_time"]

            # Build B-roll video path using naming convention
            # Use .1f format to match workflow.py naming (rounded to 1 decimal place)
            broll_filename = f"segment_{segment_id:02d}_{start_time:.1f}s_{end_time:.1f}s_video.mp4"
            broll_video_path = broll_videos_dir / broll_filename

            if not broll_video_path.exists():
                logger.warning(
                    f"B-roll video not found: {broll_video_path}, skipping segment {segment_id}"
                )
                continue

            # Generate unique public_id for B-roll
            broll_public_id = f"broll_segment_{segment_id}_{timestamp}"

            logger.info(f"Uploading B-roll video for segment {segment_id}...")

            if config.is_mock_enabled:
                # Mock upload
                logger.info(
                    f"🔧 [MOCK] Uploading B-roll video for segment {segment_id}..."
                )
                broll_video_public_id = f"mock_broll_{segment_id}_{timestamp}"
                broll_video_upload = {"public_id": broll_video_public_id}
            else:
                broll_video_upload = cloudinary.uploader.upload(
                    str(broll_video_path),
                    resource_type="video",
                    folder=cloudinary_folder,
                    public_id=broll_public_id,
                )
                broll_video_public_id = broll_video_upload.get("public_id")

            uploaded_broll_ids.append(broll_video_public_id)

            logger.info(
                f"B-roll video uploaded successfully. Public ID: {broll_video_public_id}"
            )

            # Add transformation for this B-roll segment
            broll_width = VIDEO_RESOLUTION.split("x")[0]
            transformation = {
                "overlay": {
                    "resource_type": "video",
                    "public_id": broll_video_public_id,
                },
                "width": broll_width,
                "crop": "fit",
                "gravity": "north",  # Position the B-roll at the top
                "start_offset": start_time,
                "end_offset": end_time,
            }
            transformations.append(transformation)

            logger.info(
                f"Added transformation for segment {segment_id}: {start_time}s - {end_time}s"
            )

    except Exception as e:
        logger.error(f"Error uploading B-roll videos to Cloudinary: {e}")
        raise

    if not transformations:
        raise ValueError("No valid B-roll videos found for processing")

    # Step 8: Build final video URL with all transformations (or mock)
    if config.is_mock_enabled:
        logger.info("🔧 [MOCK] Building video transformation...")
        final_video_url = f"https://mock-cloudinary.com/video/{main_video_public_id}_with_broll.mp4"
    else:
        final_video_url = cloudinary.CloudinaryVideo(
            main_video_public_id
        ).build_url(transformation=transformations)

    logger.info("Video transformation created successfully")
    logger.info(f"Final video URL: {final_video_url}")

    # Step 9: Download and save the final video (or create mock)
    try:
        logger.info("Downloading final video...")

        if config.is_mock_enabled:
            # Mock download
            logger.info("🔧 [MOCK] Downloading final video...")
            response = MockVideoEditingResponse()
        else:
            response = requests.get(final_video_url, stream=True)
            response.raise_for_status()

        # Generate output filename
        segments_str = "_".join(
            str(s["segment_id"]) for s in segments_to_process
        )
        output_filename = (
            f"final_video_with_broll_segments_{segments_str}_{timestamp}.mp4"
        )
        output_file_path = output_path / output_filename

        # Save the video
        with open(output_file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(
            f"✅ Video processing complete! Saved to: {output_file_path}"
        )

        return str(output_file_path)

    except Exception as e:
        logger.error(f"Error downloading final video: {e}")
        raise

    finally:
        # Cleanup: delete uploaded videos from Cloudinary (skip in mock mode)
        if not config.is_mock_enabled:
            try:
                cloudinary.uploader.destroy(
                    main_video_public_id, resource_type="video"
                )
                for broll_id in uploaded_broll_ids:
                    cloudinary.uploader.destroy(
                        broll_id, resource_type="video"
                    )
                logger.info("Temporary videos cleaned up from Cloudinary")
            except Exception as e:
                logger.warning(
                    f"Warning: Could not clean up temporary videos: {e}"
                )
        else:
            logger.info("🔧 [MOCK] Skipping cleanup in mock mode")


if __name__ == "__main__":
    # Example usage with new interface
    HEYGEN_VIDEO_PATH = f"b_roll/data/{VIDEO_GENERATION_DIR_NAME}/{INPUT_VIDEO_DIR_NAME}/{DEFAULT_VIDEO_FILENAME}"

    try:
        # Process all segments
        result_path = process_video_with_broll(
            heygen_video_path=HEYGEN_VIDEO_PATH
        )
        logger.info(f"Final video with all segments saved at: {result_path}")

        # Or process specific segments
        # result_path = process_video_with_broll(
        #     heygen_video_path=HEYGEN_VIDEO_PATH,
        #     segment_ids=[1, 3]  # Process only segments 1 and 3
        # )

    except Exception as e:
        logger.error(f"Failed to process video: {e}")
