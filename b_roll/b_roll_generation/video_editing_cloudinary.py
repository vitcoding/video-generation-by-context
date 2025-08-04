import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import cloudinary
import cloudinary.api
import cloudinary.uploader
import requests
from dotenv import load_dotenv

# Добавляем путь для импорта констант и логера
sys.path.append(str(Path(__file__).parent.parent))
from constants import (
    DEFAULT_VIDEOS_OUTPUT_DIR,
    VIDEO_FPS,
    VIDEO_RESOLUTION,
    base_data_dir,
)
from logger_config import logger


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

    if not all([cloud_name, api_key, api_secret]):
        raise ValueError(
            "Cloudinary credentials not found. Please check your .env file or provide them as parameters."
        )

    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        secure=True,
    )
    logger.info("Cloudinary configured successfully")

    # Step 2: Set paths using constants
    workflow_json_path = f"b_roll/{base_data_dir}/video_generation/broll_prompts/workflow_generated_prompts.json"
    broll_videos_dir = Path(DEFAULT_VIDEOS_OUTPUT_DIR)

    if output_folder is None:
        output_folder = f"b_roll/{base_data_dir}/video_generation/video_output"

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

    # Step 6: Upload main video to Cloudinary
    timestamp = int(time.time())
    try:
        logger.info("Uploading main video to Cloudinary...")
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
            broll_filename = (
                f"segment_{segment_id:02d}_{start_time}s_{end_time}s_video.mp4"
            )
            broll_video_path = broll_videos_dir / broll_filename

            if not broll_video_path.exists():
                logger.warning(
                    f"B-roll video not found: {broll_video_path}, skipping segment {segment_id}"
                )
                continue

            # Generate unique public_id for B-roll
            broll_public_id = f"broll_segment_{segment_id}_{timestamp}"

            logger.info(f"Uploading B-roll video for segment {segment_id}...")
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

    # Step 8: Build final video URL with all transformations
    final_video_url = cloudinary.CloudinaryVideo(
        main_video_public_id
    ).build_url(transformation=transformations)

    logger.info("Video transformation created successfully")
    logger.info(f"Final video URL: {final_video_url}")

    # Step 9: Download and save the final video
    try:
        logger.info("Downloading final video...")
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
        # Cleanup: delete uploaded videos from Cloudinary
        try:
            cloudinary.uploader.destroy(
                main_video_public_id, resource_type="video"
            )
            for broll_id in uploaded_broll_ids:
                cloudinary.uploader.destroy(broll_id, resource_type="video")
            logger.info("Temporary videos cleaned up from Cloudinary")
        except Exception as e:
            logger.warning(
                f"Warning: Could not clean up temporary videos: {e}"
            )


if __name__ == "__main__":
    # Example usage with new interface
    HEYGEN_VIDEO_PATH = "b_roll/data/video_generation/input_video/video.mp4"

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
