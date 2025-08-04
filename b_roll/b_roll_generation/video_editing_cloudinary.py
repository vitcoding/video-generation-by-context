import json
import os
import time
from pathlib import Path
from typing import Optional

import cloudinary
import cloudinary.api
import cloudinary.uploader
import requests
from dotenv import load_dotenv
from logger_config import logger


def process_video_with_broll(
    heygen_video_path: str,
    broll_video_path: str,
    json_path: str,
    broll_segment_id: int,
    output_folder: str = "b_roll/data/video_generation/video_output",
    cloudinary_folder: str = "video_editing_test",
    cloud_name: Optional[str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
) -> str:
    """
    Process video by adding B-roll overlay using Cloudinary.

    Args:
        heygen_video_path: Path to main video file
        broll_video_path: Path to B-roll video file
        json_path: Path to JSON file with timing information
        broll_segment_id: ID of the B-roll segment to use
        output_folder: Folder to save the final video
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

    # Step 2: Validate input files
    if not os.path.exists(heygen_video_path):
        raise FileNotFoundError(
            f"Main video file not found: {heygen_video_path}"
        )
    if not os.path.exists(broll_video_path):
        raise FileNotFoundError(
            f"B-roll video file not found: {broll_video_path}"
        )
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # Step 3: Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder created/verified: {output_path}")

    # Step 4: Generate unique public_id for B-roll
    broll_filename = os.path.splitext(os.path.basename(broll_video_path))[0]
    timestamp = int(time.time())
    broll_public_id = f"broll_{broll_filename}_{timestamp}"

    # Step 5: Read timestamps from JSON file
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        segment = next(
            (
                s
                for s in data.get("broll_segments", [])
                if s["segment_id"] == broll_segment_id
            ),
            None,
        )

        if not segment:
            raise ValueError(
                f"Segment with id {broll_segment_id} not found in JSON file."
            )

        start_time = segment.get("start_time")
        end_time = segment.get("end_time")

        if start_time is None or end_time is None:
            raise ValueError(
                "Start or end time not found in the selected segment."
            )

        logger.info(
            f"Found timestamps for segment {broll_segment_id}: Start = {start_time}s, End = {end_time}s"
        )

    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error processing JSON file: {e}")
        raise

    # Step 6: Upload videos to Cloudinary
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

        logger.info(
            f"Uploading B-roll video to Cloudinary with unique ID: {broll_public_id}"
        )
        broll_video_upload = cloudinary.uploader.upload(
            broll_video_path,
            resource_type="video",
            folder=cloudinary_folder,
            public_id=broll_public_id,
        )
        broll_video_public_id = broll_video_upload.get("public_id")
        logger.info(
            f"B-roll video uploaded successfully. Public ID: {broll_video_public_id}"
        )

    except Exception as e:
        logger.error(f"Error uploading videos to Cloudinary: {e}")
        raise

    # Step 7: Build transformation with B-roll overlay
    transformation = [
        {
            "overlay": {
                "resource_type": "video",
                "public_id": broll_video_public_id,
            },
            "width": "1.0",
            "crop": "fit",
            "gravity": "north",  # Position the B-roll at the top
            "start_offset": start_time,
            "end_offset": end_time,
        }
    ]

    final_video_url = cloudinary.CloudinaryVideo(
        main_video_public_id
    ).build_url(transformation=transformation)

    logger.info("Video transformation created successfully")
    logger.info(f"Final video URL: {final_video_url}")

    # Step 8: Download and save the final video
    try:
        logger.info("Downloading final video...")
        response = requests.get(final_video_url, stream=True)
        response.raise_for_status()

        # Generate output filename
        output_filename = f"final_video_with_broll_{timestamp}.mp4"
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
            cloudinary.uploader.destroy(
                broll_video_public_id, resource_type="video"
            )
            logger.info("Temporary videos cleaned up from Cloudinary")
        except Exception as e:
            logger.warning(
                f"Warning: Could not clean up temporary videos: {e}"
            )


if __name__ == "__main__":
    # Example usage with hardcoded paths (for testing)
    HEYGEN_VIDEO_PATH = "/Users/dahaniglikovdarkhan/Documents/repos/teleCreaaiQdrant/tests/test_video_generation/output/heygen_example.mp4"
    BROLL_VIDEO_PATH = "/Users/dahaniglikovdarkhan/Documents/repos/teleCreaaiQdrant/tests/test_video_generation/output/vertical_broll_segment_11.mp4"
    JSON_PATH = "/Users/dahaniglikovdarkhan/Documents/repos/teleCreaaiQdrant/tests/test_video_generation/broll_prompts_api_generated.json"
    BROLL_SEGMENT_ID_TO_USE = 1

    try:
        result_path = process_video_with_broll(
            heygen_video_path=HEYGEN_VIDEO_PATH,
            broll_video_path=BROLL_VIDEO_PATH,
            json_path=JSON_PATH,
            broll_segment_id=BROLL_SEGMENT_ID_TO_USE,
        )
        logger.info(f"Final video saved at: {result_path}")
    except Exception as e:
        logger.error(f"Failed to process video: {e}")
