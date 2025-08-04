#!/usr/bin/env python3
"""
Test script for Kling 1.6 Pro Image-to-Video Generator

This script tests image-to-video generation using Kling 1.6 Pro API
with the provided business team collaboration image.
"""

import json
import os

# Import configuration and mock modules
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from logger_config import logger

sys.path.append(str(Path(__file__).parent.parent))
import mock_api
from config import config
from constants import (
    DEFAULT_VIDEOS_OUTPUT_DIR,
    IMAGE_ASPECT_RATIO,
    KLING_MODEL_ENDPOINT,
    VIDEO_DURATION,
    VIDEO_FPS,
    VIDEO_RESOLUTION,
)
from mock_api import mock_fal_client, mock_requests

from .prompts import (
    DEFAULT_CFG_SCALE,
    DEFAULT_VIDEO_DURATION,
    NEGATIVE_PROMPT_VIDEO,
)

# Import real modules only if API is enabled
if config.is_api_enabled:
    import fal_client
    import requests
else:
    # Use mock modules
    fal_client = mock_fal_client
    requests = mock_requests
    # Use MockKlingImageToVideoGenerator for testing
    KlingImageToVideoGenerator = mock_api.mock_kling_image_to_video_generator


class KlingImageToVideoGenerator:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Kling 1.6 Pro Image-to-Video Generator

        Args:
            api_key: fal.ai API key. If None, will try to get from .env file
        """
        # Load environment variables from .env file in project root
        # project_root = Path(__file__).parent.parent.parent
        project_root = Path(__file__).parent.parent
        env_file = project_root / ".env"
        load_dotenv(env_file)

        # Get API key based on configuration
        self.api_key = api_key or config.get_api_key("FAL_KEY")

        if config.is_api_enabled and not self.api_key:
            logger.info(f"⚠️ Warning: FAL_KEY not found in {env_file}")
            logger.info("🔧 Using mock mode for video generation")
            # Don't raise error, continue with mock mode

        # Configure fal client only if API is enabled
        if config.is_api_enabled:
            fal_client.api_key = self.api_key
        else:
            logger.info("🔧 [MOCK] Using mock fal_client")

        # API endpoint for Kling 1.6 Pro Image-to-Video
        self.model_endpoint = KLING_MODEL_ENDPOINT

        # Output directory
        self.output_dir = Path(DEFAULT_VIDEOS_OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)

    def load_broll_prompts(
        self,
        prompts_file: str = None,
    ) -> Dict:
        # Use default prompts file if not specified
        if prompts_file is None:
            # Use data_mock when mock mode is enabled
            import sys

            sys.path.append(str(Path(__file__).parent.parent))
            from config import config

            base_data_dir = "data_mock" if config.is_mock_enabled else "data"
            prompts_file = str(
                Path(__file__).parent.parent
                / f"{base_data_dir}/video_generation/broll_prompts_api_generated.json"
            )
        """
        Load b-roll prompts from JSON file

        Args:
            prompts_file: Path to the prompts JSON file

        Returns:
            Dictionary with b-roll prompts data
        """
        try:
            if not os.path.exists(prompts_file):
                raise FileNotFoundError(
                    f"Prompts file not found: {prompts_file}"
                )

            with open(prompts_file, "r", encoding="utf-8") as f:
                prompts_data = json.load(f)

            logger.info(
                f"✅ Loaded {len(prompts_data.get('broll_segments', []))} b-roll segments from {prompts_file}"
            )
            return prompts_data

        except Exception as e:
            logger.error(f"❌ Error loading prompts: {e}")
            return {}

    def upload_image(self, image_path: str) -> str:
        """
        Upload image to fal.ai storage and return URL

        Args:
            image_path: Path to the input image

        Returns:
            URL of uploaded image
        """
        try:
            logger.info(f"📤 Uploading image: {image_path}")

            # Check if image file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Read image data
            with open(image_path, "rb") as f:
                image_data = f.read()

            # Upload using bytes data
            image_url = fal_client.upload(image_data, "image/png")

            logger.info(f"✅ Image uploaded: {image_url}")
            return image_url

        except Exception as e:
            logger.error(f"❌ Error uploading image: {e}")
            raise

    def generate_video_from_image(
        self,
        image_path: str,
        prompt: str,
        aspect_ratio: str = IMAGE_ASPECT_RATIO,
    ) -> Optional[Dict]:
        """
        Generate video from image using Kling 1.6 Pro

        Args:
            image_path: Path to input image
            prompt: Video generation prompt
            aspect_ratio: Video aspect ratio (16:9, 9:16, 1:1)

        Returns:
            Dictionary with video information or None if failed
        """
        try:
            logger.info(f"🎬 Generating video from image...")
            logger.info(
                f"📝 Prompt: {prompt[:100]}..."
            )  # Truncate long prompts
            logger.info(f"📐 Aspect Ratio: {aspect_ratio}")

            # Upload image first
            image_url = self.upload_image(image_path)

            # Submit request to fal.ai Kling 1.6 Pro Image-to-Video
            logger.info("⏳ Submitting image-to-video request...")
            result = fal_client.subscribe(
                self.model_endpoint,
                arguments={
                    "image_url": image_url,
                    "prompt": prompt,
                    "duration": DEFAULT_VIDEO_DURATION,
                    "aspect_ratio": aspect_ratio,
                    "negative_prompt": NEGATIVE_PROMPT_VIDEO,
                    "cfg_scale": DEFAULT_CFG_SCALE,
                },
            )

            if result and "video" in result:
                video_url = result["video"]["url"]
                logger.info(f"✅ Video generated successfully: {video_url}")
                return {
                    "video_url": video_url,
                    "image_path": image_path,
                    "image_url": image_url,
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "duration": "5s",
                    "fal_result": result,
                }
            else:
                logger.error(f"❌ Failed to generate video")
                return None

        except Exception as e:
            logger.error(f"❌ Error generating video: {e}")
            return None

    def generate_videos_for_broll_segments(
        self,
        prompts_data: Dict,
        images_dir: str = None,
        aspect_ratio: str = IMAGE_ASPECT_RATIO,
    ) -> List[Dict]:
        # Use default images directory if not specified
        if images_dir is None:
            # Use data_mock when mock mode is enabled
            import sys

            sys.path.append(str(Path(__file__).parent.parent))
            from config import config

            base_data_dir = "data_mock" if config.is_mock_enabled else "data"
            images_dir = str(
                Path(__file__).parent.parent
                / f"{base_data_dir}/video_generation/images_input"
            )
        """
        Generate videos for all b-roll segments

        Args:
            prompts_data: Dictionary with b-roll prompts data
            images_dir: Directory containing generated images
            aspect_ratio: Video aspect ratio (default: IMAGE_ASPECT_RATIO)

        Returns:
            List of dictionaries with generation results
        """
        results = []
        segments = prompts_data.get("broll_segments", [])

        if not segments:
            logger.error("❌ No b-roll segments found in prompts data")
            return results

        logger.info(
            f"🎬 Generating videos for {len(segments)} b-roll segments..."
        )

        for i, segment in enumerate(segments, 1):
            logger.info(f"\n{'='*40}")
            logger.info(f"🎬 SEGMENT {i}: {segment.get('segment_id', i)}")
            logger.info(f"{'='*40}")

            # Get video prompt from segment
            video_prompt = segment.get("video_prompt", "")
            if not video_prompt:
                logger.error(f"❌ No video prompt found for segment {i}")
                continue

            # Generate image filename based on segment info
            segment_id = segment.get("segment_id", i)
            start_time = segment.get("start_time", 0)
            # Convert IMAGE_ASPECT_RATIO to filename format
            aspect_ratio_filename = IMAGE_ASPECT_RATIO.replace(":", "x")
            image_filename = f"broll_segment_{segment_id:02d}_{start_time:.1f}s_{aspect_ratio_filename}.png"
            image_path = os.path.join(images_dir, image_filename)

            # Check if image exists
            if not os.path.exists(image_path):
                logger.error(f"❌ Image not found: {image_path}")
                continue

            # Generate video filename
            video_filename = f"broll_segment_{segment_id:02d}_{start_time:.1f}s_{aspect_ratio_filename}.mp4"

            # Generate video
            video_result = self.generate_video_from_image(
                image_path=image_path,
                prompt=video_prompt,
                aspect_ratio=aspect_ratio,
            )

            if video_result:
                # Download video
                download_success = self.download_video(
                    video_result["video_url"], video_filename
                )

                video_result["download_success"] = download_success
                video_result["local_filename"] = video_filename
                video_result["segment_info"] = {
                    "segment_id": segment.get("segment_id"),
                    "start_time": segment.get("start_time"),
                    "end_time": segment.get("end_time"),
                    "text_context": segment.get("text_context", ""),
                    "keywords": segment.get("keywords", []),
                    "importance_score": segment.get("importance_score", 0),
                }

                if download_success:
                    logger.info(f"✅ Segment {i} successful!")
                    logger.info(f"📁 File: {video_filename}")
                    logger.info(f"📐 Aspect Ratio: {aspect_ratio}")
                    logger.info(
                        f"🏷️ Keywords: {', '.join(segment.get('keywords', []))}"
                    )
                else:
                    logger.error(f"❌ Segment {i} failed: Download error")
            else:
                logger.error(f"❌ Segment {i} failed: Generation error")
                video_result = {
                    "error": "Generation failed",
                    "segment_info": {
                        "segment_id": segment.get("segment_id"),
                        "start_time": segment.get("start_time"),
                        "end_time": segment.get("end_time"),
                    },
                }

            results.append(video_result)

            # Save individual report
            self.save_generation_report(video_result)

            # Wait between requests
            if i < len(segments):
                logger.info("⏱️ Waiting 3 seconds before next segment...")
                time.sleep(3)

        return results

    def download_video(self, video_url: str, filename: str) -> bool:
        """
        Download video from URL

        Args:
            video_url: URL of the generated video
            filename: Local filename to save the video

        Returns:
            True if download successful, False otherwise
        """
        try:
            logger.info(f"⬇️ Downloading video: {filename}")

            response = requests.get(video_url, stream=True)
            response.raise_for_status()

            file_path = self.output_dir / filename

            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"✅ Video downloaded: {file_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Error downloading video {filename}: {e}")
            return False

    def save_generation_report(self, video_result: Dict):
        """
        Save generation report

        Args:
            video_result: Generated video information
        """
        report = {
            "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "api_used": "fal.ai Kling 1.6 Pro Image-to-Video",
            "generation_type": "image-to-video",
            "video_result": video_result,
            "success": video_result.get("download_success", False),
        }

        report_file = self.output_dir / "kling_image_to_video_report.json"

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"📊 Report saved to: {report_file}")


def test_kling_image_to_video():
    """
    Test Kling 1.6 Pro Image-to-Video generation with b-roll prompts
    """
    logger.info("🧪 Testing Kling 1.6 Pro Image-to-Video Generator")
    logger.info("=" * 60)

    try:
        # Initialize generator
        generator = KlingImageToVideoGenerator()

        # Load b-roll prompts
        prompts_data = generator.load_broll_prompts()
        if not prompts_data:
            logger.error("❌ Failed to load b-roll prompts")
            return False

        # Generate videos for all segments
        results = generator.generate_videos_for_broll_segments(
            prompts_data, aspect_ratio=IMAGE_ASPECT_RATIO
        )

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 GENERATION SUMMARY")
        logger.info(f"{'='*60}")

        successful_generations = sum(
            1 for r in results if r.get("download_success", False)
        )
        total_segments = len(results)

        logger.info(
            f"✅ Successful: {successful_generations}/{total_segments}"
        )
        logger.info(f"📁 Output directory: {generator.output_dir}")

        if successful_generations > 0:
            logger.info(f"🎉 B-roll video generation working!")
            logger.info(
                f"📊 Audio duration: {prompts_data.get('audio_duration', 0):.2f} seconds"
            )
            logger.info(
                f"🎬 B-roll segments: {len(prompts_data.get('broll_segments', []))}"
            )
            return True
        else:
            logger.error(f"❌ All generations failed")
            return False

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def generate_broll_videos(
    prompts_file: str = None,
    images_dir: str = None,
    aspect_ratio: str = IMAGE_ASPECT_RATIO,
) -> bool:
    # Use default paths if not specified
    if prompts_file is None:
        # Use data_mock when mock mode is enabled
        import sys

        sys.path.append(str(Path(__file__).parent.parent))
        from config import config

        base_data_dir = "data_mock" if config.is_mock_enabled else "data"
        prompts_file = str(
            Path(__file__).parent.parent
            / f"{base_data_dir}/video_generation/broll_prompts_api_generated.json"
        )
    if images_dir is None:
        # Use data_mock when mock mode is enabled
        import sys

        sys.path.append(str(Path(__file__).parent.parent))
        from config import config

        base_data_dir = "data_mock" if config.is_mock_enabled else "data"
        images_dir = str(
            Path(__file__).parent.parent
            / f"{base_data_dir}/video_generation/images_input"
        )
    """
    Generate videos for all b-roll segments from prompts file

    Args:
        prompts_file: Path to the prompts JSON file
        images_dir: Directory containing generated images
        aspect_ratio: Video aspect ratio (default: IMAGE_ASPECT_RATIO)

    Returns:
        True if generation successful, False otherwise
    """
    logger.info("🎬 Starting B-roll Video Generation")
    logger.info("=" * 60)

    try:
        # Initialize generator
        generator = KlingImageToVideoGenerator()

        # Load b-roll prompts
        prompts_data = generator.load_broll_prompts(prompts_file)
        if not prompts_data:
            logger.error("❌ Failed to load b-roll prompts")
            return False

        # Generate videos for all segments
        results = generator.generate_videos_for_broll_segments(
            prompts_data, images_dir, aspect_ratio
        )

        # Summary
        successful_generations = sum(
            1 for r in results if r.get("download_success", False)
        )
        total_segments = len(results)

        logger.info(f"\n{'='*60}")
        logger.info(f"📊 GENERATION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(
            f"✅ Successful: {successful_generations}/{total_segments}"
        )
        logger.info(f"📁 Output directory: {generator.output_dir}")

        if successful_generations > 0:
            logger.info(f"🎉 B-roll video generation completed!")
            logger.info(
                f"📊 Audio duration: {prompts_data.get('audio_duration', 0):.2f} seconds"
            )
            logger.info(
                f"🎬 B-roll segments: {len(prompts_data.get('broll_segments', []))}"
            )
            return True
        else:
            logger.error(f"❌ All generations failed")
            return False

    except Exception as e:
        logger.error(f"❌ Generation failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run the image-to-video test"""
    success = test_kling_image_to_video()

    if success:
        logger.info(
            f"\n🚀 Kling 1.6 Pro Image-to-Video test completed successfully!"
        )
    else:
        logger.error(f"\n❌ Test failed. Please check the errors above.")

    return success


if __name__ == "__main__":
    main()
