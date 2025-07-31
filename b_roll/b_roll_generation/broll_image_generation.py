#!/usr/bin/env python3
"""
Image Generator using fal.ai API

This script generates images using fal.ai API
with various prompts for business and professional content.
"""

import json
import os

# Import configuration and mock modules
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
import mock_api
from config import config
from constants import (
    DEFAULT_IMAGES_OUTPUT_DIR,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_PROMPTS_FILE,
    DEFAULT_SEED,
    FAL_MODEL_ENDPOINT,
    IMAGE_ASPECT_RATIO,
)
from logger_config import logger
from mock_api import mock_fal_client, mock_requests

from .prompts import NEGATIVE_PROMPT_IMAGE

# Import real modules only if API is enabled
if config.is_api_enabled:
    import fal_client
    import requests
else:
    # Use mock modules
    fal_client = mock_fal_client
    requests = mock_requests
    # Use MockImageGenerator for testing
    ImageGenerator = mock_api.mock_image_generator

# Model configuration and constants imported from constants.py


class ImageGenerator:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the fal.ai Image Generator

        Args:
            api_key: fal.ai API key. If None, will try to get from .env file
        """
        # Load environment variables from .env file in project root
        project_root = Path(__file__).parent.parent
        env_file = project_root / ".env"
        load_dotenv(env_file)

        # Get API key based on configuration
        self.api_key = api_key or config.get_api_key("FAL_KEY")

        if config.is_api_enabled and not self.api_key:
            logger.info(f"⚠️ Warning: FAL_KEY not found in {env_file}")
            logger.info("🔧 Using mock mode for image generation")
            # Don't raise error, continue with mock mode

        # Configure fal client only if API is enabled
        if config.is_api_enabled:
            fal_client.api_key = self.api_key
        else:
            logger.info("🔧 [MOCK] Using mock fal_client")

        # API endpoint for fal.ai
        self.model_endpoint = FAL_MODEL_ENDPOINT

        # Output directory
        self.output_dir = Path(DEFAULT_IMAGES_OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)

    def load_broll_prompts(
        self,
        prompts_file: str = DEFAULT_PROMPTS_FILE,
    ) -> Dict:
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
            logger.info(f"❌ Error loading prompts: {e}")
            return {}

    def generate_images_for_broll_segments(
        self, prompts_data: Dict, aspect_ratio: str = IMAGE_ASPECT_RATIO
    ) -> List[Dict]:
        """
        Generate images for all b-roll segments

        Args:
            prompts_data: Dictionary with b-roll prompts data
            aspect_ratio: Image aspect ratio (default: IMAGE_ASPECT_RATIO)

        Returns:
            List of dictionaries with generation results
        """
        results = []
        segments = prompts_data.get("broll_segments", [])

        if not segments:
            logger.info("❌ No b-roll segments found in prompts data")
            return results

        logger.info(
            f"🎨 Generating images for {len(segments)} b-roll segments..."
        )

        for i, segment in enumerate(segments, 1):
            logger.info(f"\n{'='*40}")
            logger.info(f"🎨 SEGMENT {i}: {segment.get('segment_id', i)}")
            logger.info(f"{'='*40}")

            # Get image prompt from segment
            image_prompt = segment.get("image_prompt", "")
            if not image_prompt:
                logger.info(f"❌ No image prompt found for segment {i}")
                continue

            # Generate filename based on segment info
            segment_id = segment.get("segment_id", i)
            start_time = segment.get("start_time", 0)
            filename = f"broll_segment_{segment_id:02d}_{start_time:.1f}s_{aspect_ratio.replace(':', 'x')}.png"

            # Generate image
            image_result = self.generate_image(
                prompt=image_prompt,
                aspect_ratio=aspect_ratio,
                seed=DEFAULT_SEED + i,  # Different seed for each segment
            )

            if image_result:
                # Download image
                download_success = self.download_image(
                    image_result["image_url"], filename
                )

                image_result["download_success"] = download_success
                image_result["local_filename"] = filename
                image_result["segment_info"] = {
                    "segment_id": segment.get("segment_id"),
                    "start_time": segment.get("start_time"),
                    "end_time": segment.get("end_time"),
                    "text_context": segment.get("text_context", ""),
                    "keywords": segment.get("keywords", []),
                    "importance_score": segment.get("importance_score", 0),
                }

                if download_success:
                    logger.info(f"✅ Segment {i} successful!")
                    logger.info(f"📁 File: {filename}")
                    logger.info(f"📐 Aspect Ratio: {aspect_ratio}")
                    logger.info(
                        f"🏷️ Keywords: {', '.join(segment.get('keywords', []))}"
                    )
                else:
                    logger.info(f"❌ Segment {i} failed: Download error")
            else:
                logger.info(f"❌ Segment {i} failed: Generation error")
                image_result = {
                    "error": "Generation failed",
                    "segment_info": {
                        "segment_id": segment.get("segment_id"),
                        "start_time": segment.get("start_time"),
                        "end_time": segment.get("end_time"),
                    },
                }

            results.append(image_result)

            # Save individual report
            self.save_generation_report(image_result)

            # Wait between requests
            if i < len(segments):
                logger.info("⏱️ Waiting 3 seconds before next segment...")
                time.sleep(3)

        return results

    def generate_image(
        self,
        prompt: str,
        aspect_ratio: str = IMAGE_ASPECT_RATIO,
        seed: Optional[int] = None,
    ) -> Optional[Dict]:
        """
        Generate image using fal.ai API

        Args:
            prompt: Image generation prompt
            aspect_ratio: Image aspect ratio (default: IMAGE_ASPECT_RATIO)
            seed: Random seed for reproducible results

        Returns:
            Dictionary with image information or None if failed
        """
        try:
            logger.info(f"🎨 Generating image...")
            logger.info(
                f"📝 Prompt: {prompt[:100]}..."
            )  # Truncate long prompts
            logger.info(f"📐 Aspect Ratio: {aspect_ratio}")

            # Prepare arguments
            arguments = {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "negative_prompt": NEGATIVE_PROMPT_IMAGE,
                "num_inference_steps": DEFAULT_NUM_INFERENCE_STEPS,
            }

            # Add seed if provided
            if seed is not None:
                arguments["seed"] = seed

            # Submit request to fal.ai
            logger.info("⏳ Submitting image generation request...")
            result = fal_client.subscribe(
                self.model_endpoint,
                arguments=arguments,
            )

            if result and "images" in result and len(result["images"]) > 0:
                image_url = result["images"][0]["url"]
                logger.info(f"✅ Image generated successfully: {image_url}")
                return {
                    "image_url": image_url,
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "seed": seed,
                    "fal_result": result,
                }
            else:
                logger.info(f"❌ Failed to generate image")
                return None

        except Exception as e:
            logger.info(f"❌ Error generating image: {e}")
            return None

    def download_image(self, image_url: str, filename: str) -> bool:
        """
        Download image from URL

        Args:
            image_url: URL of the generated image
            filename: Local filename to save the image

        Returns:
            True if download successful, False otherwise
        """
        try:
            logger.info(f"⬇️ Downloading image: {filename}")

            response = requests.get(image_url, stream=True)
            response.raise_for_status()

            file_path = self.output_dir / filename

            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"✅ Image downloaded: {file_path}")
            return True

        except Exception as e:
            logger.info(f"❌ Error downloading image {filename}: {e}")
            return False

    def save_generation_report(self, image_result: Dict):
        """
        Save generation report

        Args:
            image_result: Generated image information
        """
        report = {
            "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "api_used": "fal.ai",
            "model_endpoint": self.model_endpoint,
            "generation_type": "text-to-image",
            "image_result": image_result,
            "success": image_result.get("download_success", False),
        }

        report_file = self.output_dir / "image_generation_report.json"

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"📊 Report saved to: {report_file}")


def test_image_generation():
    """
    Test fal.ai image generation with b-roll prompts
    """
    logger.info("🧪 Testing fal.ai Image Generator with B-roll Prompts")
    logger.info("=" * 60)

    try:
        # Initialize generator
        generator = ImageGenerator()

        # Load b-roll prompts
        prompts_data = generator.load_broll_prompts()
        if not prompts_data:
            logger.info("❌ Failed to load b-roll prompts")
            return False

        # Generate images for all segments
        results = generator.generate_images_for_broll_segments(
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
            logger.info(f"🎉 B-roll image generation working!")
            logger.info(
                f"📊 Audio duration: {prompts_data.get('audio_duration', 0):.2f} seconds"
            )
            logger.info(
                f"🎬 B-roll segments: {len(prompts_data.get('broll_segments', []))}"
            )
            return True
        else:
            logger.info(f"❌ All generations failed")
            return False

    except Exception as e:
        logger.info(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def generate_broll_images(
    prompts_file: str = DEFAULT_PROMPTS_FILE,
    aspect_ratio: str = IMAGE_ASPECT_RATIO,
) -> bool:
    """
    Generate images for all b-roll segments from prompts file

    Args:
        prompts_file: Path to the prompts JSON file
        aspect_ratio: Image aspect ratio (default: IMAGE_ASPECT_RATIO)

    Returns:
        True if generation successful, False otherwise
    """
    logger.info("🎨 Starting B-roll Image Generation")
    logger.info("=" * 60)

    try:
        # Initialize generator
        generator = ImageGenerator()

        # Load b-roll prompts
        prompts_data = generator.load_broll_prompts(prompts_file)
        if not prompts_data:
            logger.info("❌ Failed to load b-roll prompts")
            return False

        # Generate images for all segments
        results = generator.generate_images_for_broll_segments(
            prompts_data, aspect_ratio
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
            logger.info(f"🎉 B-roll image generation completed!")
            logger.info(
                f"📊 Audio duration: {prompts_data.get('audio_duration', 0):.2f} seconds"
            )
            logger.info(
                f"🎬 B-roll segments: {len(prompts_data.get('broll_segments', []))}"
            )
            return True
        else:
            logger.info(f"❌ All generations failed")
            return False

    except Exception as e:
        logger.info(f"❌ Generation failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run the image generation test"""
    success = test_image_generation()

    if success:
        logger.info(
            f"\n🚀 fal.ai image generation test completed successfully!"
        )
    else:
        logger.info(f"\n❌ Test failed. Please check the errors above.")

    return success


if __name__ == "__main__":
    main()
