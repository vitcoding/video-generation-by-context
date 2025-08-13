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

# Replace fragile imports with robust package/direct execution handling
try:
    from .. import mock_api
    from ..config import config
    from ..constants import (
        BROLL_PROMPTS_DIR,
        DEFAULT_IMAGES_OUTPUT_DIR,
        DEFAULT_NUM_INFERENCE_STEPS,
        DEFAULT_SEED,
        ENV_FILENAME,
        FAL_MODEL_ENDPOINT,
        IMAGE_ASPECT_RATIO,
        PROJECT_ROOT_RELATIVE_PATH,
        VIDEO_DURATION,
        WORKFLOW_PROMPTS_PATH,
    )
    from ..logger_config import logger
    from ..mock_api import mock_fal_client, mock_requests
    from .prompts import NEGATIVE_PROMPT_IMAGE
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.append(str(_Path(__file__).resolve().parents[2]))
    from b_roll import mock_api
    from b_roll.b_roll_generation.prompts import NEGATIVE_PROMPT_IMAGE
    from b_roll.config import config
    from b_roll.constants import (
        BROLL_PROMPTS_DIR,
        DEFAULT_IMAGES_OUTPUT_DIR,
        DEFAULT_NUM_INFERENCE_STEPS,
        DEFAULT_SEED,
        ENV_FILENAME,
        FAL_MODEL_ENDPOINT,
        IMAGE_ASPECT_RATIO,
        PROJECT_ROOT_RELATIVE_PATH,
        VIDEO_DURATION,
        WORKFLOW_PROMPTS_PATH,
    )
    from b_roll.logger_config import logger
    from b_roll.mock_api import mock_fal_client, mock_requests

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
        env_file = project_root / ENV_FILENAME
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

    def _contains_humans(self, text: str) -> bool:
        """Heuristic check whether the prompt likely includes people."""
        if not text:
            return False
        t = text.lower()
        human_keywords = [
            "person",
            "people",
            "man",
            "woman",
            "men",
            "women",
            "adult",
            "team",
            "group",
            "crowd",
            "colleague",
            "coworker",
            "employees",
            "worker",
            "business team",
            "audience",
            "speaker",
            "meeting",
            "office team",
            "portrait",
            "headshot",
            "couple",
            "family",
        ]
        return any(k in t for k in human_keywords)

    def load_broll_prompts(
        self,
        prompts_file: str = WORKFLOW_PROMPTS_PATH,
    ) -> Dict:
        """
        Load b-roll prompts from JSON file

        Args:
            prompts_file: Path to the prompts JSON file

        Returns:
            Dictionary with b-roll prompts data
        """
        try:
            # Resolve to absolute path relative to project root if needed
            file_path = Path(prompts_file)
            if not file_path.is_absolute():
                project_root = Path(__file__).parents[2]
                file_path = (project_root / file_path).resolve()

            if not file_path.exists():
                raise FileNotFoundError(f"Prompts file not found: {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                prompts_data = json.load(f)

            logger.info(
                f"✅ Loaded {len(prompts_data.get('broll_segments', []))} b-roll segments from {file_path}"
            )
            return prompts_data

        except Exception as e:
            logger.info(f"❌ Error loading prompts: {e}")
            return {}

    def load_global_style(self) -> Dict:
        """
        Load optional global style JSON from b-roll prompts directory.
        """
        try:
            style_path = Path(BROLL_PROMPTS_DIR) / "global_style.json"
            if style_path.exists():
                with open(style_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def compose_prompt(self, base_prompt: str, style: Dict) -> str:
        """
        Merge base image prompt with global style fields.
        """
        if not style:
            return base_prompt

        tokens: List[str] = []
        prefix = str(style.get("prefix", "")).strip()
        if prefix:
            tokens.append(prefix)

        if base_prompt:
            tokens.append(base_prompt.strip())

        def add_list(name: str) -> None:
            vals = style.get(name)
            if isinstance(vals, list) and vals:
                tokens.append(", ".join([str(v) for v in vals]))

        add_list("style_tags")
        add_list("camera")
        add_list("lighting")
        add_list("color_palette")

        # Conditionally enforce human demographics when people are present
        combined_so_far = ". ".join([t for t in tokens if t])
        demographics = str(
            style.get("human_demographics", "Caucasian, middle-aged adults")
        ).strip()
        if demographics:
            lower_text = combined_so_far.lower()
            conflicting_terms = [
                "asian",
                "african",
                "black",
                "latino",
                "hispanic",
                "indian",
                "arab",
                "elderly",
                "senior",
                "young",
                "teen",
                "child",
                "middle-aged",
                "caucasian",
                "white",
            ]
            if self._contains_humans(combined_so_far) and not any(
                term in lower_text for term in conflicting_terms
            ):
                tokens.append(demographics)

        suffix = str(style.get("suffix", "")).strip()
        if suffix:
            tokens.append(suffix)

        return ". ".join([t for t in tokens if t])

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

        # Load optional global style once
        global_style = self.load_global_style()
        style_negative = (
            (global_style.get("negative") or "").strip()
            if global_style
            else ""
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

            # Apply global style
            final_prompt = self.compose_prompt(image_prompt, global_style)

            # Generate filename based on segment info
            segment_id = segment.get("segment_id", i)
            start_time = segment.get("start_time", 0)
            end_time = segment.get(
                "end_time", (start_time or 0) + VIDEO_DURATION
            )
            filename = f"segment_{segment_id:02d}_{start_time:.1f}s_{end_time:.1f}s.png"

            # Determine seed (use global if provided, else per-segment)
            global_seed = global_style.get("seed") if global_style else None
            seed_to_use = (
                int(global_seed)
                if isinstance(global_seed, int)
                else (DEFAULT_SEED + i)
            )

            # Generate image
            image_result = self.generate_image(
                prompt=final_prompt,
                aspect_ratio=aspect_ratio,
                seed=seed_to_use,  # Different seed for each segment unless global provided
                negative_prompt=(
                    f"{NEGATIVE_PROMPT_IMAGE}, {style_negative}".strip(", ")
                    if style_negative
                    else NEGATIVE_PROMPT_IMAGE
                ),
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
        negative_prompt: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Generate image using fal.ai API

        Args:
            prompt: Image generation prompt
            aspect_ratio: Image aspect ratio (default: IMAGE_ASPECT_RATIO)
            seed: Random seed for reproducible results
            negative_prompt: Optional override for negative prompt

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
                "negative_prompt": negative_prompt or NEGATIVE_PROMPT_IMAGE,
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
    prompts_file: str = WORKFLOW_PROMPTS_PATH,
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
