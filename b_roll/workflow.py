#!/usr/bin/env python3
"""
Unified Workflow for Audio Transcript Processing
Converts audio transcript to b-roll prompts, generates images, and creates videos.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

# Import configuration and modules
sys.path.append(str(Path(__file__).parent))
from b_roll_generation import (
    BRollAnalyzer,
    ImageGenerator,
    KlingImageToVideoGenerator,
)
from config import config
from constants import (
    DEFAULT_IMAGES_OUTPUT_DIR,
    DEFAULT_SEED,
    DEFAULT_VIDEOS_OUTPUT_DIR,
    ENABLE_VIDEO_GENERATION,
    IMAGE_ASPECT_RATIO,
    VIDEO_DURATION,
    VIDEO_FPS,
    VIDEO_RESOLUTION,
)
from logger_config import logger


def convert_paths_to_strings(obj):
    """Convert all PosixPath objects to strings for JSON serialization."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {
            key: convert_paths_to_strings(value) for key, value in obj.items()
        }
    elif isinstance(obj, list):
        return [convert_paths_to_strings(item) for item in obj]
    else:
        return obj


class UnifiedWorkflow:
    """
    Unified workflow for processing audio transcripts through the complete pipeline:
    1. Analyze transcript and generate b-roll prompts
    2. Generate images from prompts
    3. Convert images to videos
    """

    def __init__(self, max_segments: int = 3):
        """
        Initialize the unified workflow with all components.

        Args:
            max_segments: Maximum number of segments to generate (default: 3)
        """
        self.broll_analyzer = BRollAnalyzer(
            segment_duration=VIDEO_DURATION, max_segments=max_segments
        )
        self.image_generator = ImageGenerator()
        self.video_generator = KlingImageToVideoGenerator()
        self.max_segments = max_segments

        # Workflow directories
        # Use data_mock directory when mock mode is enabled
        base_data_dir = "data_mock" if config.is_mock_enabled else "data"
        self.data_dir = (
            Path(__file__).parent / f"{base_data_dir}/video_generation"
        )
        self.transcript_dir = self.data_dir / "audio_transcript"
        self.prompts_dir = self.data_dir / "broll_prompts"
        # Fix path construction to avoid duplication
        self.images_dir = Path(DEFAULT_IMAGES_OUTPUT_DIR)
        self.videos_dir = Path(DEFAULT_VIDEOS_OUTPUT_DIR)

        # Create directories if they don't exist
        for directory in [
            self.transcript_dir,
            self.prompts_dir,
            self.images_dir,
            self.videos_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def process_transcript_to_prompts(self, transcript_file: str) -> Dict:
        """
        Step 1: Process audio transcript and generate b-roll prompts

        Args:
            transcript_file: Path to the transcript JSON file

        Returns:
            Dictionary with b-roll prompts and metadata
        """
        logger.info("🎯 STEP 1: Processing transcript to b-roll prompts")
        logger.info("=" * 60)
        logger.info(f"🎯 Target segments: {self.max_segments}")

        try:
            # Load and analyze transcript
            transcript_data = self.broll_analyzer.load_transcript(
                transcript_file
            )
            broll_segments = self.broll_analyzer.analyze_transcript_with_ai(
                transcript_data
            )

            # Limit segments to max_segments
            if len(broll_segments) > self.max_segments:
                logger.info(
                    f"📊 Limiting segments from {len(broll_segments)} to {self.max_segments}"
                )
                broll_segments = broll_segments[: self.max_segments]

            # Generate output JSON
            output_data = self.broll_analyzer.generate_output_json(
                broll_segments, transcript_data
            )

            # Save prompts
            prompts_file = self.prompts_dir / "workflow_generated_prompts.json"
            with open(prompts_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ Generated {len(broll_segments)} b-roll prompts")
            logger.info(f"📁 Saved to: {prompts_file}")

            return output_data

        except Exception as e:
            logger.info(f"❌ Error in transcript processing: {e}")
            raise

    def generate_images_from_prompts(self, prompts_data: Dict) -> List[Dict]:
        """
        Step 2: Generate images from b-roll prompts

        Args:
            prompts_data: Dictionary with b-roll prompts from step 1

        Returns:
            List of generated image information
        """
        logger.info("\n🎨 STEP 2: Generating images from prompts")
        logger.info("=" * 60)

        image_results = []
        segments = prompts_data.get("broll_segments", [])

        for i, segment in enumerate(segments, 1):
            logger.info(f"\n🎨 Processing segment {i}/{len(segments)}")
            logger.info(
                f"⏰ Timing: {segment['start_time']}s - {segment['end_time']}s"
            )

            try:
                # Generate image from prompt
                image_result = self.image_generator.generate_image(
                    prompt=segment["image_prompt"],
                    aspect_ratio=IMAGE_ASPECT_RATIO,  # Use aspect ratio from constants
                    seed=DEFAULT_SEED + i,  # Different seed for each segment
                )

                if image_result:
                    # Download image with descriptive filename
                    filename = f"segment_{i:02d}_{segment['start_time']:.1f}s_{segment['end_time']:.1f}s.png"
                    download_success = self.image_generator.download_image(
                        image_result["image_url"], filename
                    )

                    if download_success:
                        image_result["download_success"] = True
                        image_result["local_filename"] = filename
                        image_result["segment_info"] = segment
                        image_results.append(image_result)

                        logger.info(f"✅ Image generated: {filename}")
                    else:
                        logger.info(
                            f"❌ Failed to download image for segment {i}"
                        )
                else:
                    logger.info(f"❌ Failed to generate image for segment {i}")

            except Exception as e:
                logger.info(f"❌ Error generating image for segment {i}: {e}")

            # Wait between requests to avoid rate limiting
            if i < len(segments):
                logger.info("⏱️ Waiting 2 seconds...")
                time.sleep(2)

        logger.info(f"\n✅ Generated {len(image_results)} images successfully")
        return image_results

    def create_videos_from_images(
        self, image_results: List[Dict]
    ) -> List[Dict]:
        """
        Step 3: Convert images to videos

        Args:
            image_results: List of generated image information from step 2

        Returns:
            List of generated video information
        """
        logger.info("\n🎬 STEP 3: Creating videos from images")
        logger.info("=" * 60)

        video_results = []

        for i, image_result in enumerate(image_results, 1):
            logger.info(f"\n🎬 Processing video {i}/{len(image_results)}")

            try:
                image_path = self.images_dir / image_result["local_filename"]
                segment_info = image_result.get("segment_info", {})

                # Use video prompt from segment
                video_prompt = segment_info.get(
                    "video_prompt",
                    f"Professional video sequence with subtle movements and natural lighting. {segment_info.get('text_context', '')}",
                )

                # Generate video from image
                video_result = self.video_generator.generate_video_from_image(
                    image_path=str(image_path),
                    prompt=video_prompt,
                    aspect_ratio=IMAGE_ASPECT_RATIO,
                    fps=VIDEO_FPS,
                    resolution=VIDEO_RESOLUTION,
                )

                if video_result:
                    # Download video with descriptive filename
                    base_name = image_result["local_filename"].replace(
                        ".png", ""
                    )
                    video_filename = f"{base_name}_video.mp4"

                    download_success = self.video_generator.download_video(
                        video_result["video_url"], video_filename
                    )

                    if download_success:
                        video_result["download_success"] = True
                        video_result["local_filename"] = video_filename
                        video_result["image_info"] = image_result
                        video_result["segment_info"] = segment_info
                        video_results.append(video_result)

                        logger.info(f"✅ Video created: {video_filename}")
                    else:
                        logger.info(
                            f"❌ Failed to download video for segment {i}"
                        )
                else:
                    logger.info(f"❌ Failed to generate video for segment {i}")

            except Exception as e:
                logger.info(f"❌ Error creating video for segment {i}: {e}")

            # Wait between requests
            if i < len(image_results):
                logger.info("⏱️ Waiting 3 seconds...")
                time.sleep(3)

        logger.info(f"\n✅ Created {len(video_results)} videos successfully")
        return video_results

    def run_complete_workflow(self, transcript_file: str) -> Dict:
        """
        Run the complete workflow from transcript to videos

        Args:
            transcript_file: Path to the transcript JSON file

        Returns:
            Dictionary with complete workflow results
        """
        logger.info("🚀 STARTING UNIFIED WORKFLOW")
        logger.info("=" * 60)
        logger.info(f"📁 Input transcript: {transcript_file}")
        video_step = (
            "→ Videos" if ENABLE_VIDEO_GENERATION else "→ [Videos SKIPPED]"
        )
        logger.info(f"🎯 Workflow: Transcript → Prompts → Images {video_step}")
        logger.info(
            f"🎬 Video generation: {'ENABLED' if ENABLE_VIDEO_GENERATION else 'DISABLED'}"
        )
        logger.info("=" * 60)

        start_time = time.time()

        try:
            # Step 1: Process transcript to prompts
            prompts_data = self.process_transcript_to_prompts(transcript_file)

            # Step 2: Generate images from prompts
            image_results = self.generate_images_from_prompts(prompts_data)

            # Step 3: Create videos from images (if enabled)
            if ENABLE_VIDEO_GENERATION:
                video_results = self.create_videos_from_images(image_results)
            else:
                logger.info(
                    "\n⏭️ SKIPPING VIDEO GENERATION (ENABLE_VIDEO_GENERATION = False)"
                )
                video_results = []

            # Compile final results
            workflow_results = {
                "workflow_info": {
                    "input_transcript": transcript_file,
                    "total_duration": time.time() - start_time,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "api_used": config.is_api_enabled,
                    "max_segments": self.max_segments,
                },
                "prompts_data": prompts_data,
                "image_results": image_results,
                "video_results": video_results,
                "summary": {
                    "total_segments": len(
                        prompts_data.get("broll_segments", [])
                    ),
                    "images_generated": len(image_results),
                    "videos_created": len(video_results),
                    "success_rate": {
                        "images": f"{len(image_results)}/{len(prompts_data.get('broll_segments', []))}",
                        "videos": f"{len(video_results)}/{len(image_results)}",
                    },
                },
            }

            # Save complete workflow report
            report_file = self.data_dir / "workflow_complete_report.json"
            # Convert all Path objects to strings for JSON serialization
            serializable_results = convert_paths_to_strings(workflow_results)
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(
                    serializable_results, f, ensure_ascii=False, indent=2
                )

            # Print final summary
            self.print_workflow_summary(workflow_results)

            return workflow_results

        except Exception as e:
            logger.info(f"❌ Workflow failed: {e}")
            traceback.print_exc()
            raise

    def print_workflow_summary(self, results: Dict):
        """Print a comprehensive summary of the workflow results."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 WORKFLOW COMPLETE - FINAL SUMMARY")
        logger.info("=" * 60)

        summary = results["summary"]
        workflow_info = results["workflow_info"]

        logger.info(
            f"⏱️ Total processing time: {workflow_info['total_duration']:.1f} seconds"
        )
        logger.info(
            f"🎯 Max segments configured: {workflow_info.get('max_segments', 'N/A')}"
        )
        logger.info(
            f"📝 Transcript segments analyzed: {summary['total_segments']}"
        )
        logger.info(f"🎨 Images generated: {summary['images_generated']}")
        if ENABLE_VIDEO_GENERATION:
            logger.info(f"🎬 Videos created: {summary['videos_created']}")
        else:
            logger.info(f"🎬 Videos created: [SKIPPED]")
        logger.info(f"📈 Success rates:")
        logger.info(f"   • Images: {summary['success_rate']['images']}")
        if ENABLE_VIDEO_GENERATION:
            logger.info(f"   • Videos: {summary['success_rate']['videos']}")
        else:
            logger.info(f"   • Videos: [SKIPPED]")

        logger.info(f"\n📁 Output files:")
        logger.info(f"   • Prompts: {self.prompts_dir}")
        logger.info(f"   • Images: {self.images_dir}")
        if ENABLE_VIDEO_GENERATION:
            logger.info(f"   • Videos: {self.videos_dir}")
        else:
            logger.info(f"   • Videos: [SKIPPED]")
        logger.info(
            f"   • Report: {self.data_dir}/workflow_complete_report.json"
        )

        if ENABLE_VIDEO_GENERATION and summary["videos_created"] > 0:
            logger.info(f"\n🎉 Workflow completed successfully!")
            logger.info(f"🚀 Ready to use generated videos for your content!")
        elif not ENABLE_VIDEO_GENERATION:
            logger.info(
                f"\n✅ Workflow completed successfully (videos skipped)!"
            )
            logger.info(
                f"🎨 Generated images are ready for manual video creation!"
            )
        else:
            logger.info(f"\n⚠️ Workflow completed with some failures")
            logger.info(f"🔍 Check the logs above for details")


def main():
    """Main function to run the unified workflow."""
    # Default transcript file path - use data_mock when mock mode is enabled
    base_data_dir = "data_mock" if config.is_mock_enabled else "data"
    DEFAULT_TRANSCRIPT = (
        Path(__file__).parent
        / f"{base_data_dir}/video_generation/audio_transcript/transcription_verbose_to_json.json"
    )

    # Check if transcript file exists
    if not DEFAULT_TRANSCRIPT.exists():
        logger.info(f"❌ Transcript file not found: {DEFAULT_TRANSCRIPT}")
        logger.info("Please ensure you have a valid transcript JSON file.")
        return False

    try:
        # Example: Create workflow with custom number of segments
        # You can change this value to control how many segments to generate
        MAX_SEGMENTS = 5  # Change this to control number of segments

        logger.info(
            f"🎯 Configuring workflow with max {MAX_SEGMENTS} segments"
        )

        # Initialize and run workflow
        workflow = UnifiedWorkflow(max_segments=MAX_SEGMENTS)
        results = workflow.run_complete_workflow(DEFAULT_TRANSCRIPT)

        return results["summary"]["videos_created"] > 0

    except Exception as e:
        logger.info(f"❌ Workflow execution failed: {e}")
        return False


if __name__ == "__main__":
    success = main()

    if success:
        logger.info(f"\n✅ Unified workflow completed successfully!")
    else:
        logger.info(f"\n❌ Workflow failed. Please check the errors above.")
