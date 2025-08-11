#!/usr/bin/env python3
"""
Unified Workflow for Audio Transcript Processing
Converts audio transcript to b-roll prompts, generates images, and creates videos.

NEW FUNCTIONALITY:

1. Flexible workflow execution:
   - start_stage="transcription" - start from video transcription (default)
   - start_stage="analysis" - start from existing transcript analysis

2. Skip stages:
   - skip_video_generation=True - skip video generation and final video editing

3. Convenience methods:
   - run_from_video() - run from video file
   - run_from_transcript() - run from transcript
   - run_images_only() - generate images only

Usage examples:
   # Complete workflow from video
   workflow = UnifiedWorkflow(max_segments=3)
   results = workflow.run_from_video("video.mp4")

   # Start from existing transcript (uses default path automatically)
   workflow = UnifiedWorkflow(start_stage="analysis")
   results = workflow.run_from_transcript()  # Uses default transcript file

   # Start from custom transcript file
   workflow = UnifiedWorkflow(start_stage="analysis")
   results = workflow.run_from_transcript("custom/transcript.json")

   # Generate images only
   workflow = UnifiedWorkflow(skip_video_generation=True)
   results = workflow.run_images_only(video_file_path="video.mp4")
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
    VideoTranscriber,
    process_video_with_broll,
)
from config import config
from constants import (
    AUDIO_TRANSCRIPT_DIR,
    AUDIO_TRANSCRIPT_DIR_NAME,
    BASE_DATA_DIR_PATH,
    BASE_RELATIVE_PATH,
    BROLL_PROMPTS_DIR,
    BROLL_PROMPTS_DIR_NAME,
    DEFAULT_IMAGES_OUTPUT_DIR,
    DEFAULT_SEED,
    DEFAULT_VIDEO_PATH,
    DEFAULT_VIDEOS_OUTPUT_DIR,
    ENABLE_VIDEO_GENERATION,
    IMAGE_ASPECT_RATIO,
    INPUT_VIDEO_DIR,
    INPUT_VIDEO_DIR_NAME,
    MAX_SEGMENTS,
    TRANSCRIPTION_JSON_FILENAME,
    TRANSCRIPTION_JSON_PATH,
    VIDEO_DURATION,
    VIDEO_FPS,
    VIDEO_OUTPUT_DIR,
    VIDEO_OUTPUT_DIR_NAME,
    VIDEO_RESOLUTION,
    WORKFLOW_PROMPTS_FILENAME,
    WORKFLOW_REPORT_FILENAME,
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
    Unified workflow for processing video through the complete pipeline:
    1. Transcribe video file to get word-level timestamps
    2. Analyze transcript and generate b-roll prompts
    3. Generate images from prompts
    4. Convert images to videos
    5. Insert b-roll videos into main video
    """

    def __init__(
        self,
        max_segments: int = 3,
        start_stage: str = "transcription",
        skip_video_generation: bool = False,
        early_segment_ratio: float = None,
        early_duration_ratio: float = None,
    ):
        """
        Initialize the unified workflow with all components.

        Args:
            max_segments: Maximum number of segments to generate (default: 3)
            start_stage: Stage to start from ('transcription' or 'analysis')
            skip_video_generation: Skip video generation and final video editing (default: False)
            early_segment_ratio: Ratio of segments for early portion (default: from constants)
            early_duration_ratio: Ratio of video duration for early portion (default: from constants)
        """
        self.video_transcriber = VideoTranscriber()
        self.broll_analyzer = BRollAnalyzer(
            segment_duration=VIDEO_DURATION,
            max_segments=max_segments,
            early_segment_ratio=early_segment_ratio,
            early_duration_ratio=early_duration_ratio,
        )
        self.image_generator = ImageGenerator()
        self.video_generator = KlingImageToVideoGenerator()
        self.max_segments = max_segments

        # Workflow control parameters
        self.start_stage = start_stage.lower()
        self.skip_video_generation = skip_video_generation

        # Validate start_stage parameter
        if self.start_stage not in ["transcription", "analysis"]:
            raise ValueError(
                "start_stage must be 'transcription' or 'analysis'"
            )

        # Workflow directories using constants
        self.data_dir = Path(__file__).parent / BASE_RELATIVE_PATH
        self.transcript_dir = Path(AUDIO_TRANSCRIPT_DIR)
        self.prompts_dir = Path(BROLL_PROMPTS_DIR)
        self.input_video_dir = Path(INPUT_VIDEO_DIR)
        self.final_video_dir = Path(VIDEO_OUTPUT_DIR)
        self.images_dir = Path(DEFAULT_IMAGES_OUTPUT_DIR)
        self.videos_dir = Path(DEFAULT_VIDEOS_OUTPUT_DIR)

        # Create directories if they don't exist
        for directory in [
            self.transcript_dir,
            self.prompts_dir,
            self.input_video_dir,
            self.final_video_dir,
            self.images_dir,
            self.videos_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def transcribe_video_file(self, video_file_path: str) -> str:
        """
        Step 1: Transcribe video file to get word-level timestamps

        Args:
            video_file_path: Path to the input video file

        Returns:
            Path to the generated transcript JSON file
        """
        logger.info("🎵 STEP 1: Transcribing video file")
        logger.info("=" * 60)

        try:
            # Set output path for transcript
            transcript_file = self.transcript_dir / TRANSCRIPTION_JSON_FILENAME

            # Transcribe video
            result_path = self.video_transcriber.transcribe_video_to_json(
                video_file_path=video_file_path,
                output_json_path=str(transcript_file),
            )

            logger.info(f"✅ Video transcription completed")
            logger.info(f"📁 Saved to: {result_path}")

            return result_path

        except Exception as e:
            logger.info(f"❌ Error in video transcription: {e}")
            raise

    def process_transcript_to_prompts(self, transcript_file: str) -> Dict:
        """
        Step 2: Process audio transcript and generate b-roll prompts

        Args:
            transcript_file: Path to the transcript JSON file

        Returns:
            Dictionary with b-roll prompts and metadata
        """
        logger.info("\n🎯 STEP 2: Processing transcript to b-roll prompts")
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
            prompts_file = self.prompts_dir / WORKFLOW_PROMPTS_FILENAME
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
        Step 3: Generate images from b-roll prompts

        Args:
            prompts_data: Dictionary with b-roll prompts from step 2

        Returns:
            List of generated image information
        """
        logger.info("\n🎨 STEP 3: Generating images from prompts")
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
                    # Use segment_id from data instead of loop index for consistency
                    segment_id = segment.get("segment_id", i)
                    filename = f"segment_{segment_id:02d}_{segment['start_time']:.1f}s_{segment['end_time']:.1f}s.png"
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
        Step 4: Convert images to videos

        Args:
            image_results: List of generated image information from step 3

        Returns:
            List of generated video information
        """
        logger.info("\n🎬 STEP 4: Creating videos from images")
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

    def insert_broll_into_main_video(
        self, main_video_path: str, video_results: List[Dict]
    ) -> str:
        """
        Step 5: Insert b-roll videos into main video

        Args:
            main_video_path: Path to the main video file
            video_results: List of generated video information from step 4

        Returns:
            Path to the final video with b-roll inserted
        """
        logger.info("\n🎞️ STEP 5: Inserting b-roll into main video")
        logger.info("=" * 60)

        try:
            # Process video with b-roll
            final_video_path = process_video_with_broll(
                heygen_video_path=main_video_path,
                output_folder=str(self.final_video_dir),
            )

            logger.info(f"✅ Final video created with b-roll insertions")
            logger.info(f"📁 Saved to: {final_video_path}")

            return final_video_path

        except Exception as e:
            logger.info(f"❌ Error in video editing: {e}")
            raise

    def run_complete_workflow(
        self,
        video_file_path: Optional[str] = None,
        transcript_file_path: Optional[str] = None,
    ) -> Dict:
        """
        Run the complete workflow from video file to final edited video

        Args:
            video_file_path: Path to the input video file (required if start_stage='transcription')
            transcript_file_path: Path to existing transcript file (required if start_stage='analysis')

        Returns:
            Dictionary with complete workflow results
        """
        logger.info("🚀 STARTING COMPLETE VIDEO WORKFLOW")
        logger.info("=" * 60)

        # Validate input parameters based on start stage
        if self.start_stage == "transcription" and not video_file_path:
            raise ValueError(
                "video_file_path is required when start_stage='transcription'"
            )
        elif self.start_stage == "analysis" and not transcript_file_path:
            # Use default transcript file path if not provided
            default_transcript_file = (
                self.transcript_dir / TRANSCRIPTION_JSON_FILENAME
            )
            if default_transcript_file.exists():
                transcript_file_path = str(default_transcript_file)
                logger.info(
                    f"🔍 Using default transcript file: {transcript_file_path}"
                )
            else:
                raise ValueError(
                    f"transcript_file_path is required when start_stage='analysis'. "
                    f"Default file not found: {default_transcript_file}"
                )

        logger.info(f"🚀 Starting from stage: {self.start_stage.upper()}")
        logger.info(f"📁 Input video: {video_file_path or 'N/A'}")
        logger.info(f"📄 Input transcript: {transcript_file_path or 'N/A'}")

        workflow_stages = []
        if self.start_stage == "transcription":
            workflow_stages.append("Video → Transcript")
        workflow_stages.extend(["Transcript → Prompts", "Prompts → Images"])

        if not self.skip_video_generation:
            workflow_stages.extend(["Images → Videos", "Videos → Final Edit"])

        logger.info(f"🎯 Workflow stages: {' → '.join(workflow_stages)}")

        # Check what will be skipped
        if self.skip_video_generation:
            logger.info(
                f"⏭️ Skipped stages: Video Generation, Final Video Editing"
            )

        logger.info("=" * 60)

        start_time = time.time()

        try:
            # Step 1: Transcribe video file (if starting from transcription)
            if self.start_stage == "transcription":
                transcript_file = self.transcribe_video_file(video_file_path)
            else:
                transcript_file = transcript_file_path
                logger.info(
                    "⏭️ SKIPPING VIDEO TRANSCRIPTION (starting from analysis)"
                )
                logger.info(f"📄 Using existing transcript: {transcript_file}")

            # Step 2: Process transcript to prompts
            prompts_data = self.process_transcript_to_prompts(transcript_file)

            # Step 3: Generate images from prompts
            image_results = self.generate_images_from_prompts(prompts_data)

            # Step 4: Create videos from images (if not skipped)
            if not self.skip_video_generation and ENABLE_VIDEO_GENERATION:
                video_results = self.create_videos_from_images(image_results)
            else:
                if self.skip_video_generation:
                    logger.info(
                        "\n⏭️ SKIPPING VIDEO GENERATION (skip_video_generation = True)"
                    )
                else:
                    logger.info(
                        "\n⏭️ SKIPPING VIDEO GENERATION (ENABLE_VIDEO_GENERATION = False)"
                    )
                video_results = []

            # Step 5: Insert b-roll into main video (if not skipped and videos were generated)
            if (
                not self.skip_video_generation
                and ENABLE_VIDEO_GENERATION
                and video_results
                and video_file_path  # Check if original video file exists
            ):
                final_video_path = self.insert_broll_into_main_video(
                    video_file_path, video_results
                )
            else:
                skip_reasons = []
                if self.skip_video_generation:
                    skip_reasons.append("skip_video_generation = True")
                if not ENABLE_VIDEO_GENERATION:
                    skip_reasons.append("ENABLE_VIDEO_GENERATION = False")
                if not video_results:
                    skip_reasons.append("no b-roll videos generated")
                if not video_file_path:
                    skip_reasons.append("no original video file provided")

                logger.info(
                    f"\n⏭️ SKIPPING FINAL VIDEO EDITING ({', '.join(skip_reasons)})"
                )
                final_video_path = None

            # Compile final results
            workflow_results = {
                "workflow_info": {
                    "input_video": video_file_path,
                    "input_transcript": (
                        transcript_file_path
                        if self.start_stage == "analysis"
                        else None
                    ),
                    "transcript_file": transcript_file,
                    "final_video": final_video_path,
                    "total_duration": time.time() - start_time,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "api_used": config.is_api_enabled,
                    "max_segments": self.max_segments,
                    "start_stage": self.start_stage,
                    "skip_video_generation": self.skip_video_generation,
                },
                "prompts_data": prompts_data,
                "image_results": image_results,
                "video_results": video_results,
                "final_video_path": final_video_path,
                "summary": {
                    "total_segments": len(
                        prompts_data.get("broll_segments", [])
                    ),
                    "images_generated": len(image_results),
                    "videos_created": len(video_results),
                    "final_video_created": final_video_path is not None,
                    "success_rate": {
                        "images": f"{len(image_results)}/{len(prompts_data.get('broll_segments', []))}",
                        "videos": f"{len(video_results)}/{len(image_results)}",
                    },
                },
            }

            # Save complete workflow report
            report_file = self.data_dir / WORKFLOW_REPORT_FILENAME
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

    def run_from_video(self, video_file_path: str) -> Dict:
        """
        Convenience method to run complete workflow starting from video transcription.

        Args:
            video_file_path: Path to the input video file

        Returns:
            Dictionary with complete workflow results
        """
        return self.run_complete_workflow(video_file_path=video_file_path)

    def run_from_transcript(
        self,
        transcript_file_path: Optional[str] = None,
        video_file_path: Optional[str] = None,
    ) -> Dict:
        """
        Convenience method to run workflow starting from transcript analysis.

        Args:
            transcript_file_path: Path to existing transcript file.
                                If None, uses default path: {AUDIO_TRANSCRIPT_DIR_NAME}/{TRANSCRIPTION_JSON_FILENAME}
            video_file_path: Path to original video file for final editing with b-roll insertion.

        Returns:
            Dictionary with workflow results (excluding transcription)
        """
        return self.run_complete_workflow(
            transcript_file_path=transcript_file_path,
            video_file_path=video_file_path,
        )

    def run_images_only(
        self,
        video_file_path: Optional[str] = None,
        transcript_file_path: Optional[str] = None,
    ) -> Dict:
        """
        Convenience method to run workflow up to image generation only.

        Args:
            video_file_path: Path to video file (if starting from transcription)
            transcript_file_path: Path to transcript file (if starting from analysis)

        Returns:
            Dictionary with workflow results (images only)
        """
        # Temporarily set skip flags
        original_skip_video = self.skip_video_generation

        self.skip_video_generation = True

        try:
            return self.run_complete_workflow(
                video_file_path=video_file_path,
                transcript_file_path=transcript_file_path,
            )
        finally:
            # Restore original settings
            self.skip_video_generation = original_skip_video

    def print_workflow_summary(self, results: Dict):
        """Print a comprehensive summary of the workflow results."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 COMPLETE VIDEO WORKFLOW - FINAL SUMMARY")
        logger.info("=" * 60)

        summary = results["summary"]
        workflow_info = results["workflow_info"]

        logger.info(
            f"⏱️ Total processing time: {workflow_info['total_duration']:.1f} seconds"
        )
        logger.info(
            f"🚀 Started from stage: {workflow_info['start_stage'].upper()}"
        )
        logger.info(
            f"🎯 Max segments configured: {workflow_info.get('max_segments', 'N/A')}"
        )
        logger.info(
            f"📝 Transcript segments analyzed: {summary['total_segments']}"
        )
        logger.info(f"🎨 Images generated: {summary['images_generated']}")

        if (
            not workflow_info.get("skip_video_generation", False)
            and ENABLE_VIDEO_GENERATION
        ):
            logger.info(f"🎬 Videos created: {summary['videos_created']}")
            logger.info(
                f"🎞️ Final video created: {'✅ YES' if summary['final_video_created'] else '❌ NO'}"
            )
        else:
            logger.info(f"🎬 Videos created: [SKIPPED]")
            logger.info(f"🎞️ Final video created: [SKIPPED]")

        logger.info(f"📈 Success rates:")
        logger.info(f"   • Images: {summary['success_rate']['images']}")
        if (
            not workflow_info.get("skip_video_generation", False)
            and ENABLE_VIDEO_GENERATION
        ):
            logger.info(f"   • Videos: {summary['success_rate']['videos']}")
        else:
            logger.info(f"   • Videos: [SKIPPED]")

        logger.info(f"\n📁 Output files:")
        if workflow_info.get("input_transcript"):
            logger.info(
                f"   • Input transcript: {workflow_info['input_transcript']}"
            )
        logger.info(
            f"   • Transcript: {workflow_info.get('transcript_file', 'N/A')}"
        )
        logger.info(f"   • Prompts: {self.prompts_dir}")
        logger.info(f"   • Images: {self.images_dir}")

        if (
            not workflow_info.get("skip_video_generation", False)
            and ENABLE_VIDEO_GENERATION
        ):
            logger.info(f"   • B-roll videos: {self.videos_dir}")
            if workflow_info.get("final_video"):
                logger.info(
                    f"   • Final video: {workflow_info['final_video']}"
                )
        else:
            logger.info(f"   • Videos: [SKIPPED]")

        logger.info(f"   • Report: {self.data_dir}/{WORKFLOW_REPORT_FILENAME}")

        # Final status messages
        video_gen_enabled = (
            not workflow_info.get("skip_video_generation", False)
            and ENABLE_VIDEO_GENERATION
        )

        if video_gen_enabled and summary.get("final_video_created"):
            logger.info(f"\n🎉 Complete workflow finished successfully!")
            logger.info(f"🚀 Final video with b-roll is ready to use!")
        elif video_gen_enabled and summary["videos_created"] > 0:
            logger.info(f"\n⚠️ Workflow completed with partial success!")
            logger.info(f"🎬 B-roll videos created but final editing failed!")
        elif not video_gen_enabled:
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
    """
    Main function to run the unified workflow.

    Examples of usage with new parameters:

    # Standard workflow from video to final result
    workflow = UnifiedWorkflow(max_segments=3)
    results = workflow.run_from_video("path/to/video.mp4")

    # Start from existing transcript (automatic default path)
    workflow = UnifiedWorkflow(start_stage="analysis")
    results = workflow.run_from_transcript()  # Uses default transcript file

    # Start from custom transcript file
    workflow = UnifiedWorkflow(start_stage="analysis")
    results = workflow.run_from_transcript("path/to/custom/transcript.json")

    # Generate only images (skip video generation)
    workflow = UnifiedWorkflow(skip_video_generation=True)
    results = workflow.run_images_only(video_file_path="path/to/video.mp4")

    # Skip video generation and editing
    workflow = UnifiedWorkflow(skip_video_generation=True)
    results = workflow.run_from_video("path/to/video.mp4")
    """
    # Default video file path using constants
    DEFAULT_VIDEO_FILE = (
        Path(__file__).parent / DEFAULT_VIDEO_PATH[2:]
    )  # Remove "./" prefix

    # Check if video file exists
    if not DEFAULT_VIDEO_FILE.exists():
        logger.info(f"❌ Video file not found: {DEFAULT_VIDEO_FILE}")
        logger.info(
            f"Please ensure you have a valid video file in the {INPUT_VIDEO_DIR_NAME} directory."
        )

        # Create the directory if it doesn't exist
        DEFAULT_VIDEO_FILE.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directory: {DEFAULT_VIDEO_FILE.parent}")
        logger.info(
            f"📝 Please place your video file at: {DEFAULT_VIDEO_FILE}"
        )
        return False

    try:
        logger.info(
            f"🎯 Configuring workflow with max {MAX_SEGMENTS} segments"
        )

        # Initialize and run workflow
        workflow = UnifiedWorkflow(max_segments=MAX_SEGMENTS)
        results = workflow.run_complete_workflow(
            video_file_path=str(DEFAULT_VIDEO_FILE)
        )

        return (
            results["summary"].get("final_video_created", False)
            or results["summary"]["videos_created"] > 0
        )

    except Exception as e:
        logger.info(f"❌ Workflow execution failed: {e}")
        return False


if __name__ == "__main__":
    success = main()

    if success:
        logger.info(f"\n✅ Complete video workflow finished successfully!")
        logger.info(f"🎬 Your video with b-roll content is ready!")
    else:
        logger.info(f"\n❌ Workflow failed. Please check the errors above.")
