#!/usr/bin/env python3
"""
Main entry point for the complete video workflow with b-roll generation.

This workflow includes:
1. Video transcription with word-level timestamps
2. Transcript analysis and b-roll prompt generation
3. Image generation from prompts
4. Video generation from images
5. Final video editing with b-roll insertion
"""

import os
from pathlib import Path

from b_roll.config import config
from b_roll.constants import MAX_SEGMENTS
from b_roll.logger_config import logger
from b_roll.workflow import UnifiedWorkflow


def main():
    """Main function to run the complete video workflow."""
    logger.info("🎬 COMPLETE VIDEO WORKFLOW WITH B-ROLL GENERATION")
    logger.info("=" * 60)

    # Display API configuration
    logger.info("🔧 API Configuration:")
    logger.info(f"  API Request Enabled: {config.is_api_enabled}")
    logger.info(f"  Mock Mode: {config.is_mock_enabled}")
    logger.info(
        f"  Environment Variable API_REQUEST: {config.api_request_enabled}"
    )
    logger.info("")

    # Workflow description
    logger.info("📋 Workflow Steps:")
    logger.info("  1. 🎵 Video Transcription (word-level timestamps)")
    logger.info("  2. 🎯 Transcript Analysis & B-roll Prompt Generation")
    logger.info("  3. 🎨 Image Generation from Prompts")
    logger.info("  4. 🎬 Video Generation from Images")
    logger.info("  5. 🎞️ Final Video Editing with B-roll Insertion")
    logger.info("")

    try:
        logger.info(
            f"🎯 Configured for maximum {MAX_SEGMENTS} b-roll segments"
        )

        # -------WORKFLOW------------------------------------------------------
        # Initialize unified workflow

        # -------WORKFLOW CONFIGURATION 1 (DEFAULT)----------------------------
        workflow = UnifiedWorkflow(
            max_segments=MAX_SEGMENTS,
            start_stage="analysis",
        )

        # -------WORKFLOW CONFIGURATION 2--------------------------------------
        # For images only (skip video generation):
        # workflow = UnifiedWorkflow(
        #     max_segments=MAX_SEGMENTS,
        #     # start_stage="transcription",  # Start from transcription (default)
        #     start_stage="analysis",  # Start from transcription analysis
        #     skip_video_generation=True,  # Skip video generation if needed
        # )
        # Alternative configurations:
        # For images only (skip video generation):
        # workflow = UnifiedWorkflow(max_segments=MAX_SEGMENTS, skip_video_generation=True)
        # results = workflow.run_images_only(video_file_path=video_file_path)

        # -------VIDEO FILE PATH-----------------------------------------------
        # Set video file path based on mock mode
        base_data_dir = "data_mock" if config.is_mock_enabled else "data"
        video_file_path = (
            f"b_roll/{base_data_dir}/video_generation/input_video/video.mp4"
        )

        # Check if video file exists
        if not os.path.exists(video_file_path):
            logger.info(f"❌ Video file not found: {video_file_path}")
            logger.info(
                "📝 Please place your video file in the input_video directory"
            )

            # Create directory if it doesn't exist
            video_dir = Path(video_file_path).parent
            video_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Created directory: {video_dir}")

            return False

        logger.info(f"📹 Input video file: {video_file_path}")
        logger.info("")

        # -------RUN THE WORKFLOW----------------------------------------------
        # Run the complete workflow
        if "analysis" in str(workflow.start_stage):
            # Start from transcript analysis (uses default transcript file path)
            results = workflow.run_from_transcript(
                video_file_path=video_file_path
            )
        else:
            # Start from video transcription
            results = workflow.run_from_video(video_file_path)

        # Check results
        summary = results.get("summary", {})
        final_video_created = summary.get("final_video_created", False)
        videos_created = summary.get("videos_created", 0)

        if final_video_created:
            logger.info("\n🎉 SUCCESS: Complete workflow finished!")
            logger.info("🎬 Your final video with b-roll content is ready!")
            final_video_path = results.get("workflow_info", {}).get(
                "final_video"
            )
            if final_video_path:
                logger.info(f"📁 Final video location: {final_video_path}")
        elif videos_created > 0:
            logger.info(
                "\n⚠️ PARTIAL SUCCESS: B-roll videos created but final editing failed"
            )
            logger.info("🎬 You can use the generated b-roll videos manually")
        else:
            logger.info("\n❌ FAILED: No videos were generated")
            logger.info("🔍 Check the logs above for error details")
            return False

        return True

    except Exception as e:
        logger.error(f"❌ Workflow execution failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()

    logger.info("\n" + "=" * 60)
    if success:
        logger.info("🚀 Video workflow completed successfully!")
        logger.info("✨ Your content is ready for use!")
    else:
        logger.info("❌ Workflow failed - please check the errors above")
    logger.info("=" * 60)
