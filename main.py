import os

from b_roll.b_roll_generation.broll_image_generation import (
    main as image_generation_main,
)
from b_roll.b_roll_generation.broll_prompts import main as broll_prompts_main
from b_roll.b_roll_generation.kling_image_to_video import (
    main as kling_image_to_video_main,
)
from b_roll.config import config
from b_roll.constants import (
    DEFAULT_IMAGES_OUTPUT_DIR,
    DEFAULT_VIDEOS_OUTPUT_DIR,
)
from b_roll.logger_config import logger
from b_roll.workflow import UnifiedWorkflow
from file_archiver import FileArchiver


def archive_files(enable_archiving=True, delete_originals=True):
    """
    Archive files from specified directories.

    Args:
        enable_archiving (bool): Flag to enable/disable archiving functionality
        delete_originals (bool): Flag to delete original files after archiving
    """
    if not enable_archiving:
        logger.info("📦 File archiving disabled")
        return

    logger.info("📦 Starting file archiving process...")
    directories_to_archive = [
        DEFAULT_VIDEOS_OUTPUT_DIR,
        DEFAULT_IMAGES_OUTPUT_DIR,
    ]

    # Check that directories exist
    existing_directories = []
    for directory in directories_to_archive:
        if os.path.exists(directory):
            existing_directories.append(directory)
        else:
            logger.info(f"⚠️  Directory not found: {directory}")

    if existing_directories:
        # Create archiver and run the process
        archiver = FileArchiver(existing_directories)

        results = archiver.archive_all_directories(
            delete_originals=delete_originals
        )

        # Check if files were archived
        total_archived = sum(result["archived"] for result in results.values())
        total_deleted = sum(result["deleted"] for result in results.values())

        if total_archived > 0:
            logger.info(f"\n✅ Archiving completed successfully!")
            logger.info(
                f"📦 All files copied to 'archive' folders with timestamp"
            )
            if delete_originals and total_deleted > 0:
                logger.info(f"🗑️  Original files deleted: {total_deleted}")
        else:
            logger.info(f"\n📭 No files found for archiving")
    else:
        logger.info("❌ No available directories for archiving!")


if __name__ == "__main__":
    logger.info("🔧 API Configuration:")
    logger.info(f"  API Request Enabled: {config.is_api_enabled}")
    logger.info(f"  Mock Mode: {config.is_mock_enabled}")
    logger.info(
        f"  Environment Variable API_REQUEST: {config.api_request_enabled}"
    )
    logger.info("")

    # -------------------------------------------------------------------------

    # broll_prompts_main()

    # image_generation_main()

    # kling_image_to_video_main()

    # -------------------------------------------------------------------------

    # Unified workflow - complete pipeline
    logger.info("\n🚀 Starting unified workflow...")

    # Configure number of segments to generate
    MAX_SEGMENTS = 3  # Change this to control number of segments
    # MAX_SEGMENTS = 1  # Change this to control number of segments

    workflow = UnifiedWorkflow(max_segments=MAX_SEGMENTS)

    # Use data_mock directory when mock mode is enabled
    base_data_dir = "data_mock" if config.is_mock_enabled else "data"
    transcript_path = f"b_roll/{base_data_dir}/video_generation/audio_transcript/transcription_verbose_to_json.json"

    workflow.run_complete_workflow(transcript_path)

    # File archiving functionality
    # Set to False to disable archiving
    # ENABLE_ARCHIVING = True
    ENABLE_ARCHIVING = False
    # Set to False to keep original files
    # DELETE_ORIGINALS = True
    DELETE_ORIGINALS = False

    archive_files(
        enable_archiving=ENABLE_ARCHIVING, delete_originals=DELETE_ORIGINALS
    )

    logger.info("\n" + "=" * 50)
    logger.info("🚀 All generation completed!")
    logger.info("=" * 50)
