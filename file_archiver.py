#!/usr/bin/env python3
"""
File Archiver

This script creates a timestamped folder in _temp/generated_media/result_examples
and moves/copies specified files from the video generation workflow.
"""

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import configuration
sys.path.append(str(Path(__file__).parent / "b_roll"))
try:
    from b_roll import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


class FileArchiver:
    """Archives workflow files to timestamped result folders."""

    def __init__(self, base_path: str = None):
        """
        Initialize the archiver.

        Args:
            base_path: Base path to the project (defaults to current script directory)
        """
        if base_path is None:
            self.base_path = Path(__file__).parent
        else:
            self.base_path = Path(base_path)

        # Define source paths
        self.data_dir = self.base_path / "b_roll" / "data" / "video_generation"

        # Files to move
        self.files_to_move = [
            self.data_dir
            / "broll_prompts"
            / "workflow_generated_prompts.json",
            self.data_dir / "workflow_complete_report.json",
        ]

        # Directories to move (all files from these directories)
        self.dirs_to_move = [
            self.data_dir / "images_input",
            self.data_dir / "videos_output",
            self.data_dir / "video_output",
        ]

        # Files to copy
        self.files_to_copy = [
            self.data_dir
            / "audio_transcript"
            / "transcription_verbose_to_json.json",
        ]

    def get_timestamp(self) -> str:
        """Returns current timestamp in YYYYMMDD_HHMMSS format."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def create_result_folder(self) -> Path:
        """
        Creates a timestamped result folder in _temp/generated_media/result_examples.

        Returns:
            Path to the created result folder
        """
        timestamp = self.get_timestamp()
        result_path = (
            self.base_path
            / "_temp"
            / "generated_media"
            / "result_examples"
            / timestamp
        )

        # Create the full path if it doesn't exist
        result_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created result folder: {result_path}")

        return result_path

    def move_file(self, source_path: Path, dest_folder: Path) -> bool:
        """
        Moves a file to the destination folder.

        Args:
            source_path: Path to the source file
            dest_folder: Destination folder path

        Returns:
            True if successful, False otherwise
        """
        if not source_path.exists():
            logger.warning(f"⚠️  Source file not found: {source_path}")
            return False

        dest_path = dest_folder / source_path.name

        try:
            shutil.move(str(source_path), str(dest_path))
            logger.info(f"✅ Moved: {source_path.name}")
            return True
        except Exception as e:
            logger.error(f"❌ Error moving {source_path.name}: {e}")
            return False

    def copy_file(self, source_path: Path, dest_folder: Path) -> bool:
        """
        Copies a file to the destination folder.

        Args:
            source_path: Path to the source file
            dest_folder: Destination folder path

        Returns:
            True if successful, False otherwise
        """
        if not source_path.exists():
            logger.warning(f"⚠️  Source file not found: {source_path}")
            return False

        dest_path = dest_folder / source_path.name

        try:
            shutil.copy2(str(source_path), str(dest_path))
            logger.info(f"✅ Copied: {source_path.name}")
            return True
        except Exception as e:
            logger.error(f"❌ Error copying {source_path.name}: {e}")
            return False

    def move_directory_contents(
        self, source_dir: Path, dest_folder: Path
    ) -> int:
        """
        Moves all files from source directory to destination folder.

        Args:
            source_dir: Path to the source directory
            dest_folder: Destination folder path

        Returns:
            Number of successfully moved files
        """
        if not source_dir.exists():
            logger.warning(f"⚠️  Source directory not found: {source_dir}")
            return 0

        moved_count = 0

        try:
            for file_path in source_dir.iterdir():
                if file_path.is_file():
                    if self.move_file(file_path, dest_folder):
                        moved_count += 1
        except Exception as e:
            logger.error(f"❌ Error processing directory {source_dir}: {e}")

        return moved_count

    def archive_workflow_files(self) -> dict:
        """
        Archives all workflow files to a timestamped result folder.

        Returns:
            Dictionary with archiving results
        """
        logger.info("🚀 Starting workflow files archiving...")
        logger.info("=" * 50)

        # Create result folder
        result_folder = self.create_result_folder()

        results = {
            "result_folder": str(result_folder),
            "moved_files": 0,
            "copied_files": 0,
            "errors": [],
        }

        # Move individual files
        logger.info("\n📄 Moving individual files:")
        for file_path in self.files_to_move:
            if self.move_file(file_path, result_folder):
                results["moved_files"] += 1
            else:
                results["errors"].append(f"Failed to move: {file_path}")

        # Move all files from directories
        logger.info("\n📁 Moving files from directories:")
        for dir_path in self.dirs_to_move:
            logger.info(f"Processing directory: {dir_path.name}")
            moved_count = self.move_directory_contents(dir_path, result_folder)
            results["moved_files"] += moved_count
            if moved_count == 0 and dir_path.exists():
                results["errors"].append(f"No files moved from: {dir_path}")

        # Copy files
        logger.info("\n📋 Copying files:")
        for file_path in self.files_to_copy:
            if self.copy_file(file_path, result_folder):
                results["copied_files"] += 1
            else:
                results["errors"].append(f"Failed to copy: {file_path}")

        return results

    def print_results(self, results: dict):
        """
        Prints archiving results.

        Args:
            results: Results dictionary from archive_workflow_files
        """
        logger.info("\n" + "=" * 50)
        logger.info("📊 ARCHIVING RESULTS")
        logger.info("=" * 50)

        logger.info(f"📁 Result folder: {results['result_folder']}")
        logger.info(f"📦 Files moved: {results['moved_files']}")
        logger.info(f"📋 Files copied: {results['copied_files']}")

        if results["errors"]:
            logger.info(f"⚠️  Errors: {len(results['errors'])}")
            for error in results["errors"]:
                logger.info(f"   - {error}")

        total_files = results["moved_files"] + results["copied_files"]
        if total_files > 0:
            logger.info(f"\n✅ Archiving completed successfully!")
            logger.info(f"📦 Total files processed: {total_files}")
        else:
            logger.info(f"\n📭 No files were processed")


def main():
    """Main function to run archiving."""

    # Create archiver and run the process
    archiver = FileArchiver()

    # Archive workflow files
    results = archiver.archive_workflow_files()

    # Print results
    archiver.print_results(results)


if __name__ == "__main__":
    main()
