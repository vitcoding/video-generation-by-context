#!/usr/bin/env python3
"""
File Archiver

This script archives files from specified directories by copying them
to timestamped archive folders and optionally deleting the originals.
"""

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Import configuration
sys.path.append(str(Path(__file__).parent))
from constants import DEFAULT_IMAGES_OUTPUT_DIR, DEFAULT_VIDEOS_OUTPUT_DIR
from logger_config import logger


class FileArchiver:
    """Archives files with timestamp added to their names."""

    def __init__(self, source_directories: List[str]):
        """
        Initialize the archiver.

        Args:
            source_directories: List of paths to folders for archiving
        """
        self.source_directories = source_directories

    def get_timestamp(self) -> str:
        """Returns current timestamp in YYYYMMDD_HHMMSS format."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def create_archive_folder(self, directory: str) -> str:
        """
        Creates an archive folder in the specified directory.

        Args:
            directory: Path to the directory

        Returns:
            Path to the created archive folder
        """
        archive_path = os.path.join(directory, "archive")

        if not os.path.exists(archive_path):
            os.makedirs(archive_path)
            logger.info(f"📁 Created folder: {archive_path}")
        else:
            logger.info(f"📁 Folder already exists: {archive_path}")

        return archive_path

    def get_file_extension(self, filename: str) -> tuple:
        """
        Splits filename into name and extension.

        Args:
            filename: File name

        Returns:
            Tuple (name, extension)
        """
        name, ext = os.path.splitext(filename)
        return name, ext

    def archive_files_in_directory(
        self, directory: str, delete_originals: bool = False
    ) -> tuple:
        """
        Archives files in the specified directory.

        Args:
            directory: Path to the directory
            delete_originals: Flag to enable/disable deletion of original files after archiving

        Returns:
            Tuple (number of archived files, number of deleted files)
        """
        if not os.path.exists(directory):
            logger.info(f"❌ Directory does not exist: {directory}")
            return 0, 0

        archive_path = self.create_archive_folder(directory)
        timestamp = self.get_timestamp()
        archived_count = 0
        deleted_count = 0

        logger.info(f"\n🔍 Processing directory: {directory}")

        try:
            # Get list of files in the directory
            files = [
                f
                for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f))
            ]

            if not files:
                logger.info("   📭 Directory is empty")
                return 0, 0

            logger.info(f"   📄 Found files: {len(files)}")

            for filename in files:
                # Skip files in archive folder
                if filename == "archive":
                    continue

                file_path = os.path.join(directory, filename)
                name, ext = self.get_file_extension(filename)

                # Create new name with timestamp
                new_filename = f"{timestamp}_{name}{ext}"
                new_file_path = os.path.join(archive_path, new_filename)

                # Check if file with such name already exists
                counter = 1
                while os.path.exists(new_file_path):
                    new_filename = f"{timestamp}_{name}_{counter}{ext}"
                    new_file_path = os.path.join(archive_path, new_filename)
                    counter += 1

                # Copy the file to archive
                try:
                    shutil.copy2(file_path, new_file_path)
                    logger.info(f"   ✅ Archived: {filename} → {new_filename}")
                    archived_count += 1

                    # Delete original file if flag is enabled
                    if delete_originals:
                        try:
                            os.remove(file_path)
                            logger.info(f"   🗑️  Deleted original: {filename}")
                            deleted_count += 1
                        except Exception as e:
                            logger.info(
                                f"   ❌ Error deleting original {filename}: {e}"
                            )

                except Exception as e:
                    logger.info(f"   ❌ Error archiving {filename}: {e}")

        except Exception as e:
            logger.info(f"   ❌ Error processing directory {directory}: {e}")

        return archived_count, deleted_count

    def archive_all_directories(self, delete_originals: bool = False) -> dict:
        """
        Archives files in all specified directories.

        Args:
            delete_originals: Flag to enable/disable deletion of original files after archiving

        Returns:
            Dictionary with archiving results by directory
        """
        results = {}
        total_archived = 0
        total_deleted = 0

        logger.info("🚀 Starting file archiving...")
        if delete_originals:
            logger.info("🗑️  Original files will be deleted after archiving")
        logger.info("=" * 50)

        for directory in self.source_directories:
            archived_count, deleted_count = self.archive_files_in_directory(
                directory, delete_originals
            )
            results[directory] = {
                "archived": archived_count,
                "deleted": deleted_count,
            }
            total_archived += archived_count
            total_deleted += deleted_count

        logger.info("\n" + "=" * 50)
        logger.info("📊 ARCHIVING RESULTS")
        logger.info("=" * 50)

        for directory, counts in results.items():
            logger.info(f"📁 {directory}: {counts['archived']} files archived")
            if delete_originals and counts["deleted"] > 0:
                logger.info(
                    f"   🗑️  {counts['deleted']} original files deleted"
                )

        logger.info(f"\n🎯 Total archived: {total_archived} files")
        if delete_originals:
            logger.info(f"🗑️  Total deleted: {total_deleted} original files")

        return results


def main():
    """Main function to run archiving."""

    # List of directories to archive
    # Change this list to your needed paths
    directories_to_archive = [
        DEFAULT_VIDEOS_OUTPUT_DIR,
        DEFAULT_IMAGES_OUTPUT_DIR,
        # Add other directories as needed
    ]

    # Check that directories exist
    existing_directories = []
    for directory in directories_to_archive:
        if os.path.exists(directory):
            existing_directories.append(directory)
        else:
            logger.info(f"⚠️  Directory not found: {directory}")

    if not existing_directories:
        logger.info("❌ No available directories for archiving!")
        return

    # Create archiver and run the process
    archiver = FileArchiver(existing_directories)

    # Set delete_originals flag (change to True to delete original files)
    delete_originals = False

    results = archiver.archive_all_directories(
        delete_originals=delete_originals
    )

    # Check if files were archived
    total_archived = sum(result["archived"] for result in results.values())
    total_deleted = sum(result["deleted"] for result in results.values())

    if total_archived > 0:
        logger.info(f"\n✅ Archiving completed successfully!")
        logger.info(f"📦 All files copied to 'archive' folders with timestamp")
        if delete_originals and total_deleted > 0:
            logger.info(f"🗑️  Original files deleted: {total_deleted}")
    else:
        logger.info(f"\n📭 No files found for archiving")


if __name__ == "__main__":
    main()
