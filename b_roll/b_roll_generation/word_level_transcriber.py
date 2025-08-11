#!/usr/bin/env python3
"""
Word-Level Video Transcriber Module
Transcribes video files with word-by-word timestamps using OpenAI Whisper API
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# Adding a path for importing constants and logger
sys.path.append(str(Path(__file__).parent.parent))
import mock_api
from config import config
from constants import (
    AUDIO_TRANSCRIPT_DIR_NAME,
    BASE_RELATIVE_PATH,
    DEFAULT_VIDEO_FILENAME,
    INPUT_VIDEO_DIR_NAME,
    TRANSCRIPTION_JSON_FILENAME,
    VIDEO_GENERATION_DIR_NAME,
    base_data_dir,
)
from logger_config import logger
from mock_api import mock_openai_client

# Import real OpenAI module only if API is enabled
if config.is_api_enabled:
    from openai import OpenAI
else:
    # Use mock OpenAI client
    OpenAI = mock_openai_client


class VideoTranscriber:
    """Video transcription class with API/mock support"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the video transcriber

        Args:
            api_key: OpenAI API key. If None, will try to get from .env file
        """
        # Load environment variables from .env file in project root
        project_root = Path(__file__).parent.parent
        env_file = project_root / ".env"
        load_dotenv(env_file)

        # Get API key based on configuration
        self.api_key = api_key or config.get_api_key("OPENAI_API_KEY")

        if config.is_api_enabled and not self.api_key:
            logger.info(f"⚠️ Warning: OPENAI_API_KEY not found in {env_file}")
            logger.info("🔧 Using mock mode for transcription")

        # Initialize OpenAI client
        if config.is_api_enabled:
            self.client = OpenAI(api_key=self.api_key)
        else:
            logger.info("🔧 [MOCK] Using mock OpenAI client")
            self.client = OpenAI()

        # OpenAI API file size limit (25 MB)
        self.max_file_size_mb = 25

    def _get_file_size_mb(self, file_path: str) -> float:
        """Get file size in megabytes"""
        return os.path.getsize(file_path) / (1024 * 1024)

    def _extract_audio_for_transcription(self, input_path: str) -> str:
        """
        Extract audio from video file for transcription

        Args:
            input_path: Path to original video file

        Returns:
            str: Path to extracted audio file
        """
        # Create temporary file for extracted audio
        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp3", prefix="audio_")
        os.close(temp_fd)  # Close file descriptor, we only need the path

        try:
            # Check if ffmpeg is available
            subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise Exception(
                "FFmpeg is not installed or not available in PATH. Please install FFmpeg to extract audio from video files."
            )

        try:
            # Use ffmpeg to extract audio with optimized settings for smaller file size
            cmd = [
                "ffmpeg",
                "-i",
                input_path,
                "-vn",  # No video
                "-acodec",
                "libmp3lame",  # MP3 codec
                "-ab",
                "64k",  # Reduced bitrate to 64kbps (sufficient for speech)
                "-ar",
                "16000",  # Reduced sample rate to 16kHz (standard for speech)
                "-ac",
                "1",  # Mono audio (sufficient for transcription)
                "-y",  # Overwrite output file
                temp_path,
            ]

            logger.info(
                f"Extracting audio for transcription: {input_path} -> {temp_path}"
            )
            logger.info(
                f"Original video size: {self._get_file_size_mb(input_path):.1f} MB"
            )

            # Run ffmpeg with suppressed output
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )

            audio_size = self._get_file_size_mb(temp_path)
            logger.info(f"Extracted audio size: {audio_size:.1f} MB")

            # Check if audio file exceeds OpenAI API limit
            if audio_size > self.max_file_size_mb:
                logger.warning(
                    f"⚠️ Audio file size ({audio_size:.1f} MB) exceeds OpenAI API limit ({self.max_file_size_mb} MB)"
                )
                # Try to reduce quality further
                temp_path_compressed = self._compress_audio_further(temp_path)
                if temp_path_compressed:
                    # Clean up original temp file
                    os.unlink(temp_path)
                    temp_path = temp_path_compressed
                    audio_size = self._get_file_size_mb(temp_path)
                    logger.info(f"Compressed audio size: {audio_size:.1f} MB")

                    if audio_size > self.max_file_size_mb:
                        raise Exception(
                            f"Audio file is still too large ({audio_size:.1f} MB) after compression. "
                            f"Maximum allowed size is {self.max_file_size_mb} MB. "
                            f"Consider using a shorter video or splitting it into segments."
                        )

            return temp_path

        except subprocess.CalledProcessError as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            logger.error(f"FFmpeg audio extraction failed: {e.stderr}")
            raise Exception(
                f"Audio extraction failed. FFmpeg error: {e.stderr}"
            )
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e

    def _compress_audio_further(self, audio_path: str) -> Optional[str]:
        """
        Further compress audio file if it's too large for OpenAI API

        Args:
            audio_path: Path to audio file to compress

        Returns:
            Optional[str]: Path to compressed audio file, or None if compression failed
        """
        try:
            # Create new temporary file for compressed audio
            temp_fd, compressed_path = tempfile.mkstemp(
                suffix=".mp3", prefix="audio_compressed_"
            )
            os.close(temp_fd)

            # Use even more aggressive compression settings
            cmd = [
                "ffmpeg",
                "-i",
                audio_path,
                "-acodec",
                "libmp3lame",
                "-ab",
                "32k",  # Very low bitrate (32kbps)
                "-ar",
                "8000",  # Very low sample rate (8kHz - minimum for speech)
                "-ac",
                "1",  # Mono
                "-y",  # Overwrite output file
                compressed_path,
            ]

            logger.info(
                "Applying additional compression to reduce file size..."
            )
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )

            compressed_size = self._get_file_size_mb(compressed_path)
            logger.info(
                f"Further compressed audio size: {compressed_size:.1f} MB"
            )

            return compressed_path

        except Exception as e:
            logger.error(f"Additional compression failed: {e}")
            if os.path.exists(compressed_path):
                os.unlink(compressed_path)
            return None

    def transcribe_video_to_json(
        self,
        video_file_path: str,
        output_json_path: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe video file with word-level timestamps and save to JSON.

        Args:
            video_file_path: Path to input video file
            output_json_path: Path to save JSON transcription (optional, uses default if not provided)
            language: Language code for transcription (e.g., 'ru', 'en', 'es'). If None, auto-detects language.

        Returns:
            str: Path to the saved JSON transcription file

        Raises:
            ValueError: If required parameters are missing
            FileNotFoundError: If input video file doesn't exist
            Exception: If transcription fails
        """
        # Step 1: Set paths using constants
        if output_json_path is None:
            output_json_path = f"b_roll/{base_data_dir}/{VIDEO_GENERATION_DIR_NAME}/{AUDIO_TRANSCRIPT_DIR_NAME}/{TRANSCRIPTION_JSON_FILENAME}"

        # Step 2: Validate input file
        if not os.path.exists(video_file_path):
            raise FileNotFoundError(f"Video file not found: {video_file_path}")

        # Step 3: Create output directory
        output_path = Path(output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory created/verified: {output_path.parent}")

        # Step 4: Extract audio from video for transcription
        logger.info("Extracting audio from video for transcription...")

        # Always extract audio file for better performance and reliability
        audio_file = self._extract_audio_for_transcription(video_file_path)

        try:
            audio_size_mb = self._get_file_size_mb(audio_file)
            logger.info(f"Extracted audio size: {audio_size_mb:.1f} MB")

            # Step 5: Transcribe audio file
            logger.info(f"Starting transcription of: {audio_file}")
            logger.info("Processing with word-level timestamps...")

            with open(audio_file, "rb") as audio_file_handle:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file_handle,
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                    language=language,
                )

            logger.info("Transcription completed successfully")

        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            # Clean up audio file if it exists
            if audio_file and os.path.exists(audio_file):
                os.unlink(audio_file)
                logger.info(f"Cleaned up audio file: {audio_file}")
            raise
        finally:
            # Clean up audio file if it exists
            if audio_file and os.path.exists(audio_file):
                os.unlink(audio_file)
                logger.info(f"Cleaned up audio file: {audio_file}")

        # Step 6: Process transcript data
        try:
            transcript_data = self._build_transcript_data(transcript)
            logger.info(
                f"Processed transcript with {len(transcript_data['words'])} words"
            )

        except Exception as e:
            logger.error(f"Error processing transcript data: {e}")
            raise

        # Step 7: Save transcript to JSON file
        try:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(transcript_data, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ Transcription completed successfully!")
            logger.info(f"📄 Full text: {transcript_data['text'][:100]}...")
            logger.info(f"🗣️ Language: {transcript_data['language']}")
            logger.info(f"⏱️ Duration: {transcript_data['duration']} seconds")
            logger.info(f"📝 Word count: {len(transcript_data['words'])}")
            logger.info(f"💾 Saved to: {output_json_path}")

            return str(output_json_path)

        except Exception as e:
            logger.error(f"Error saving transcript to JSON: {e}")
            raise

    def _build_transcript_data(self, transcript) -> Dict:
        """
        Build structured transcript data with word-level timestamps.

        Args:
            transcript: OpenAI transcript response object

        Returns:
            Dict: Structured transcript data
        """
        transcript_data = {
            "text": transcript.text,
            "language": transcript.language,
            "duration": transcript.duration,
            "words": [],
        }

        if hasattr(transcript, "words") and transcript.words:
            # Add punctuation with estimated timestamps
            words_with_punctuation = self._add_punctuation_timestamps(
                transcript.text, transcript.words
            )
            transcript_data["words"] = words_with_punctuation

        return transcript_data

    def _add_punctuation_timestamps(
        self, full_text: str, words: List
    ) -> List[Dict]:
        """
        Add punctuation marks with estimated timestamps.

        Args:
            full_text: Complete transcribed text
            words: List of word objects with timestamps

        Returns:
            List[Dict]: Words and punctuation with timestamps
        """
        words_with_punctuation = []

        # Split text into individual characters to preserve spaces and punctuation
        text_chars = list(full_text)
        word_index = 0
        char_index = 0

        while char_index < len(text_chars) and word_index < len(words):
            char = text_chars[char_index]

            # Skip whitespace
            if char.isspace():
                char_index += 1
                continue

            # Check if we're at the start of the current word
            current_word = words[word_index].word
            text_slice = "".join(
                text_chars[char_index : char_index + len(current_word)]
            )

            if text_slice.lower() == current_word.lower():
                # Add the word
                words_with_punctuation.append(
                    {
                        "word": current_word,
                        "start": words[word_index].start,
                        "end": words[word_index].end,
                        "type": "word",
                    }
                )

                # Move past the word
                char_index += len(current_word)
                word_index += 1

                # Check for punctuation immediately following the word
                while (
                    char_index < len(text_chars)
                    and not text_chars[char_index].isspace()
                    and not text_chars[char_index].isalnum()
                ):
                    punct_char = text_chars[char_index]

                    # Estimate punctuation timestamp
                    if word_index > 0:
                        prev_word_end = words[word_index - 1].end
                        punct_start = prev_word_end
                        punct_end = punct_start + 0.05
                    else:
                        punct_start = 0.0
                        punct_end = 0.05

                    words_with_punctuation.append(
                        {
                            "word": punct_char,
                            "start": punct_start,
                            "end": punct_end,
                            "type": "punctuation",
                        }
                    )

                    char_index += 1
            else:
                # If we can't match, move to next character
                char_index += 1

        return words_with_punctuation


def transcribe_video_to_json(
    video_file_path: str,
    output_json_path: Optional[str] = None,
    api_key: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """
    Transcribe video file with word-level timestamps and save to JSON.

    Args:
        video_file_path: Path to input video file
        output_json_path: Path to save JSON transcription (optional, uses default if not provided)
        api_key: OpenAI API key (optional, will use env if not provided)
        language: Language code for transcription (e.g., 'ru', 'en', 'es'). If None, auto-detects language.

    Returns:
        str: Path to the saved JSON transcription file
    """
    transcriber = VideoTranscriber(api_key=api_key)
    return transcriber.transcribe_video_to_json(
        video_file_path, output_json_path, language
    )


if __name__ == "__main__":
    # Example usage
    # VIDEO_FILE_PATH = f"b_roll/{base_data_dir}/{VIDEO_GENERATION_DIR_NAME}/{INPUT_VIDEO_DIR_NAME}/{DEFAULT_VIDEO_FILENAME}"
    VIDEO_FILE_PATH = f"b_roll/{base_data_dir}/{VIDEO_GENERATION_DIR_NAME}/{INPUT_VIDEO_DIR_NAME}/extracted_audio.mp3"

    # Language configuration
    # Supported languages: 'ru' (Russian), 'en' (English), 'es' (Spanish), 'fr' (French), 'de' (German), etc.
    # Set to None for auto-detection
    TRANSCRIPTION_LANGUAGE = (
        # None,  # Change this to specify language, e.g., 'ru' for Russian
        "ru",
    )

    # Check for command line arguments
    if len(sys.argv) > 1:
        # First argument can be video file path
        VIDEO_FILE_PATH = sys.argv[1]

    if len(sys.argv) > 2:
        # Second argument can be language code
        TRANSCRIPTION_LANGUAGE = sys.argv[2]
        if TRANSCRIPTION_LANGUAGE.lower() == "auto":
            TRANSCRIPTION_LANGUAGE = None

    try:
        # Log configuration
        logger.info(f"Video file: {VIDEO_FILE_PATH}")
        logger.info(f"Language: {TRANSCRIPTION_LANGUAGE or 'auto-detect'}")

        # Transcribe video with word-level timestamps
        result_path = transcribe_video_to_json(
            video_file_path=VIDEO_FILE_PATH, language=TRANSCRIPTION_LANGUAGE
        )
        logger.info(f"Transcription saved at: {result_path}")

    except Exception as e:
        logger.error(f"Failed to transcribe video: {e}")
        logger.info(
            "Usage: python word_level_transcriber.py [video_file_path] [language_code]"
        )
        logger.info("Example: python word_level_transcriber.py video.mp4 ru")
        logger.info("Example: python word_level_transcriber.py video.mp4 auto")
