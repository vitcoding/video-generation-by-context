#!/usr/bin/env python3
"""
Word-Level Video Transcriber Module
Transcribes video files with word-by-word timestamps using OpenAI Whisper API
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# Добавляем путь для импорта констант и логера
sys.path.append(str(Path(__file__).parent.parent))
import mock_api
from config import config
from constants import base_data_dir
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

    def transcribe_video_to_json(
        self,
        video_file_path: str,
        output_json_path: Optional[str] = None,
    ) -> str:
        """
        Transcribe video file with word-level timestamps and save to JSON.

        Args:
            video_file_path: Path to input video file
            output_json_path: Path to save JSON transcription (optional, uses default if not provided)

        Returns:
            str: Path to the saved JSON transcription file

        Raises:
            ValueError: If required parameters are missing
            FileNotFoundError: If input video file doesn't exist
            Exception: If transcription fails
        """
        # Step 1: Set paths using constants
        if output_json_path is None:
            output_json_path = f"b_roll/{base_data_dir}/video_generation/audio_transcript/transcription_verbose_to_json.json"

        # Step 2: Validate input file
        if not os.path.exists(video_file_path):
            raise FileNotFoundError(f"Video file not found: {video_file_path}")

        # Step 3: Create output directory
        output_path = Path(output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory created/verified: {output_path.parent}")

        # Step 4: Transcribe video file
        try:
            logger.info(f"Starting transcription of: {video_file_path}")
            logger.info("Processing with word-level timestamps...")

            with open(video_file_path, "rb") as video_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=video_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                )

            logger.info("Transcription completed successfully")

        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise

        # Step 5: Process transcript data
        try:
            transcript_data = self._build_transcript_data(transcript)
            logger.info(
                f"Processed transcript with {len(transcript_data['words'])} words"
            )

        except Exception as e:
            logger.error(f"Error processing transcript data: {e}")
            raise

        # Step 6: Save transcript to JSON file
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
) -> str:
    """
    Transcribe video file with word-level timestamps and save to JSON.

    Args:
        video_file_path: Path to input video file
        output_json_path: Path to save JSON transcription (optional, uses default if not provided)
        api_key: OpenAI API key (optional, will use env if not provided)

    Returns:
        str: Path to the saved JSON transcription file
    """
    transcriber = VideoTranscriber(api_key=api_key)
    return transcriber.transcribe_video_to_json(
        video_file_path, output_json_path
    )


if __name__ == "__main__":
    # Example usage
    VIDEO_FILE_PATH = (
        f"b_roll/{base_data_dir}/video_generation/input_video/video.mp4"
    )

    try:
        # Transcribe video with word-level timestamps
        result_path = transcribe_video_to_json(video_file_path=VIDEO_FILE_PATH)
        logger.info(f"Transcription saved at: {result_path}")

    except Exception as e:
        logger.error(f"Failed to transcribe video: {e}")
