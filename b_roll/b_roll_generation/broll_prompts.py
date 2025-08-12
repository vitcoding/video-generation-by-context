#!/usr/bin/env python3
"""
B-roll Prompts Generator

This script analyzes audio transcripts and generates b-roll prompts
for video generation based on the content analysis.
"""

import json
import os
import re

# Import configuration and mock modules
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tiktoken

sys.path.append(str(Path(__file__).parent.parent))
from config import config
from constants import (
    AUDIO_TRANSCRIPT_DIR_NAME,
    BROLL_PROMPTS_DIR,
    DEFAULT_PROMPTS_FILE,
    MAX_SEGMENTS,
    OPENAI_MODEL,
    TRANSCRIPTION_JSON_FILENAME,
    VIDEO_DURATION,
    VIDEO_FPS,
    VIDEO_GENERATION_DIR_NAME,
    VIDEO_RESOLUTION,
    WORKFLOW_PROMPTS_FILENAME,
    base_data_dir,
)
from dotenv import load_dotenv
from logger_config import logger
from mock_api import mock_openai_client
from openai import OpenAI
from transcript_chunker import TranscriptChunk, create_transcript_chunks

sys.path.append(str(Path(__file__).parent))
from prompts import SYSTEM_PROMPT_ANALYSIS, USER_PROMPT_TEMPLATE

CHUNK_DEBUG = False
# For debugging
CHUNK_DEBUG = True

TIKTOKEN_AVAILABLE = True

load_dotenv()


# Try to import OpenAI, install if needed
if not config.is_api_enabled:
    # Use mock OpenAI client
    OpenAI = mock_openai_client


@dataclass
class BRollSegment:
    """Represents a b-roll segment with timing and content info."""

    start_time: float
    end_time: float
    duration: float
    text_context: str
    image_prompt: str
    video_prompt: str
    importance_score: float
    keywords: List[str]


class BRollAnalyzer:
    """Analyzes transcript and generates b-roll insertion points using LLM API."""

    def __init__(
        self,
        segment_duration: float = VIDEO_DURATION,
        max_segments: int = 3,
        early_segment_ratio: float = None,
        early_duration_ratio: float = None,
        token_threshold: int = 7000,
        target_tokens_per_chunk: int = 4000,
        min_chunk_start_time: float = 10.0,
    ):
        """
        Initialize the analyzer.

        Args:
            segment_duration: Duration of each b-roll segment in seconds (default: VIDEO_DURATION)
            max_segments: Maximum number of segments to select (default: 3)
            early_segment_ratio: Ratio of segments for early portion (default: from constants)
            early_duration_ratio: Ratio of video duration for early portion (default: from constants)
            token_threshold: Max total tokens before applying chunking strategy
            target_tokens_per_chunk: Target max tokens per chunk for safe prompts
            min_chunk_start_time: Seconds to skip at start when creating chunks (set 0 to disable)
        """
        self.segment_duration = segment_duration
        self.max_segments = max_segments

        # Import constants here to avoid circular imports
        import sys
        from pathlib import Path

        sys.path.append(str(Path(__file__).parent.parent))
        from constants import EARLY_DURATION_RATIO, EARLY_SEGMENT_RATIO

        # Use provided values or defaults from constants
        self.early_segment_ratio = (
            early_segment_ratio
            if early_segment_ratio is not None
            else EARLY_SEGMENT_RATIO
        )
        self.early_duration_ratio = (
            early_duration_ratio
            if early_duration_ratio is not None
            else EARLY_DURATION_RATIO
        )

        # Calculate derived values
        self.remaining_segment_ratio = 1.0 - self.early_segment_ratio
        self.remaining_duration_ratio = 1.0 - self.early_duration_ratio

        # Token/Chunking controls
        self.token_threshold = int(token_threshold)
        self.target_tokens_per_chunk = int(target_tokens_per_chunk)
        self.min_chunk_start_time = float(min_chunk_start_time)

        # Initialize OpenAI client
        api_key = config.get_api_key("OPENAI_API_KEY")

        if config.is_api_enabled and not api_key:
            logger.info("⚠️ Warning: OPENAI_API_KEY not found in .env file")
            logger.info("🔧 Using mock mode for prompt generation")
            # Don't raise error, continue with mock mode

        if config.is_api_enabled:
            self.client = OpenAI(api_key=api_key)
        else:
            logger.info("🔧 [MOCK] Using mock OpenAI client")
            self.client = OpenAI()

    def _save_chunks_to_temp_dir(
        self, chunks: List[TranscriptChunk], label: str
    ) -> None:
        """
        Save transcript chunks as JSON files into a temporary folder under BROLL_PROMPTS_DIR.

        Args:
            chunks: List of TranscriptChunk to save
            label: Subfolder label (e.g., "initial" or "final")
        """
        try:
            chunks_root = Path(BROLL_PROMPTS_DIR) / "chunks_temp"
            target_dir = chunks_root / label

            # Re-create label directory to ensure a clean state
            if target_dir.exists():
                # Remove existing files in the label directory
                for path in target_dir.glob("*"):
                    try:
                        path.unlink()
                    except Exception:
                        pass
            else:
                target_dir.mkdir(parents=True, exist_ok=True)

            for idx, chunk in enumerate(chunks, start=1):
                file_name = (
                    f"{label}_chunk_{idx:02d}_{chunk.chunk_type}_"
                    f"{chunk.start_time:.1f}s_{chunk.end_time:.1f}s.json"
                )
                file_path = target_dir / file_name

                # Serialize chunk dataclass to dict
                chunk_dict = {
                    "chunk_id": chunk.chunk_id,
                    "chunk_type": chunk.chunk_type,
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "duration": chunk.duration,
                    "target_segments": chunk.target_segments,
                    "text": chunk.text,
                    "word_timestamps": chunk.word_timestamps,
                }

                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(chunk_dict, f, ensure_ascii=False, indent=2)

            logger.info(
                f"💾 Saved {len(chunks)} '{label}' chunks to: {target_dir}"
            )
        except Exception as e:
            logger.error(f"❌ Failed to save '{label}' chunks: {e}")

    def count_tokens(self, text: str, model: str = None) -> int:
        """
        Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for
            model: Model name to select an appropriate encoding. Defaults to configured OPENAI_MODEL.

        Returns:
            Number of tokens
        """
        if model is None:
            model = OPENAI_MODEL

        if not TIKTOKEN_AVAILABLE:
            # Fallback estimation: ~4 characters per token
            return len(text) // 4

        try:
            # Get encoding for the model
            if isinstance(model, str) and model.startswith("gpt-4"):
                encoding = tiktoken.encoding_for_model("gpt-4")
            else:
                # Use cl100k_base encoding for other models (gpt-3.5, etc.)
                encoding = tiktoken.get_encoding("cl100k_base")

            # Count tokens
            return len(encoding.encode(text))
        except Exception as e:
            logger.info(f"⚠️ Token counting failed: {e}. Using rough estimate.")
            return len(text) // 4

    def load_transcript(self, file_path: str) -> Dict[str, Any]:
        """Load transcript JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Transcript file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _calculate_distribution_parameters(
        self, total_duration: float
    ) -> Dict[str, Any]:
        """
        Calculate distribution parameters for b-roll placement.

        Args:
            total_duration: Total video duration in seconds

        Returns:
            Dictionary with calculated distribution parameters
        """
        # Calculate segment counts
        early_segment_count = max(
            1, int(self.max_segments * self.early_segment_ratio)
        )
        remaining_segment_count = self.max_segments - early_segment_count

        # Calculate duration splits
        early_duration_seconds = total_duration * self.early_duration_ratio
        remaining_duration_seconds = (
            total_duration * self.remaining_duration_ratio
        )

        # Calculate time ranges (considering minimum start time of 10.0 seconds)
        min_start_time = 10.0
        early_start_time = min_start_time
        early_end_time = max(min_start_time, early_duration_seconds)
        remaining_start_time = early_end_time
        remaining_end_time = total_duration

        return {
            "early_segment_count": early_segment_count,
            "early_segment_percentage": int(self.early_segment_ratio * 100),
            "remaining_segment_count": remaining_segment_count,
            "remaining_segment_percentage": int(
                self.remaining_segment_ratio * 100
            ),
            "early_duration_percentage": int(self.early_duration_ratio * 100),
            "remaining_duration_percentage": int(
                self.remaining_duration_ratio * 100
            ),
            "early_duration_seconds": round(early_duration_seconds, 1),
            "remaining_duration_seconds": round(remaining_duration_seconds, 1),
            "early_start_time": round(early_start_time, 1),
            "early_end_time": round(early_end_time, 1),
            "remaining_start_time": round(remaining_start_time, 1),
            "remaining_end_time": round(remaining_end_time, 1),
            "total_duration": round(total_duration, 1),
        }

    def analyze_transcript_with_ai(
        self, transcript_data: Dict[str, Any]
    ) -> List[BRollSegment]:
        """
        Analyze transcript using OpenAI API to identify story themes and generate prompts.

        Args:
            transcript_data: Transcript data with text, words, and duration

        Returns:
            List of BRollSegment objects with timing, prompts, and importance scores
        """

        # Prepare word timestamps for the prompt
        word_timestamps = self._format_word_timestamps(
            transcript_data.get("words", [])
        )

        # Calculate distribution parameters
        distribution_params = self._calculate_distribution_parameters(
            transcript_data["duration"]
        )

        # Use system prompt as-is (contains JSON structure that shouldn't be formatted)
        system_prompt = SYSTEM_PROMPT_ANALYSIS

        user_prompt = USER_PROMPT_TEMPLATE.format(
            transcript_text=transcript_data["text"],
            word_timestamps=word_timestamps,
            duration=transcript_data["duration"],
            segment_duration=self.segment_duration,
            max_segments=self.max_segments,
            **distribution_params,
        )

        # Use configured model name directly
        selected_model = OPENAI_MODEL

        # Calculate safe max_tokens with a conservative fixed cap
        input_tokens = self.count_tokens(
            system_prompt + user_prompt, selected_model
        )

        # Conservative output token limit without relying on model context info
        safe_max_output = 1200

        try:
            logger.info("🤖 Analyzing transcript with OpenAI API...")
            logger.info(
                f"📊 Model: {selected_model}, Input: {input_tokens} tokens, Max output: {safe_max_output} tokens"
            )

            response = self.client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=safe_max_output,
            )

            response_content = response.choices[0].message.content.strip()

            # Parse JSON response
            try:
                # Check if response is wrapped in markdown code block
                if response_content.startswith("```json"):
                    # Extract JSON from markdown code block
                    json_start = response_content.find("```json") + 7
                    json_end = response_content.rfind("```")
                    if json_end > json_start:
                        response_content = response_content[
                            json_start:json_end
                        ].strip()
                elif response_content.startswith("```"):
                    # Handle generic code block
                    json_start = response_content.find("```") + 3
                    json_end = response_content.rfind("```")
                    if json_end > json_start:
                        response_content = response_content[
                            json_start:json_end
                        ].strip()

                ai_analysis = json.loads(response_content)
            except json.JSONDecodeError as e:
                logger.info(f"❌ Error parsing AI response: {e}")
                logger.info(f"Response content: {response_content}")
                raise SystemExit(
                    "❌ Failed to parse AI response JSON. Stopping execution."
                )

            # Convert AI analysis to BRollSegment objects
            segments = []
            for segment_data in ai_analysis.get("segments", []):
                # Get start_time from AI response (transcript timing)
                start_time = float(segment_data.get("start_time", 0))

                # Calculate end_time to ensure exact duration match
                end_time = start_time + self.segment_duration

                segment = BRollSegment(
                    start_time=start_time,
                    end_time=end_time,
                    duration=self.segment_duration,
                    text_context=segment_data.get("text_context", ""),
                    image_prompt=segment_data.get("image_prompt", ""),
                    video_prompt=segment_data.get("video_prompt", ""),
                    importance_score=float(
                        segment_data.get("importance_score", 5)
                    ),
                    keywords=segment_data.get("keywords", []),
                )
                segments.append(segment)

            # Sort by importance score and limit to max_segments
            segments.sort(key=lambda x: x.importance_score, reverse=True)
            segments = segments[: self.max_segments]

            # Sort by start time for final output
            segments.sort(key=lambda x: x.start_time)

            logger.info(
                f"✅ Generated {len(segments)} AI-powered b-roll segments"
            )
            return segments

        except Exception as e:
            logger.info(f"❌ Error in AI analysis: {e}")
            raise SystemExit(
                "❌ Failed to analyze transcript with AI. Stopping execution."
            )

    def analyze_transcript_with_adaptive_chunking(
        self, transcript_data: Dict[str, Any]
    ) -> List[BRollSegment]:
        """
        Analyze transcript using token-based chunking with existing transcript_chunker.

        This method checks if the transcript needs chunking based on token count.
        If the total tokens exceed self.token_threshold, it uses the existing transcript_chunker.py
        to split the transcript with proper segment distribution according to
        EARLY_SEGMENT_RATIO and REMAINING_SEGMENT_RATIO from constants.
        If individual chunks are still too large (> self.target_tokens_per_chunk tokens), they are further
        subdivided while maintaining the original segment allocation.

        Args:
            transcript_data: Complete transcript data with text, words, and duration

        Returns:
            List of BRollSegment objects with exact MAX_SEGMENTS count
        """
        logger.info("🔍 Starting adaptive transcript analysis...")

        # Prepare initial prompts to check token count
        word_timestamps = self._format_word_timestamps(
            transcript_data.get("words", [])
        )
        distribution_params = self._calculate_distribution_parameters(
            transcript_data["duration"]
        )

        system_prompt = SYSTEM_PROMPT_ANALYSIS
        user_prompt = USER_PROMPT_TEMPLATE.format(
            transcript_text=transcript_data["text"],
            word_timestamps=word_timestamps,
            duration=transcript_data["duration"],
            segment_duration=self.segment_duration,
            max_segments=self.max_segments,
            **distribution_params,
        )

        total_tokens = self.count_tokens(system_prompt + user_prompt)
        logger.info(f"📊 Total prompt tokens: {total_tokens}")

        # Threshold for chunking decision
        if total_tokens <= self.token_threshold:
            logger.info("✅ Token count within limits, using single request")
            return self.analyze_transcript_with_ai(transcript_data)

        logger.info(
            f"⚠️ Token count ({total_tokens}) exceeds threshold ({self.token_threshold})"
        )
        logger.info(
            "🧩 Using existing transcript chunker for proper segment distribution..."
        )

        # Use existing transcript_chunker.py for proper chunk creation
        initial_chunks = create_transcript_chunks(
            transcript_data,
            self.max_segments,
            min_start_time=self.min_chunk_start_time,
        )
        # Save initial chunks
        if CHUNK_DEBUG:
            self._save_chunks_to_temp_dir(initial_chunks, label="initial")

        # Check if chunks need further subdivision
        final_chunks = []
        for chunk in initial_chunks:
            subdivided_chunks = self._subdivide_large_chunk_if_needed(
                chunk, system_prompt, self.target_tokens_per_chunk
            )
            final_chunks.extend(subdivided_chunks)

        # Save final chunks (after subdivision decision)
        if CHUNK_DEBUG:
            self._save_chunks_to_temp_dir(final_chunks, label="final")

        all_segments = []
        total_requested_segments = 0

        for i, chunk in enumerate(final_chunks):
            logger.info(
                f"📦 Processing {chunk.chunk_type} chunk {i+1}/{len(final_chunks)}..."
            )
            logger.info(
                f"   • Time range: {chunk.start_time:.1f}s - {chunk.end_time:.1f}s"
            )
            logger.info(f"   • Target segments: {chunk.target_segments}")
            logger.info(f"   • Text length: {len(chunk.text)} chars")

            total_requested_segments += chunk.target_segments

            # Create chunk-specific transcript data
            chunk_transcript_data = {
                "text": chunk.text,
                "words": chunk.word_timestamps,
                "duration": chunk.duration,
            }

            # Verify token count for this chunk
            chunk_word_timestamps = self._format_word_timestamps(
                chunk.word_timestamps
            )
            chunk_distribution_params = (
                self._calculate_distribution_parameters(chunk.duration)
            )

            chunk_user_prompt = USER_PROMPT_TEMPLATE.format(
                transcript_text=chunk.text,
                word_timestamps=chunk_word_timestamps,
                duration=chunk.duration,
                segment_duration=self.segment_duration,
                max_segments=chunk.target_segments,
                **chunk_distribution_params,
            )

            chunk_tokens = self.count_tokens(system_prompt + chunk_user_prompt)
            logger.info(f"   • Chunk tokens: {chunk_tokens}")

            if chunk_tokens > self.target_tokens_per_chunk:
                logger.warning(
                    f"⚠️ Chunk {i+1} still has {chunk_tokens} tokens (target: {self.target_tokens_per_chunk})"
                )
                logger.warning(
                    "   This may cause API issues or higher costs due to model fallback"
                )

            # Temporarily override max_segments for this chunk
            original_max_segments = self.max_segments
            self.max_segments = chunk.target_segments

            try:
                # Analyze this chunk
                chunk_segments = self.analyze_transcript_with_ai(
                    chunk_transcript_data
                )

                # Adjust segment times to global timeline if needed
                for segment in chunk_segments:
                    # Ensure segments respect the chunk's time boundaries
                    if segment.start_time < chunk.start_time:
                        segment.start_time = chunk.start_time
                    if segment.end_time > chunk.end_time:
                        segment.end_time = chunk.end_time

                all_segments.extend(chunk_segments)
                logger.info(
                    f"✅ Generated {len(chunk_segments)} segments from {chunk.chunk_type} chunk {i+1}"
                )

            except Exception as e:
                logger.error(
                    f"❌ Error processing {chunk.chunk_type} chunk {i+1}: {e}"
                )
                # Continue with other chunks even if one fails
                continue

            finally:
                # Restore original max_segments
                self.max_segments = original_max_segments

        # Sort all segments by start time
        all_segments.sort(key=lambda x: x.start_time)

        logger.info(f"📊 Total requested segments: {total_requested_segments}")
        logger.info(f"📊 Total generated segments: {len(all_segments)}")

        # Ensure exact MAX_SEGMENTS count
        all_segments = self._ensure_exact_segment_count(all_segments)

        logger.info(f"🎯 Final segment count: {len(all_segments)}")
        return all_segments

    def _subdivide_large_chunk_if_needed(
        self, chunk: TranscriptChunk, system_prompt: str, target_tokens: int
    ) -> List[TranscriptChunk]:
        """
        Subdivide a chunk if it's too large, while maintaining segment allocation.

        Args:
            chunk: Original chunk to potentially subdivide
            system_prompt: System prompt for token calculation
            target_tokens: Target token count per chunk

        Returns:
            List of subdivided chunks or original chunk if subdivision not needed
        """
        # Calculate token count for this chunk
        chunk_word_timestamps = self._format_word_timestamps(
            chunk.word_timestamps
        )
        chunk_distribution_params = self._calculate_distribution_parameters(
            chunk.duration
        )

        chunk_user_prompt = USER_PROMPT_TEMPLATE.format(
            transcript_text=chunk.text,
            word_timestamps=chunk_word_timestamps,
            duration=chunk.duration,
            segment_duration=self.segment_duration,
            max_segments=chunk.target_segments,
            **chunk_distribution_params,
        )

        chunk_tokens = self.count_tokens(system_prompt + chunk_user_prompt)

        # If chunk is within limits, return as-is
        if chunk_tokens <= target_tokens:
            return [chunk]

        # Calculate how many sub-chunks we need
        subdivision_factor = max(2, int((chunk_tokens / target_tokens) + 0.5))
        logger.info(
            f"🔪 Subdividing {chunk.chunk_type} chunk into {subdivision_factor} sub-chunks"
        )

        # Create sub-chunks
        sub_chunks = []
        chunk_duration = chunk.duration
        sub_chunk_duration = chunk_duration / subdivision_factor

        # Distribute segments across sub-chunks
        segments_per_sub_chunk = self._distribute_segments_across_subchunks(
            chunk.target_segments, subdivision_factor
        )

        for i in range(subdivision_factor):
            sub_start_time = chunk.start_time + (i * sub_chunk_duration)
            sub_end_time = chunk.start_time + ((i + 1) * sub_chunk_duration)
            if (
                i == subdivision_factor - 1
            ):  # Last sub-chunk gets remaining duration
                sub_end_time = chunk.end_time

            # Filter words for this sub-chunk
            sub_chunk_words = []
            sub_chunk_text_parts = []

            for word_data in chunk.word_timestamps:
                word_start = word_data.get("start", 0)
                word_end = word_data.get("end", 0)

                # Include word if it overlaps with sub-chunk time range
                if word_start < sub_end_time and word_end > sub_start_time:
                    sub_chunk_words.append(word_data)
                    sub_chunk_text_parts.append(word_data.get("word", ""))

            # Create sub-chunk text
            sub_chunk_text = " ".join(sub_chunk_text_parts).strip()

            # Fallback if sub-chunk text is empty
            if not sub_chunk_text:
                logger.warning(
                    f"⚠️ Empty sub-chunk text, using portion of original text"
                )
                start_ratio = i / subdivision_factor
                end_ratio = (i + 1) / subdivision_factor
                start_idx = int(len(chunk.text) * start_ratio)
                end_idx = int(len(chunk.text) * end_ratio)
                sub_chunk_text = chunk.text[start_idx:end_idx]

            sub_chunk = TranscriptChunk(
                chunk_id=chunk.chunk_id * 100 + i + 1,  # Unique ID
                start_time=sub_start_time,
                end_time=sub_end_time,
                duration=sub_end_time - sub_start_time,
                text=sub_chunk_text,
                word_timestamps=sub_chunk_words,
                target_segments=segments_per_sub_chunk[i],
                chunk_type=f"{chunk.chunk_type}_sub{i+1}",
            )

            sub_chunks.append(sub_chunk)
            logger.info(
                f"   📦 Sub-chunk {i+1}: {len(sub_chunk_text)} chars, {segments_per_sub_chunk[i]} segments"
            )

        # Verify total segments preserved
        total_sub_segments = sum(sc.target_segments for sc in sub_chunks)
        if total_sub_segments != chunk.target_segments:
            logger.error(
                f"❌ Segment count mismatch after subdivision: {total_sub_segments} != {chunk.target_segments}"
            )

        return sub_chunks

    def _distribute_segments_across_subchunks(
        self, total_segments: int, num_subchunks: int
    ) -> List[int]:
        """
        Distribute segments evenly across sub-chunks.

        Args:
            total_segments: Total number of segments to distribute
            num_subchunks: Number of sub-chunks to distribute across

        Returns:
            List of segment counts for each sub-chunk
        """
        base_segments_per_subchunk = total_segments // num_subchunks
        remainder = total_segments % num_subchunks

        segments_per_subchunk = [base_segments_per_subchunk] * num_subchunks

        # Distribute remainder segments to ensure exact total
        for i in range(remainder):
            segments_per_subchunk[i] += 1

        logger.info(
            f"📊 Sub-chunk segment distribution: {segments_per_subchunk} (total: {sum(segments_per_subchunk)})"
        )

        assert (
            sum(segments_per_subchunk) == total_segments
        ), "Sub-chunk segment distribution error"

        return segments_per_subchunk

    def _ensure_exact_segment_count(
        self, segments: List[BRollSegment]
    ) -> List[BRollSegment]:
        """
        Ensure exactly MAX_SEGMENTS segments are returned.

        Args:
            segments: List of generated segments

        Returns:
            List with exactly MAX_SEGMENTS segments
        """
        current_count = len(segments)

        if current_count == self.max_segments:
            logger.info("✅ Segment count matches target exactly")
            return segments

        if current_count > self.max_segments:
            logger.info(
                f"📊 Trimming segments from {current_count} to {self.max_segments}"
            )
            # Sort by importance and take top segments
            segments.sort(key=lambda x: x.importance_score, reverse=True)
            segments = segments[: self.max_segments]
            # Re-sort by start time
            segments.sort(key=lambda x: x.start_time)
            return segments

        if current_count < self.max_segments:
            shortage = self.max_segments - current_count
            logger.warning(
                f"⚠️ Generated {current_count} segments, need {self.max_segments}"
            )
            logger.warning(
                f"   Missing {shortage} segments - this indicates API issues"
            )
            # Return what we have - better than failing completely
            return segments

        return segments

    def analyze_transcript_with_chunking(
        self, transcript_data: Dict[str, Any]
    ) -> List[BRollSegment]:
        """
        Legacy chunking method - now redirects to adaptive chunking.

        Args:
            transcript_data: Complete transcript data with text, words, and duration

        Returns:
            List of BRollSegment objects from adaptive chunking
        """
        logger.info("🔄 Redirecting to adaptive chunking method...")
        return self.analyze_transcript_with_adaptive_chunking(transcript_data)

    def _format_word_timestamps(self, words: List[Dict]) -> str:
        """
        Format word timestamps for inclusion in the AI prompt.

        Args:
            words: List of word objects with timestamps

        Returns:
            str: Formatted string with word timestamps
        """
        if not words:
            return "No word-level timestamps available."

        # Format words with timestamps, filtering out very short segments and focusing on content words
        formatted_words = []
        for word_data in words:
            word = word_data.get("word", "")
            start = word_data.get("start", 0)
            word_type = word_data.get("type", "word")

            # Include words and important punctuation, skip very short function words for brevity
            if word_type == "word" or word in [".", "?", "!"]:
                formatted_words.append(f"{word} ({start:.2f}s)")

        # Group words into lines for better readability
        lines = []
        current_line = []
        words_per_line = 8

        for i, formatted_word in enumerate(formatted_words):
            current_line.append(formatted_word)
            if (i + 1) % words_per_line == 0:
                lines.append(" | ".join(current_line))
                current_line = []

        # Add remaining words
        if current_line:
            lines.append(" | ".join(current_line))

        return "\n".join(lines)

    def generate_output_json(
        self, segments: List[BRollSegment], transcript_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate the final output JSON with b-roll prompts and timestamps."""
        output = {
            "audio_duration": transcript_data["duration"],
            "original_text": transcript_data["text"],
            "broll_segments": [],
            "generation_info": {
                "total_broll_segments": len(segments),
                "total_broll_duration": sum(seg.duration for seg in segments),
                "coverage_percentage": (
                    sum(seg.duration for seg in segments)
                    / transcript_data["duration"]
                )
                * 100,
                "segment_duration": self.segment_duration,
                "max_segments": self.max_segments,
                "selection_strategy": f"AI-powered analysis selecting top {self.max_segments} segments by importance score",
                "prompt_generation": f"OpenAI {OPENAI_MODEL}",
                "api_used": config.is_api_enabled,
            },
        }

        for i, segment in enumerate(segments):
            output["broll_segments"].append(
                {
                    "segment_id": i + 1,
                    "start_time": round(segment.start_time, 2),
                    "end_time": round(segment.end_time, 2),
                    "duration": round(segment.duration, 2),
                    "text_context": segment.text_context,
                    "image_prompt": segment.image_prompt,
                    "video_prompt": segment.video_prompt,
                    "keywords": segment.keywords,
                    "importance_score": round(segment.importance_score, 2),
                    "video_generation_params": {
                        "duration": f"{self.segment_duration} seconds",
                        "style": "professional, modern, high-quality",
                        "resolution": VIDEO_RESOLUTION,
                        "fps": VIDEO_FPS,
                    },
                    "generated_by": f"OpenAI {OPENAI_MODEL}",
                }
            )

        return output


def main():
    """Main function to process transcript and generate b-roll prompts."""
    # File paths - use data_mock when mock mode is enabled
    import sys

    sys.path.append(str(Path(__file__).parent.parent))
    from config import config

    base_data_dir = "data_mock" if config.is_mock_enabled else "data"

    INPUT_FILE = str(
        Path(__file__).parent.parent
        / f"{base_data_dir}/{VIDEO_GENERATION_DIR_NAME}/{AUDIO_TRANSCRIPT_DIR_NAME}/{TRANSCRIPTION_JSON_FILENAME}"
    )
    OUTPUT_FILE = str(Path(BROLL_PROMPTS_DIR) / WORKFLOW_PROMPTS_FILENAME)

    output_dir = Path(OUTPUT_FILE).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"📄 Output file: {OUTPUT_FILE}")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.info("❌ Error: OPENAI_API_KEY environment variable not found!")
        logger.info(
            "Please add your OpenAI API key to a .env file in the project root:"
        )
        logger.info("OPENAI_API_KEY=your_api_key_here")
        return

    # Initialize analyzer with custom parameters
    analyzer = BRollAnalyzer(
        segment_duration=VIDEO_DURATION, max_segments=MAX_SEGMENTS
    )

    try:
        # Load transcript
        logger.info("📁 Loading transcript...")
        transcript_data = analyzer.load_transcript(INPUT_FILE)

        # Analyze for b-roll opportunities using AI with adaptive chunking
        logger.info("🔍 Analyzing transcript for b-roll opportunities...")
        broll_segments = analyzer.analyze_transcript_with_adaptive_chunking(
            transcript_data
        )

        logger.info(
            f"✅ Generated {len(broll_segments)} AI-powered b-roll segments"
        )

        # Generate output JSON
        output_data = analyzer.generate_output_json(
            broll_segments, transcript_data
        )

        # Save output
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 B-roll prompts saved to: {OUTPUT_FILE}")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("🎬 AI-GENERATED CINEMATIC B-ROLL PROMPTS")
        logger.info("=" * 60)
        logger.info(
            f"• Total audio duration: {transcript_data['duration']:.2f} seconds"
        )
        logger.info(f"• B-roll segments: {len(broll_segments)}")
        logger.info(
            f"• Total b-roll duration: {sum(seg.duration for seg in broll_segments):.2f} seconds"
        )
        logger.info(
            f"• B-roll coverage: {(sum(seg.duration for seg in broll_segments) / transcript_data['duration']) * 100:.1f}%"
        )
        logger.info(f"• Segment duration: {analyzer.segment_duration} seconds")
        logger.info(f"• Max segments: {analyzer.max_segments}")
        logger.info(f"• Prompt generation: OpenAI {OPENAI_MODEL}")

        logger.info("\n🎯 Selected B-roll segments:")
        for i, segment in enumerate(broll_segments, 1):
            logger.info(
                f"\n{i}. {segment.start_time:.2f}s - {segment.end_time:.2f}s (Score: {segment.importance_score:.1f}/10)"
            )
            logger.info(f"   📝 Context: {segment.text_context[:60]}...")
            logger.info(f"   🏷️  Keywords: {', '.join(segment.keywords)}")
            logger.info(
                f"   🖼️  Image prompt preview: {segment.image_prompt[:100]}..."
            )
            logger.info(
                f"   🎬 Video prompt preview: {segment.video_prompt[:100]}..."
            )

        logger.info(f"\n🚀 Ready to use with AI video generators!")

    except Exception as e:
        logger.info(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
