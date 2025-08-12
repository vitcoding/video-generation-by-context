#!/usr/bin/env python3
"""
Transcript chunking module for processing large transcripts in parts.

This module handles splitting transcripts into manageable chunks based on
the segment distribution requirements defined in constants.py:
- EARLY_SEGMENT_RATIO: 40% of segments in first 25% of duration
- REMAINING_SEGMENT_RATIO: 60% of segments in remaining 75% of duration
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.append(str(Path(__file__).parent.parent))
from constants import (
    EARLY_DURATION_RATIO,
    EARLY_SEGMENT_RATIO,
    REMAINING_DURATION_RATIO,
    REMAINING_SEGMENT_RATIO,
)
from logger_config import logger


@dataclass
class TranscriptChunk:
    """Represents a chunk of transcript data with timing information."""

    chunk_id: int
    start_time: float
    end_time: float
    duration: float
    text: str
    word_timestamps: List[Dict[str, Any]]
    target_segments: int
    chunk_type: str  # "early" or "remaining"


class TranscriptChunker:
    """
    Handles splitting large transcripts into manageable chunks.
    """

    def __init__(self, max_segments: int = 20, min_start_time: float = 10.0):
        """
        Initialize the transcript chunker.

        Args:
            max_segments: Total number of segments to generate across all chunks
            min_start_time: Seconds to skip at the beginning (set 0 to disable)
        """
        self.max_segments = max_segments
        self.min_start_time = max(0.0, float(min_start_time))
        self.early_segments = max(1, int(max_segments * EARLY_SEGMENT_RATIO))
        self.remaining_segments = max_segments - self.early_segments

        logger.info(f"🔧 Chunker initialized:")
        logger.info(f"   • Total segments: {max_segments}")
        logger.info(
            f"   • Early segments: {self.early_segments} ({EARLY_SEGMENT_RATIO*100:.0f}%)"
        )
        logger.info(
            f"   • Remaining segments: {self.remaining_segments} ({REMAINING_SEGMENT_RATIO*100:.0f}%)"
        )
        logger.info(
            f"   • Min start time: {self.min_start_time:.1f}s (set 0 to disable skipping)"
        )

    def create_chunks(
        self, transcript_data: Dict[str, Any]
    ) -> List[TranscriptChunk]:
        """
        Split transcript into chunks based on timing distribution.

        Args:
            transcript_data: Complete transcript data with word timestamps

        Returns:
            List of TranscriptChunk objects
        """
        total_duration = transcript_data["duration"]
        word_timestamps = transcript_data.get("words", [])
        full_text = transcript_data["text"]

        # Calculate time boundaries
        early_end_time = total_duration * EARLY_DURATION_RATIO
        remaining_start_time = early_end_time

        # Configurable minimum start time (skip initial seconds if needed)
        min_start_time = self.min_start_time

        logger.info(f"📊 Chunking strategy:")
        logger.info(f"   • Total duration: {total_duration:.1f}s")
        logger.info(
            f"   • Early chunk: {min_start_time:.1f}s - {early_end_time:.1f}s"
        )
        logger.info(
            f"   • Remaining chunk: {remaining_start_time:.1f}s - {total_duration:.1f}s"
        )

        chunks = []

        # Create early chunk
        early_chunk = self._create_chunk(
            chunk_id=1,
            start_time=min_start_time,
            end_time=early_end_time,
            word_timestamps=word_timestamps,
            full_text=full_text,
            target_segments=self.early_segments,
            chunk_type="early",
        )
        chunks.append(early_chunk)

        # Create remaining chunk
        remaining_chunk = self._create_chunk(
            chunk_id=2,
            start_time=remaining_start_time,
            end_time=total_duration,
            word_timestamps=word_timestamps,
            full_text=full_text,
            target_segments=self.remaining_segments,
            chunk_type="remaining",
        )
        chunks.append(remaining_chunk)

        logger.info(f"✅ Created {len(chunks)} transcript chunks")
        return chunks

    def _create_chunk(
        self,
        chunk_id: int,
        start_time: float,
        end_time: float,
        word_timestamps: List[Dict[str, Any]],
        full_text: str,
        target_segments: int,
        chunk_type: str,
    ) -> TranscriptChunk:
        """
        Create a single transcript chunk.

        Args:
            chunk_id: Unique identifier for the chunk
            start_time: Start time in seconds
            end_time: End time in seconds
            word_timestamps: Complete word timestamp data
            full_text: Complete transcript text
            target_segments: Number of segments to generate from this chunk
            chunk_type: "early" or "remaining"

        Returns:
            TranscriptChunk object
        """
        # Filter words within time range
        chunk_words = []
        chunk_text_parts = []

        for word_data in word_timestamps:
            word_start = word_data.get("start", 0)
            word_end = word_data.get("end", 0)

            # Include word if it overlaps with chunk time range
            if word_start < end_time and word_end > start_time:
                chunk_words.append(word_data)
                chunk_text_parts.append(word_data.get("word", ""))

        # Create chunk text
        chunk_text = " ".join(chunk_text_parts).strip()

        # If chunk text is empty, use a portion of full text as fallback
        if not chunk_text:
            logger.warning(
                f"⚠️ Empty chunk text for {chunk_type} chunk, using fallback"
            )
            chunk_text = full_text[:1000] + "..."

        chunk = TranscriptChunk(
            chunk_id=chunk_id,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            text=chunk_text,
            word_timestamps=chunk_words,
            target_segments=target_segments,
            chunk_type=chunk_type,
        )

        logger.info(f"📦 {chunk_type.title()} chunk created:")
        logger.info(
            f"   • Duration: {chunk.duration:.1f}s ({start_time:.1f}s - {end_time:.1f}s)"
        )
        logger.info(f"   • Words: {len(chunk_words)}")
        logger.info(f"   • Text length: {len(chunk_text)} chars")
        logger.info(f"   • Target segments: {target_segments}")

        return chunk

    def validate_chunks(self, chunks: List[TranscriptChunk]) -> bool:
        """
        Validate that chunks meet the distribution requirements.

        Args:
            chunks: List of transcript chunks

        Returns:
            True if validation passes
        """
        total_target_segments = sum(chunk.target_segments for chunk in chunks)

        if total_target_segments != self.max_segments:
            logger.error(
                f"❌ Chunk validation failed: {total_target_segments} != {self.max_segments}"
            )
            return False

        logger.info(
            f"✅ Chunk validation passed: {total_target_segments} total segments"
        )
        return True


def create_transcript_chunks(
    transcript_data: Dict[str, Any],
    max_segments: int = 20,
    min_start_time: float = 10.0,
) -> List[TranscriptChunk]:
    """
    Convenience function to create transcript chunks.

    Args:
        transcript_data: Complete transcript data
        max_segments: Total number of segments to generate
        min_start_time: Seconds to skip at the beginning (set 0 to disable)

    Returns:
        List of TranscriptChunk objects
    """
    chunker = TranscriptChunker(
        max_segments=max_segments, min_start_time=min_start_time
    )
    chunks = chunker.create_chunks(transcript_data)

    if not chunker.validate_chunks(chunks):
        raise ValueError("Chunk validation failed")

    return chunks


if __name__ == "__main__":
    # Test the chunker with sample data
    sample_transcript = {
        "duration": 2359.68,
        "text": "Sample transcript text...",
        "words": [
            {"word": "Sample", "start": 10.0, "end": 10.5},
            {"word": "transcript", "start": 10.5, "end": 11.0},
            # ... more words would be here
        ],
    }

    chunks = create_transcript_chunks(sample_transcript, max_segments=20)
    logger.info(f"🧪 Test completed: {len(chunks)} chunks created")
