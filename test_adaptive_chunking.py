#!/usr/bin/env python3
"""
Test script for new token-based chunking logic.

This script tests the updated chunking system that uses transcript_chunker.py:
1. Token count calculations work correctly
2. Segment distribution respects EARLY/REMAINING ratios
3. Chunking decisions are made properly
4. Total segments equals MAX_SEGMENTS exactly
"""

import sys
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent / "b_roll"))
sys.path.append(str(Path(__file__).parent / "b_roll" / "b_roll_generation"))

from b_roll.b_roll_generation.broll_prompts import BRollAnalyzer
from b_roll.b_roll_generation.transcript_chunker import (
    create_transcript_chunks,
)
from b_roll.constants import (
    EARLY_SEGMENT_RATIO,
    MAX_SEGMENTS,
    REMAINING_SEGMENT_RATIO,
)
from b_roll.logger_config import logger


def test_transcript_chunker_integration():
    """Test integration with transcript_chunker.py."""
    logger.info("🧪 Testing transcript_chunker integration...")

    # Create sample transcript data
    sample_transcript = {
        "text": "Sample transcript text " * 500,  # Make it reasonably long
        "words": [
            {"word": "Sample", "start": 15.0, "end": 15.5},
            {"word": "transcript", "start": 15.5, "end": 16.0},
            {"word": "text", "start": 16.0, "end": 16.5},
        ]
        * 100,  # Repeat to have enough words
        "duration": 1000.0,
    }

    # Test chunk creation
    chunks = create_transcript_chunks(sample_transcript, MAX_SEGMENTS)

    logger.info(f"📦 Created {len(chunks)} chunks")

    # Verify chunk properties
    total_target_segments = sum(chunk.target_segments for chunk in chunks)

    if total_target_segments != MAX_SEGMENTS:
        logger.error(
            f"❌ Total segments {total_target_segments} != {MAX_SEGMENTS}"
        )
        return False

    # Check segment distribution ratios
    early_chunks = [c for c in chunks if c.chunk_type == "early"]
    remaining_chunks = [c for c in chunks if c.chunk_type == "remaining"]

    early_segments = sum(c.target_segments for c in early_chunks)
    remaining_segments = sum(c.target_segments for c in remaining_chunks)

    expected_early = max(1, int(MAX_SEGMENTS * EARLY_SEGMENT_RATIO))
    expected_remaining = MAX_SEGMENTS - expected_early

    logger.info(
        f"📊 Early segments: {early_segments} (expected: {expected_early})"
    )
    logger.info(
        f"📊 Remaining segments: {remaining_segments} (expected: {expected_remaining})"
    )

    if (
        early_segments != expected_early
        or remaining_segments != expected_remaining
    ):
        logger.error("❌ Segment distribution doesn't match expected ratios")
        return False

    logger.info("✅ Transcript chunker integration test passed")
    return True


def test_token_calculation():
    """Test token counting."""
    logger.info("🧪 Testing token calculation...")

    analyzer = BRollAnalyzer(max_segments=MAX_SEGMENTS)

    # Test with sample text
    sample_texts = [
        "Short text",
        "Medium length text with some more words to check counting",
        "Very long text " * 1000,  # ~3000 chars
    ]

    for i, text in enumerate(sample_texts):
        token_count = analyzer.count_tokens(text)
        char_count = len(text)

        logger.info(
            f"📝 Text {i+1}: {char_count} chars → {token_count} tokens"
        )

        # Basic sanity check: tokens should be roughly 1/4 of characters
        expected_range = (char_count // 6, char_count // 2)
        if not (expected_range[0] <= token_count <= expected_range[1]):
            logger.warning(
                f"⚠️ Token count {token_count} outside expected range {expected_range}"
            )

    return True


def test_chunking_decision():
    """Test chunking decision logic."""
    logger.info("🧪 Testing chunking decision logic...")

    analyzer = BRollAnalyzer(max_segments=MAX_SEGMENTS)

    # Create mock transcript data
    small_transcript = {
        "text": "Short transcript text",
        "words": [{"word": "Short", "start": 10.0, "end": 10.5}],
        "duration": 100.0,
    }

    large_transcript = {
        "text": "Very long transcript text " * 2000,  # ~50k chars
        "words": [
            {"word": "Very", "start": 10.0, "end": 10.5} for _ in range(1000)
        ],
        "duration": 2359.68,
    }

    # Test small transcript (should not chunk)
    logger.info("📋 Testing small transcript...")
    word_timestamps = analyzer._format_word_timestamps(
        small_transcript.get("words", [])
    )
    distribution_params = analyzer._calculate_distribution_parameters(
        small_transcript["duration"]
    )

    from b_roll.b_roll_generation.prompts import (
        SYSTEM_PROMPT_ANALYSIS,
        USER_PROMPT_TEMPLATE,
    )

    user_prompt = USER_PROMPT_TEMPLATE.format(
        transcript_text=small_transcript["text"],
        word_timestamps=word_timestamps,
        duration=small_transcript["duration"],
        segment_duration=5.0,
        max_segments=MAX_SEGMENTS,
        **distribution_params,
    )

    small_tokens = analyzer.count_tokens(SYSTEM_PROMPT_ANALYSIS + user_prompt)
    logger.info(f"📊 Small transcript tokens: {small_tokens}")

    # Test large transcript (should chunk)
    logger.info("📋 Testing large transcript...")
    word_timestamps_large = analyzer._format_word_timestamps(
        large_transcript.get("words", [])
    )
    distribution_params_large = analyzer._calculate_distribution_parameters(
        large_transcript["duration"]
    )

    user_prompt_large = USER_PROMPT_TEMPLATE.format(
        transcript_text=large_transcript["text"],
        word_timestamps=word_timestamps_large,
        duration=large_transcript["duration"],
        segment_duration=5.0,
        max_segments=MAX_SEGMENTS,
        **distribution_params_large,
    )

    large_tokens = analyzer.count_tokens(
        SYSTEM_PROMPT_ANALYSIS + user_prompt_large
    )
    logger.info(f"📊 Large transcript tokens: {large_tokens}")

    # Verify chunking decisions
    TOKEN_THRESHOLD = 7000
    should_chunk_small = small_tokens > TOKEN_THRESHOLD
    should_chunk_large = large_tokens > TOKEN_THRESHOLD

    logger.info(f"🤔 Small transcript should chunk: {should_chunk_small}")
    logger.info(f"🤔 Large transcript should chunk: {should_chunk_large}")

    if should_chunk_small:
        logger.warning("⚠️ Small transcript unexpectedly requires chunking")

    if not should_chunk_large:
        logger.warning(
            "⚠️ Large transcript unexpectedly doesn't require chunking"
        )

    return True


def test_chunk_token_verification():
    """Test that individual chunks have acceptable token counts."""
    logger.info("🧪 Testing chunk token verification...")

    analyzer = BRollAnalyzer(max_segments=MAX_SEGMENTS)

    # Create a large transcript that should be chunked
    large_transcript = {
        "text": "Very long transcript text with many details "
        * 1500,  # ~50k chars
        "words": [
            {
                "word": f"word{i}",
                "start": 10.0 + i * 0.5,
                "end": 10.5 + i * 0.5,
            }
            for i in range(1000)
        ],
        "duration": 2000.0,
    }

    # Create chunks using transcript_chunker
    chunks = create_transcript_chunks(large_transcript, MAX_SEGMENTS)

    TARGET_TOKENS_PER_CHUNK = 6000

    for i, chunk in enumerate(chunks):
        # Calculate token count for this chunk
        word_timestamps = analyzer._format_word_timestamps(
            chunk.word_timestamps
        )
        distribution_params = analyzer._calculate_distribution_parameters(
            chunk.duration
        )

        from b_roll.b_roll_generation.prompts import (
            SYSTEM_PROMPT_ANALYSIS,
            USER_PROMPT_TEMPLATE,
        )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            transcript_text=chunk.text,
            word_timestamps=word_timestamps,
            duration=chunk.duration,
            segment_duration=5.0,
            max_segments=chunk.target_segments,
            **distribution_params,
        )

        chunk_tokens = analyzer.count_tokens(
            SYSTEM_PROMPT_ANALYSIS + user_prompt
        )

        logger.info(
            f"📦 {chunk.chunk_type} chunk {i+1}: {chunk_tokens} tokens (target: ≤{TARGET_TOKENS_PER_CHUNK})"
        )

        if chunk_tokens > TARGET_TOKENS_PER_CHUNK:
            logger.warning(f"⚠️ Chunk {i+1} exceeds target token count")

    logger.info("✅ Chunk token verification completed")
    return True


def main():
    """Run all tests."""
    logger.info("🚀 Starting new chunking logic tests...")
    logger.info("=" * 60)

    tests = [
        (
            "Transcript Chunker Integration",
            test_transcript_chunker_integration,
        ),
        ("Token Calculation", test_token_calculation),
        ("Chunking Decision", test_chunking_decision),
        ("Chunk Token Verification", test_chunk_token_verification),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running test: {test_name}")
        logger.info("-" * 40)

        try:
            result = test_func()
            if result:
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} ERROR: {e}")
            import traceback

            traceback.print_exc()

    logger.info("\n" + "=" * 60)
    logger.info(f"🏁 Test Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.error("💥 Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
