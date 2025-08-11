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

sys.path.append(str(Path(__file__).parent.parent))
from config import config
from constants import (
    AUDIO_TRANSCRIPT_DIR_NAME,
    BROLL_PROMPTS_DIR,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_PROMPTS_FILE,
    FALLBACK_OPENAI_MODEL,
    MAX_SEGMENTS,
    OPENAI_MODELS,
    PREMIUM_OPENAI_MODEL,
    TRANSCRIPTION_JSON_FILENAME,
    VIDEO_DURATION,
    VIDEO_FPS,
    VIDEO_GENERATION_DIR_NAME,
    VIDEO_RESOLUTION,
    WORKFLOW_PROMPTS_FILENAME,
    base_data_dir,
)
from logger_config import logger
from mock_api import mock_openai_client

sys.path.append(str(Path(__file__).parent))
from prompts import SYSTEM_PROMPT_ANALYSIS, USER_PROMPT_TEMPLATE

# Try to import tiktoken for token counting
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    logger.info("tiktoken library not found. Installing...")
    import subprocess

    subprocess.check_call(["pip", "install", "tiktoken"])
    import tiktoken

    TIKTOKEN_AVAILABLE = True

# Try to import OpenAI, install if needed
if config.is_api_enabled:
    try:
        from openai import OpenAI
    except ImportError:
        logger.info("OpenAI library not found. Installing...")
        import subprocess

        subprocess.check_call(["pip", "install", "openai"])
        from openai import OpenAI
else:
    # Use mock OpenAI client
    OpenAI = mock_openai_client

# Try to import dotenv, install if needed
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    logger.info("python-dotenv not found. Installing...")
    import subprocess

    subprocess.check_call(["pip", "install", "python-dotenv"])
    from dotenv import load_dotenv

    load_dotenv()


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
    ):
        """
        Initialize the analyzer.

        Args:
            segment_duration: Duration of each b-roll segment in seconds (default: VIDEO_DURATION)
            max_segments: Maximum number of segments to select (default: 3)
            early_segment_ratio: Ratio of segments for early portion (default: from constants)
            early_duration_ratio: Ratio of video duration for early portion (default: from constants)
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

    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """
        Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for
            model: Model name to get appropriate encoding

        Returns:
            Number of tokens
        """
        if not TIKTOKEN_AVAILABLE:
            # Fallback estimation: ~4 characters per token
            return len(text) // 4

        try:
            # Get encoding for the model
            if model.startswith("gpt-4"):
                encoding = tiktoken.encoding_for_model("gpt-4")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")

            return len(encoding.encode(text))
        except Exception as e:
            logger.info(
                f"⚠️ Token counting error: {e}, using fallback estimation"
            )
            return len(text) // 4

    def select_best_model(self, system_prompt: str, user_prompt: str) -> str:
        """
        Select the best OpenAI model based on token count.

        Args:
            system_prompt: System prompt text
            user_prompt: User prompt text

        Returns:
            Model name to use
        """
        total_text = system_prompt + user_prompt

        # Try models in order of preference (cost-effective first)
        models_to_try = [
            DEFAULT_OPENAI_MODEL,
            FALLBACK_OPENAI_MODEL,
            PREMIUM_OPENAI_MODEL,
        ]

        for model_key in models_to_try:
            model_info = OPENAI_MODELS[model_key]
            token_count = self.count_tokens(total_text, model_info["name"])

            # Leave some buffer for response tokens (estimated 2000 tokens)
            max_input_tokens = model_info["max_context_tokens"] - 2000

            if token_count <= max_input_tokens:
                if model_key != DEFAULT_OPENAI_MODEL:
                    logger.info(
                        f"🔄 Switching to {model_info['name']} due to token limit"
                    )
                    logger.info(
                        f"📊 Token count: {token_count}, Model limit: {model_info['max_context_tokens']}"
                    )
                else:
                    logger.info(
                        f"✅ Using {model_info['name']} - Token count: {token_count}"
                    )
                return model_info["name"]

        # If even the most powerful model can't handle it, use it anyway and let OpenAI handle the error
        logger.info(
            f"⚠️ Text too long even for {PREMIUM_OPENAI_MODEL}, trying anyway..."
        )
        return OPENAI_MODELS[PREMIUM_OPENAI_MODEL]["name"]

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

        # Automatically select the best model based on token count
        selected_model = self.select_best_model(system_prompt, user_prompt)

        # Find model info for max_tokens
        model_info = None
        for model_key, info in OPENAI_MODELS.items():
            if info["name"] == selected_model:
                model_info = info
                break

        max_output_tokens = (
            model_info["max_output_tokens"] if model_info else 1500
        )

        try:
            logger.info("🤖 Analyzing transcript with OpenAI API...")

            response = self.client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=min(
                    max_output_tokens, 8000
                ),  # Increased for more segments
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
                segment = BRollSegment(
                    start_time=float(segment_data.get("start_time", 0)),
                    end_time=float(
                        segment_data.get("end_time", self.segment_duration)
                    ),
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

    # def _generate_fallback_segments(
    #     self, transcript_data: Dict[str, Any]
    # ) -> List[BRollSegment]:
    #     """
    #     Generate fallback segments when AI analysis fails.
    #
    #     DEPRECATED: This method is no longer used. Script stops on errors instead.
    #     """
    #     pass

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
                "prompt_generation": "OpenAI GPT-4 API",
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
                    "generated_by": "OpenAI GPT-4 API",
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

        # Analyze for b-roll opportunities using AI
        logger.info("🔍 Analyzing transcript for b-roll opportunities...")
        broll_segments = analyzer.analyze_transcript_with_ai(transcript_data)

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
        logger.info(f"• Prompt generation: OpenAI GPT-4 API")

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
