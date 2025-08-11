"""
Prompts for B-roll Generation

This module contains all prompts used in the b-roll generation process:
- System prompts for AI analysis
- User prompts for transcript analysis
- Fallback prompts for image and video generation
- Negative prompts for image/video generation
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from constants import (
    DEFAULT_CFG_SCALE,
    EARLY_DURATION_RATIO,
    EARLY_SEGMENT_RATIO,
    REMAINING_DURATION_RATIO,
    REMAINING_SEGMENT_RATIO,
    VIDEO_DURATION,
)

# System prompts for AI analysis
SYSTEM_PROMPT_ANALYSIS = """You are an expert video content analyst and cinematic director. Your task is to analyze a Russian audio transcript with word-level timestamps and identify the most important story themes for b-roll video generation.

Your analysis should:
1. Identify distinct story themes in the transcript
2. Create 5-second segments for each theme
3. Evaluate importance of each theme (1-10 scale)
4. Generate detailed image prompts for each theme
5. Generate detailed video prompts for each theme

Return your analysis in this exact JSON format:
{{
  "segments": [
    {{
      "start_time": float,
      "end_time": float,
      "text_context": "context around this segment",
      "theme_description": "brief description of the story theme",
      "importance_score": float (1-10),
      "keywords": ["keyword1", "keyword2"],
      "image_prompt": "detailed prompt for image generation",
      "video_prompt": "detailed cinematic prompt for video generation"
    }}
  ]
}}

⚠️ CRITICALLY IMPORTANT - START TIME REQUIREMENTS:
- ONLY select themes that begin AFTER 10.0 seconds in the transcript
- start_time must be EXACTLY taken from the word timestamps provided in the transcript data
- Find the EXACT timestamp where your selected text segment begins by matching words to their timestamps
- Use the "start" timestamp of the first word in your selected segment as the start_time
- DO NOT estimate or approximate - use PRECISE timestamp values from the word-level data
- Start timestamp must strictly correspond to the word timing data provided

⚠️ DISTRIBUTION STRATEGY - B-ROLL PLACEMENT REQUIREMENTS:
- Follow the specified distribution pattern for segment placement across video timeline
- Early segments (approximately 40% of total) should be placed in the first 25% of video duration
- Remaining segments (approximately 60% of total) should be distributed evenly across the remaining 75% of video duration
- Early segments should target high-impact themes that hook viewer attention
- Remaining segments should maintain engagement throughout the video
- Specific distribution parameters will be provided in the user prompt

Guidelines:
- Each segment should be exactly 5 seconds long
- Importance score: 1-3 (low), 4-6 (medium), 7-10 (high)
- You MUST return the EXACT number of segments requested - this is critical
- If fewer high-importance themes exist, include medium and low-importance themes to meet the count
- Image prompts should be detailed and specific for AI image generation
- Image prompts should focus on characters, people, and visual scenes without any text, letters, or written content
- Image prompts should avoid any mention of text, signs, labels, or written elements
- Video prompts should include camera angles, lighting, movement, style
- Video prompts should focus on dynamic scenes with character interactions and movement
- Focus on themes that would benefit from visual support
- Consider emotional impact and narrative flow
- Keywords should be relevant to the theme
- Text context should capture the essence of what's being said

Only return valid JSON, no additional text."""

# User prompt template for transcript analysis
USER_PROMPT_TEMPLATE = """
Analyze this Russian audio transcript with word-level timestamps and create b-roll segments:

Full transcript text: "{transcript_text}"

Word-level timestamps:
{word_timestamps}

Total duration: {duration} seconds
Target segment duration: {segment_duration} seconds
Maximum segments to select: {max_segments}

📊 B-ROLL DISTRIBUTION REQUIREMENTS:
- Early segments: {early_segment_count} segments ({early_segment_percentage}%) in first {early_duration_percentage}% ({early_duration_seconds} seconds)
- Remaining segments: {remaining_segment_count} segments ({remaining_segment_percentage}%) distributed evenly in remaining {remaining_duration_percentage}% ({remaining_duration_seconds} seconds)
- Early segments time range: {early_start_time} to {early_end_time} seconds
- Remaining segments time range: {remaining_start_time} to {remaining_end_time} seconds

Identify the most important story themes and create detailed prompts for both image and video generation. Focus on themes that would benefit most from visual support.

🔴 CRITICALLY IMPORTANT - EXACT START TIME MATCHING:
- ONLY analyze and select themes that begin AFTER 10.0 seconds in the transcript
- start_time must be EXACTLY taken from the word timestamps provided above
- Find the specific words that begin your selected theme and use the "start" timestamp of the first word
- DO NOT estimate or approximate start_time values - use the exact timestamps from the word data
- Match your selected text segments to the word timestamps to get precise start_time values
- If a theme starts before 10.0 seconds, skip it entirely and find other themes that start later

🎯 SEGMENT SELECTION REQUIREMENT:
- You MUST generate EXACTLY {max_segments} segments - no more, no less
- If you cannot find enough high-quality themes, lower your importance threshold and include more segments
- Better to have {max_segments} segments with varying importance than fewer segments
- Use the full range of importance scores (1-10) to reach the required count

EXAMPLE: If you want to select a segment starting with "Граница между программой", find those exact words in the word timestamps and use the start time of "Граница".

ADDITIONAL REQUIREMENTS: 
- For image prompts, avoid any text, letters, signs, or written content. Focus on visual scenes, people, objects, and environments without any textual elements.
- Use precise timestamps that align with the word-level timing data provided.

Return only the JSON response with segments sorted by importance_score (highest first).
"""

# Negative prompts for image generation
NEGATIVE_PROMPT_IMAGE = "blur, low quality, distorted, ugly, watermark, text, words, letters, signs, captions, subtitles, typography, fonts, writing, inscriptions, labels, logos, brand names, any alphanumeric characters, any symbols that represent text, any visual elements that could be interpreted as text"

# Negative prompts for video generation
NEGATIVE_PROMPT_VIDEO = "blur, distort, low quality, static image, static scene, no movement, still frame"

# Default generation parameters
DEFAULT_VIDEO_DURATION = str(
    int(VIDEO_DURATION)
)  # Use VIDEO_DURATION from constants.py, convert to integer string
# DEFAULT_CFG_SCALE moved to constants.py
