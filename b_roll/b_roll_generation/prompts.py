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

# Replace fragile imports with robust package/direct execution handling
try:
    from ..constants import (
        EARLY_DURATION_RATIO,
        EARLY_SEGMENT_RATIO,
        REMAINING_DURATION_RATIO,
        REMAINING_SEGMENT_RATIO,
        VIDEO_DURATION,
    )
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.append(str(_Path(__file__).resolve().parents[2]))
    from b_roll.constants import (
        EARLY_DURATION_RATIO,
        EARLY_SEGMENT_RATIO,
        REMAINING_DURATION_RATIO,
        REMAINING_SEGMENT_RATIO,
        VIDEO_DURATION,
    )

# System prompts for AI analysis
SYSTEM_PROMPT_ANALYSIS = """You are an expert video content analyst and cinematic director. Analyze a Russian audio transcript with word-level timestamps and select story themes for b-roll generation.

Obey these rules:
- Use ONLY the numeric values provided in the user message (total duration, target segment duration, number of segments, time windows, and quotas). Never invent values.
- Start time policy:
  - Select only segments that begin strictly after 10.0 seconds.
  - start_time must be the EXACT 'start' timestamp of the first word in the chosen text span.
  - Do not estimate or round; use precise floats from the word-level data.
  - Each segment must fully fit the timeline: start_time <= (total_duration - target_segment_duration).
- Distribution:
  - Allocate segments exactly according to the time windows and quotas described in the user message (e.g., early vs remaining).
  - Each segment must lie entirely within its assigned window.
  - Segments must not overlap. For i<j: segments[j].start_time >= segments[i].start_time + target_segment_duration.
- Language per field:
  - text_context: Russian (1–2 short sentences).
  - image_prompt, video_prompt, keywords: English only.
- Image prompt:
  - No text, letters, signs, captions, subtitles, logos, brands, watermarks.
  - Concrete, visual, photorealistic if relevant; avoid repeating transcript text.
  - Keep concise (<= 220 characters).
- Video prompt:
  - Cinematic, include: shot type, subject, action, setting, camera movement, lens, composition, lighting, color palette, mood, era/style.
  - Dynamic (avoid static scenes). Keep concise (<= 300 characters).
- Keywords:
  - 5–8 lowercase English nouns, single words, no verbs or phrases.
- Diversity:
  - Select segments that are thematically and visually diverse: vary topics, settings, subjects, actions, camera/lens choices, composition, lighting, color palette, mood.
  - Avoid redundancy: do not choose multiple segments with similar meaning or nearly identical prompts/keywords.
- People depiction policy:
  - If people are depicted, default to "Caucasian, middle-aged adults" unless an explicit demographic or age is already specified by the scene context. Do not contradict explicit context.
- Output:
  - Return ONLY valid JSON, no markdown, no comments.
  - Exactly the number of segments specified by the user.
  - Sort segments by importance_score (highest first).

JSON structure (example, values are illustrative):
{
  "segments": [
    {
      "start_time": 12.34,
      "end_time": 17.34,
      "text_context": "Короткая русская выжимка контекста.",
      "theme_description": "Brief English theme title.",
      "importance_score": 7.5,
      "keywords": ["business", "meeting", "office", "team", "discussion"],
      "image_prompt": "English, concrete visual description, no text/logos/watermarks.",
      "video_prompt": "English, cinematic description with camera, lens, movement, lighting, mood."
    }
  ]
}
"""

# User prompt template for transcript analysis
USER_PROMPT_TEMPLATE = """
Analyze this audiotranscript and produce exactly {max_segments} b-roll segments.

Full transcript text:
"{transcript_text}"

Word-level timestamps:
{word_timestamps}

Global constraints:
- Total duration: {duration} seconds
- Target segment duration: {segment_duration} seconds
- Strict start time policy:
  - start_time must be EXACTLY the 'start' timestamp of the first word of the chosen text span.
  - No estimation, no rounding.

Distribution requirements (use these quotas and windows):
- Early window: from {early_start_time}s to {early_end_time}s, allocate exactly {early_segment_count} segments.
- Remaining window: from {remaining_start_time}s to {remaining_end_time}s, allocate exactly {remaining_segment_count} segments.
- Segments must not overlap and must fully fit into their window.

Formatting and quality:
- Return ONLY JSON with exactly {max_segments} segments.
- Sort segments by importance_score (highest first).
- text_context in Russian (1–2 short sentences).
- image_prompt and video_prompt in English.
- image_prompt: concise (<= 220 chars), detailed, photorealistic if relevant, no text/logos/watermarks.
- video_prompt: concise (<= 300 chars), include: shot type, subject, action, setting, camera movement, lens, composition, lighting, color palette, mood, era/style; avoid static scenes.
- keywords: 5–8 lowercase English nouns, no phrases or verbs.
- Ensure diversity: select segments that vary in themes, visuals, settings, subjects, actions, camera/lens, composition, lighting, color palette, mood; avoid redundant or near-duplicate prompts/keywords.

Example instruction:
If selecting a segment starting with “Граница между программой…”, find these exact words in the timestamps and use the 'start' time of the first word “Граница”.

Return ONLY the JSON object.
"""

# Negative prompts for image generation
NEGATIVE_PROMPT_IMAGE = "blur, low quality, distorted, ugly, watermark, text, words, letters, signs, captions, subtitles, logos, brand names, any alphanumeric characters"

# Negative prompts for video generation
NEGATIVE_PROMPT_VIDEO = "blur, distort, low quality, static image, static scene, no movement, still frame, text, words, letters, captions, subtitles, logos, brand names, watermarks"

# Default generation parameters
DEFAULT_VIDEO_DURATION = str(
    int(VIDEO_DURATION)
)  # Use VIDEO_DURATION from constants.py, convert to integer string
