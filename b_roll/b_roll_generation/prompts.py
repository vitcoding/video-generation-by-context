"""
Prompts for B-roll Generation

This module contains all prompts used in the b-roll generation process:
- System prompts for AI analysis
- User prompts for transcript analysis
- Fallback prompts for image and video generation
- Negative prompts for image/video generation
"""

# System prompts for AI analysis
SYSTEM_PROMPT_ANALYSIS = """You are an expert video content analyst and cinematic director. Your task is to analyze a Russian audio transcript and identify the most important story themes for b-roll video generation.

Your analysis should:
1. Identify distinct story themes in the transcript
2. Create 5-second segments for each theme
3. Evaluate importance of each theme (1-10 scale)
4. Generate detailed image prompts for each theme
5. Generate detailed video prompts for each theme

Return your analysis in this exact JSON format:
{
  "segments": [
    {
      "start_time": float,
      "end_time": float,
      "text_context": "context around this segment",
      "theme_description": "brief description of the story theme",
      "importance_score": float (1-10),
      "keywords": ["keyword1", "keyword2"],
      "image_prompt": "detailed prompt for image generation",
      "video_prompt": "detailed cinematic prompt for video generation"
    }
  ]
}

Guidelines:
- Each segment should be exactly 5 seconds long
- Importance score: 1-3 (low), 4-6 (medium), 7-10 (high)
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
Analyze this Russian audio transcript and create b-roll segments:

Transcript text: "{transcript_text}"
Total duration: {duration} seconds
Target segment duration: {segment_duration} seconds
Maximum segments to select: {max_segments}

Identify the most important story themes and create detailed prompts for both image and video generation. Focus on themes that would benefit most from visual support.

IMPORTANT: For image prompts, avoid any text, letters, signs, or written content. Focus on visual scenes, people, objects, and environments without any textual elements.

Return only the JSON response with segments sorted by importance_score (highest first).
"""

# Negative prompts for image generation
NEGATIVE_PROMPT_IMAGE = "blur, low quality, distorted, ugly, watermark, text, words, letters, signs, captions, subtitles, typography, fonts, writing, inscriptions, labels, logos, brand names, any alphanumeric characters, any symbols that represent text, any visual elements that could be interpreted as text"

# Negative prompts for video generation
NEGATIVE_PROMPT_VIDEO = "blur, distort, low quality, static image, static scene, no movement, still frame"

# Default generation parameters
DEFAULT_VIDEO_DURATION = "5"
DEFAULT_CFG_SCALE = 0.7
