# Video Generation by Context

A comprehensive Python project for generating B-roll videos from audio transcripts using AI-powered image and video generation.

## 🎯 Project Overview

This project implements a complete pipeline for converting audio transcripts into B-roll video content. It analyzes transcripts, generates contextual prompts, creates images, and converts them into videos using advanced AI models.

## 🚀 Features

- **Video Transcription**: Automatic word-level timestamped transcript generation from video files
- **AI-Powered B-roll Analysis**: Intelligent transcript analysis with contextual B-roll prompt generation
- **High-Quality Image Generation**: Creates professional images using FAL AI's model
- **Dynamic Video Generation**: Converts static images to motion videos using AI technology
- **Automated B-roll Integration**: Seamless insertion of B-roll content into main video with precise timing
- **Flexible Workflow Execution**: Start from any stage (transcription or analysis) with configurable parameters
- **Stage Skipping**: Option to skip video generation for image-only workflows
- **Comprehensive Logging**: Detailed progress tracking with success rates and performance metrics
- **Mock Mode**: Complete testing capabilities with mock API responses
- **Intelligent File Organization**: Automatic organization and cleanup of generated files
- **Configurable Parameters**: Customizable video resolution, duration, quality, and segment limits
- **Error Recovery**: Robust error handling with graceful degradation

## 📁 Project Structure

```
video-generation-by-context/
├── main.py                 # Main entry point and workflow orchestration
├── requirements.txt        # Python dependencies
├── b_roll/                # Core project module
│   ├── b_roll_generation/ # AI generation components
│   │   ├── word_level_transcriber.py # Video to transcript conversion
│   │   ├── broll_prompts.py      # Transcript to B-roll prompt analysis
│   │   ├── broll_image_generation.py # AI image generation from prompts
│   │   ├── kling_image_to_video.py   # AI video generation from images
│   │   ├── video_editing_cloudinary.py # B-roll video integration
│   │   └── prompts.py             # Prompt templates and utilities
│   ├── workflow.py        # Unified workflow orchestration with flexible execution
│   ├── config.py          # Configuration management and environment variables
│   ├── constants.py       # Project constants and workflow settings
│   ├── logger_config.py   # Comprehensive logging configuration
│   ├── mock_api.py        # Mock API responses for testing
│   ├── data/              # Production data directory
│   └── data_mock/         # Mock data for testing
└── README.md              # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd video-generation-by-context
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the `b_roll/` directory:
   ```
   API_REQUEST=true  # Set to false for mock mode
   OPENAI_API_KEY=your_openai_key_here
   FAL_KEY=your_fal_key_here
   CLOUDINARY_CLOUD_NAME=your_cloudinary_cloud_name
   CLOUDINARY_API_KEY=your_cloudinary_api_key
   CLOUDINARY_API_SECRET=your_cloudinary_api_secret
   ```

## 🎮 Usage

### Basic Usage

Run the complete workflow:
```bash
python main.py
```

### Advanced Workflow Control

The workflow now supports flexible execution with multiple entry points:

#### 1. Complete Workflow from Video
```python
from b_roll.workflow import UnifiedWorkflow

# Standard workflow from video to final result
workflow = UnifiedWorkflow(max_segments=3)
results = workflow.run_from_video("path/to/video.mp4")
```

#### 2. Start from Existing Transcript
```python
# Start from existing transcript (uses default path automatically)
workflow = UnifiedWorkflow(start_stage="analysis")
results = workflow.run_from_transcript()

# Start from custom transcript file
workflow = UnifiedWorkflow(start_stage="analysis")
results = workflow.run_from_transcript("path/to/custom/transcript.json")
```

#### 3. Generate Images Only
```python
# Generate only images (skip video generation)
workflow = UnifiedWorkflow(skip_video_generation=True)
results = workflow.run_images_only(video_file_path="path/to/video.mp4")
```

### Workflow Parameters

- **`max_segments`**: Maximum number of B-roll segments to generate (default: 3)
- **`start_stage`**: Starting point of workflow
  - `"transcription"` - Start from video transcription (default)
  - `"analysis"` - Start from existing transcript analysis
- **`skip_video_generation`**: Skip video generation and final editing (default: False)

### Configuration Options

The project supports several configuration options:

- **API Mode**: Toggle between real API calls and mock mode
- **Video Resolution**: Configure output video quality (1280x720, 1920x1080, etc.)
- **Video Duration**: Set video length in seconds
- **Max Segments**: Control number of video segments generated
- **Flexible Execution**: Choose starting stage and skip unwanted stages

### Workflow Steps

1. **Video Transcription**: Converts video to word-level timestamped transcript
2. **Transcript Analysis**: Analyzes transcript and generates contextual B-roll prompts
3. **Image Generation**: Creates high-quality images from prompts using AI models
4. **Video Creation**: Converts images to videos with specified parameters
5. **B-roll Insertion**: Inserts generated B-roll videos into the main video
6. **File Organization**: Organizes generated files

## 🔧 Configuration

### Environment Variables

- `API_REQUEST`: Enable/disable real API calls (true/false)
- `OPENAI_API_KEY`: OpenAI API key for B-roll prompt analysis and generation
- `FAL_KEY`: FAL AI API key for high-quality image generation and image-to-video conversion
- `CLOUDINARY_CLOUD_NAME`: Cloudinary cloud name for video processing
- `CLOUDINARY_API_KEY`: Cloudinary API key for video editing services
- `CLOUDINARY_API_SECRET`: Cloudinary API secret for secure operations

### Constants (constants.py)

Key configurable parameters:
- `VIDEO_RESOLUTION`: Output video resolution
- `VIDEO_DURATION`: Video length in seconds
- `VIDEO_FPS`: Frames per second
- `IMAGE_ASPECT_RATIO`: Image aspect ratio
- `ENABLE_VIDEO_GENERATION`: Toggle video generation

## 🧪 Testing

The project includes comprehensive testing capabilities:

### Testing Modes

- **Mock Mode**: Test without API calls using `API_REQUEST=false`

### Mock Data

- **Mock Transcript Data**: Pre-generated transcript files for testing
- **Mock API Responses**: Simulated AI service responses
- **Test Video Files**: Sample videos for workflow testing

## 📊 Output Structure

Generated files are organized in the following structure:
```
b_roll/data/video_generation/
├── input_video/           # Input video files
├── audio_transcript/      # Word-level timestamped transcripts (JSON)
├── broll_prompts/         # Generated B-roll prompts with timing
├── images_input/          # AI-generated images from prompts
├── videos_output/         # B-roll videos converted from images
├── video_output/          # Final edited videos with B-roll insertions
└── workflow_complete_report.json   # Detailed workflow execution report
```

### Output Files Description

- **Transcript Files**: JSON format with word-level timestamps and confidence scores
- **Prompt Files**: Structured B-roll prompts with image/video instructions and timing data
- **Image Files**: High-resolution images with descriptive filenames including timing
- **Video Files**: MP4 B-roll videos with motion and professional quality
- **Report Files**: Comprehensive execution reports with success rates and performance metrics

## 🔄 Workflow Components

### 1. VideoTranscriber
- Converts video files to word-level timestamped transcripts
- Supports various video formats
- Generates JSON output with precise timing data

### 2. BRollAnalyzer
- Analyzes video transcripts using AI
- Generates contextual B-roll prompts with timing information
- Supports configurable segment duration and limits
- Creates both image and video prompts

### 3. ImageGenerator
- Uses FAL AI's model for high-quality image generation
- Configurable resolution, aspect ratio, and quality settings
- Content policy compliance and safety filtering
- Automatic image downloading and organization

### 4. KlingImageToVideoGenerator
- Converts static images to dynamic videos using Kling AI
- Configurable duration, FPS, and resolution
- Supports custom video prompts for motion control
- High-quality video output with natural movements

### 5. UnifiedWorkflow
- Orchestrates the complete 5-stage pipeline
- Flexible execution with configurable entry points
- Stage skipping capabilities for partial workflows
- Comprehensive error handling and recovery
- Detailed progress tracking and logging
- Automatic file organization and cleanup
- Performance metrics and success rate reporting

### 6. Video Editing Integration
- Seamless B-roll insertion into main video
- Timing-based placement using transcript data
- Cloudinary-powered video processing
- Professional output quality

## 🚨 Error Handling

The project includes robust error handling:
- API failure recovery
- File system error management
- Detailed logging for debugging

## 📝 Logging

Comprehensive logging system provides:
- Step-by-step progress tracking
- Error details and stack traces
- Performance metrics
- File operation status

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 🆘 Support

For issues and questions:
1. Review the configuration options
2. Enable detailed logging for debugging
3. Use mock mode for testing without API costs

---

**Note**: This project requires API keys for multiple AI services:
- **OpenAI**: For intelligent B-roll prompt generation and transcript analysis
- **FAL AI**: For high-quality image generation and converting static images to dynamic videos
- **Cloudinary**: For professional video editing and B-roll integration

Ensure you have valid credentials for all services before running in production mode. The project supports mock mode for testing without API costs.
