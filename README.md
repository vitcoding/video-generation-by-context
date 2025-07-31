# Video Generation by Context

A comprehensive Python project for generating B-roll videos from audio transcripts using AI-powered image and video generation.

## 🎯 Project Overview

This project implements a complete pipeline for converting audio transcripts into B-roll video content. It analyzes transcripts, generates contextual prompts, creates images, and converts them into videos using advanced AI models.

## 🚀 Features

- **Transcript Analysis**: Converts audio transcripts into contextual B-roll prompts
- **AI Image Generation**: Creates high-quality images using FAL AI's Imagen4 model
- **Video Generation**: Converts images to videos using Kling AI technology
- **Unified Workflow**: Complete end-to-end pipeline processing
- **Mock Mode**: Testing capabilities with mock API responses
- **File Archiving**: Automatic organization and cleanup of generated files
- **Configurable Parameters**: Customizable video resolution, duration, and quality settings

## 📁 Project Structure

```
video-generation-by-context/
├── main.py                 # Main entry point and workflow orchestration
├── requirements.txt        # Python dependencies
├── b_roll/                # Core project module
│   ├── b_roll_generation/ # AI generation components
│   │   ├── broll_prompts.py      # Transcript to prompt conversion
│   │   ├── broll_image_generation.py # Image generation logic
│   │   ├── kling_image_to_video.py   # Video generation from images
│   │   └── prompts.py             # Prompt templates and utilities
│   ├── workflow.py        # Unified workflow orchestration
│   ├── config.py          # Configuration management
│   ├── constants.py       # Project constants and settings
│   ├── file_archiver.py   # File organization and cleanup
│   ├── logger_config.py   # Logging configuration
│   ├── mock_api.py        # Mock API for testing
│   ├── data/              # Production data directory
│   ├── data_mock/         # Mock data for testing
│   └── test_*.py          # Test files
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
   FAL_API_KEY=your_fal_key_here
   ```

## 🎮 Usage

### Basic Usage

Run the complete workflow:
```bash
python main.py
```

### Configuration Options

The project supports several configuration options:

- **API Mode**: Toggle between real API calls and mock mode
- **Video Resolution**: Configure output video quality (1280x720, 1920x1080, etc.)
- **Video Duration**: Set video length in seconds
- **Max Segments**: Control number of video segments generated
- **File Archiving**: Enable/disable automatic file organization

### Workflow Steps

1. **Transcript Processing**: Analyzes audio transcript and generates B-roll prompts
2. **Image Generation**: Creates images from prompts using AI models
3. **Video Creation**: Converts images to videos with specified parameters
4. **File Archiving**: Organizes and optionally cleans up generated files

## 🔧 Configuration

### Environment Variables

- `API_REQUEST`: Enable/disable real API calls (true/false)
- `OPENAI_API_KEY`: OpenAI API key for prompt generation
- `FAL_API_KEY`: FAL AI API key for image generation

### Constants (constants.py)

Key configurable parameters:
- `VIDEO_RESOLUTION`: Output video resolution
- `VIDEO_DURATION`: Video length in seconds
- `VIDEO_FPS`: Frames per second
- `IMAGE_ASPECT_RATIO`: Image aspect ratio
- `ENABLE_VIDEO_GENERATION`: Toggle video generation

## 🧪 Testing

The project includes comprehensive testing capabilities:

- **Mock Mode**: Test without API calls using `API_REQUEST=false`
- **Integration Tests**: Test complete workflow
- **Unit Tests**: Test individual components

Run tests:
```bash
python -m pytest b_roll/test_*.py
```

## 📊 Output Structure

Generated files are organized in the following structure:
```
b_roll/data/video_generation/
├── audio_transcript/      # Input transcript files
├── broll_prompts/         # Generated B-roll prompts
├── images_input/          # Generated images
└── videos_output/         # Final video files
```

## 🔄 Workflow Components

### 1. BRollAnalyzer
- Analyzes audio transcripts
- Generates contextual B-roll prompts
- Supports multiple segments

### 2. ImageGenerator
- Uses FAL AI's Imagen4 model
- Configurable resolution and quality
- Content policy compliance

### 3. KlingImageToVideoGenerator
- Converts images to videos
- Configurable duration and FPS
- High-quality video output

### 4. UnifiedWorkflow
- Orchestrates complete pipeline
- Handles error recovery
- Provides detailed logging

## 🚨 Error Handling

The project includes robust error handling:
- API failure recovery
- File system error management
- Detailed logging for debugging
- Graceful degradation in mock mode

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
1. Check the test files for usage examples
2. Review the configuration options
3. Enable detailed logging for debugging
4. Use mock mode for testing without API costs

---

**Note**: This project requires API keys for OpenAI and FAL AI services. Ensure you have valid credentials before running in production mode.
