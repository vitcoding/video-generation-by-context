#!/usr/bin/env python3
"""
Mock API for testing without real API calls

This module provides mock implementations of API clients
for testing the video generation pipeline without making real API calls.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import configuration
sys.path.append(str(Path(__file__).parent))
from config import config
from constants import (
    DEFAULT_IMAGES_OUTPUT_DIR,
    DEFAULT_SEED,
    DEFAULT_VIDEOS_OUTPUT_DIR,
    VIDEO_DURATION,
    VIDEO_FPS,
    VIDEO_RESOLUTION,
)


class MockFALClient:
    """Mock fal.ai client for testing"""

    def __init__(self):
        self.api_key = "mock_fal_key"

    def upload(self, data: bytes, content_type: str) -> str:
        """Mock image upload"""
        print(
            f"🔧 [MOCK] Uploading image (size: {len(data)} bytes, type: {content_type})"
        )
        return "https://mock-fal-ai.com/uploaded-image.png"

    def subscribe(
        self, model_endpoint: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mock model subscription (used for video and image generation)"""
        print(f"🔧 [MOCK] Subscribing to model: {model_endpoint}")
        print(f"🔧 [MOCK] Arguments: {arguments}")

        # Simulate processing time
        time.sleep(1)

        # Check if this is an image generation request
        if "imagen4" in model_endpoint.lower() or (
            "prompt" in arguments and "image_url" not in arguments
        ):
            # Mock image generation response
            return {
                "images": [
                    {
                        "url": "https://mock-fal-ai.com/generated-image.png",
                        "width": 1024,
                        "height": 1024,
                        "seed": arguments.get("seed", DEFAULT_SEED),
                    }
                ],
                "status": "completed",
                "model_endpoint": model_endpoint,
                "prompt": arguments.get("prompt", ""),
                "aspect_ratio": arguments.get("aspect_ratio", "1:1"),
            }
        elif "kling" in model_endpoint.lower() or "image_url" in arguments:
            # Mock image-to-video generation response
            return {
                "video": {
                    "url": "https://mock-fal-ai.com/generated-video-from-image.mp4",
                    "duration": VIDEO_DURATION,
                    "fps": VIDEO_FPS,
                    "resolution": VIDEO_RESOLUTION,
                },
                "status": "completed",
                "model_endpoint": model_endpoint,
                "prompt": arguments.get("prompt", ""),
                "aspect_ratio": arguments.get("aspect_ratio", "16:9"),
                "image_url": arguments.get("image_url", ""),
            }
        else:
            # Mock regular video generation response
            return {
                "video": {
                    "url": "https://mock-fal-ai.com/generated-video.mp4",
                    "duration": VIDEO_DURATION,
                    "fps": VIDEO_FPS,
                    "resolution": VIDEO_RESOLUTION,
                },
                "status": "completed",
                "model_endpoint": model_endpoint,
            }

    def run(self, model_endpoint: str, **kwargs) -> Dict[str, Any]:
        """Mock model execution"""
        print(f"🔧 [MOCK] Running model: {model_endpoint}")
        print(f"🔧 [MOCK] Parameters: {kwargs}")

        # Simulate processing time
        time.sleep(1)

        return {
            "video": "https://mock-fal-ai.com/generated-video.mp4",
            "status": "completed",
            "duration": VIDEO_DURATION,
            "fps": VIDEO_FPS,
            "resolution": VIDEO_RESOLUTION,
        }


class MockImageGenerator:
    """Mock image generator for testing broll_image_generation"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize mock image generator"""
        self.api_key = api_key or "mock_fal_key"
        self.model_endpoint = "fal-ai/imagen4/preview/fast"
        self.output_dir = Path(DEFAULT_IMAGES_OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)
        print("🔧 [MOCK] Initialized MockImageGenerator")

    def generate_image(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        seed: Optional[int] = None,
    ) -> Optional[Dict]:
        """Mock image generation"""
        print(f"🔧 [MOCK] Generating image...")
        print(f"🔧 [MOCK] Prompt: {prompt}")
        print(f"🔧 [MOCK] Aspect Ratio: {aspect_ratio}")
        print(f"🔧 [MOCK] Seed: {seed}")

        # Simulate processing time
        time.sleep(1)

        # Generate mock image URL based on prompt
        mock_image_url = f"https://mock-fal-ai.com/generated-image-{hash(prompt) % 1000}.png"

        return {
            "image_url": mock_image_url,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "seed": seed,
            "fal_result": {
                "images": [
                    {
                        "url": mock_image_url,
                        "width": 1024,
                        "height": 1024,
                        "seed": seed or DEFAULT_SEED,
                    }
                ],
                "status": "completed",
                "model_endpoint": self.model_endpoint,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
            },
        }

    def download_image(self, image_url: str, filename: str) -> bool:
        """Mock image download"""
        print(f"🔧 [MOCK] Downloading image: {filename}")
        print(f"🔧 [MOCK] URL: {image_url}")

        # Simulate download time
        time.sleep(0.5)

        # Create mock image file
        file_path = self.output_dir / filename
        file_path.parent.mkdir(exist_ok=True)

        # Create a small mock image file (just for testing)
        with open(file_path, "wb") as f:
            f.write(b"mock_image_data_for_testing")

        print(f"🔧 [MOCK] Image downloaded: {file_path}")
        return True

    def save_generation_report(self, image_result: Dict):
        """Mock report saving"""
        report = {
            "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "api_used": "fal.ai Imagen4 (MOCK)",
            "generation_type": "text-to-image",
            "image_result": image_result,
            "success": image_result.get("download_success", False),
            "mock": True,
        }

        report_file = self.output_dir / "imagen4_image_report.json"
        report_file.parent.mkdir(exist_ok=True)

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"🔧 [MOCK] Report saved to: {report_file}")


class MockKlingImageToVideoGenerator:
    """Mock Kling Image-to-Video generator for testing kling_image_to_video"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize mock Kling image-to-video generator"""
        self.api_key = api_key or "mock_fal_key"
        self.model_endpoint = "fal-ai/kling-video/v1.6/pro/image-to-video"
        self.output_dir = Path(DEFAULT_VIDEOS_OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)
        print("🔧 [MOCK] Initialized MockKlingImageToVideoGenerator")

    def upload_image(self, image_path: str) -> str:
        """Mock image upload"""
        print(f"🔧 [MOCK] Uploading image: {image_path}")

        # Simulate upload time
        time.sleep(0.5)

        # Generate mock image URL based on filename
        mock_image_url = f"https://mock-fal-ai.com/uploaded-image-{hash(image_path) % 1000}.png"

        print(f"🔧 [MOCK] Image uploaded: {mock_image_url}")
        return mock_image_url

    def generate_video_from_image(
        self,
        image_path: str,
        prompt: str,
        aspect_ratio: str = "16:9",
        fps: int = VIDEO_FPS,
        resolution: str = VIDEO_RESOLUTION,
    ) -> Optional[Dict]:
        """Mock video generation from image"""
        print(f"🔧 [MOCK] Generating video from image...")
        print(f"🔧 [MOCK] Image path: {image_path}")
        print(f"🔧 [MOCK] Prompt: {prompt}")
        print(f"🔧 [MOCK] Aspect Ratio: {aspect_ratio}")
        print(f"🔧 [MOCK] FPS: {fps}")
        print(f"🔧 [MOCK] Resolution: {resolution}")

        # Simulate processing time
        time.sleep(1)

        # Upload image first
        image_url = self.upload_image(image_path)

        # Generate mock video URL
        mock_video_url = f"https://mock-fal-ai.com/generated-video-from-image-{hash(prompt) % 1000}.mp4"

        return {
            "video_url": mock_video_url,
            "image_path": image_path,
            "image_url": image_url,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "fps": fps,
            "resolution": resolution,
            "duration": "5s",
            "fal_result": {
                "video": {
                    "url": mock_video_url,
                    "duration": VIDEO_DURATION,
                    "fps": fps,
                    "resolution": resolution,
                },
                "status": "completed",
                "model_endpoint": self.model_endpoint,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "image_url": image_url,
            },
        }

    def download_video(self, video_url: str, filename: str) -> bool:
        """Mock video download"""
        print(f"🔧 [MOCK] Downloading video: {filename}")
        print(f"🔧 [MOCK] URL: {video_url}")

        # Simulate download time
        time.sleep(0.5)

        # Create mock video file
        file_path = self.output_dir / filename
        file_path.parent.mkdir(exist_ok=True)

        # Create a small mock video file (just for testing)
        with open(file_path, "wb") as f:
            f.write(b"mock_video_data_for_testing")

        print(f"🔧 [MOCK] Video downloaded: {file_path}")
        return True

    def save_generation_report(self, video_result: Dict):
        """Mock report saving"""
        report = {
            "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "api_used": "fal.ai Kling 1.6 Pro Image-to-Video (MOCK)",
            "generation_type": "image-to-video",
            "video_result": video_result,
            "success": video_result.get("download_success", False),
            "mock": True,
        }

        report_file = self.output_dir / "kling_image_to_video_report.json"
        report_file.parent.mkdir(exist_ok=True)

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"🔧 [MOCK] Report saved to: {report_file}")


class MockOpenAI:
    """Mock OpenAI class that can be called as constructor"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or "mock_openai_key"
        self.chat = MockOpenAIClient()
        self.audio = MockOpenAIAudio()


class MockOpenAIAudio:
    """Mock OpenAI Audio client for transcription"""

    def __init__(self):
        self.transcriptions = MockOpenAITranscriptions()


class MockOpenAITranscriptions:
    """Mock OpenAI Transcriptions API"""

    def create(
        self,
        model: str,
        file: Any,
        response_format: str = "json",
        timestamp_granularities: Optional[List[str]] = None,
    ) -> Dict:
        """Mock transcription creation"""
        print(f"🔧 [MOCK] OpenAI transcription")
        print(f"🔧 [MOCK] Model: {model}")
        print(f"🔧 [MOCK] Response format: {response_format}")
        print(f"🔧 [MOCK] Timestamp granularities: {timestamp_granularities}")

        # Simulate processing time
        time.sleep(1)

        # Mock transcript data
        mock_text = "А вы знали, что худшая ошибка тех, кто боится искусственного интеллекта, думать, что он делает нас глупее? Это совершенно не так. Мы стоим на пороге четвертой революции, она многократно превосходит все, что было раньше в истории человечества."

        # Create mock word-level timestamps
        words = mock_text.split()
        word_timestamps = []
        current_time = 0.0

        for word in words:
            word_duration = (
                len(word) * 0.1 + 0.1
            )  # Approximate duration based on word length
            word_timestamps.append(
                {
                    "word": word,
                    "start": current_time,
                    "end": current_time + word_duration,
                }
            )
            current_time += word_duration + 0.1  # Add pause between words

        # Create mock response object with attributes
        class MockTranscriptResponse:
            def __init__(self):
                self.text = mock_text
                self.language = "ru"
                self.duration = current_time
                self.words = [
                    type(
                        "Word",
                        (),
                        {
                            "word": w["word"],
                            "start": w["start"],
                            "end": w["end"],
                        },
                    )()
                    for w in word_timestamps
                ]

        return MockTranscriptResponse()


class MockOpenAIClient:
    """Mock OpenAI client for testing"""

    def __init__(self):
        self.api_key = "mock_openai_key"

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs) -> Dict[str, Any]:
        """Mock chat completion"""
        print(f"🔧 [MOCK] OpenAI chat completion")
        print(f"🔧 [MOCK] Messages: {kwargs.get('messages', [])}")

        # Generate mock response based on input
        messages = kwargs.get("messages", [])
        system_prompt = ""
        user_prompt = ""

        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            elif msg.get("role") == "user":
                user_prompt = msg.get("content", "")

        # Generate mock response
        mock_response = self._generate_mock_response(
            system_prompt, user_prompt
        )

        # Create a mock response object with choices attribute
        class MockResponse:
            def __init__(self, content):
                self.choices = [MockChoice(content)]

        class MockChoice:
            def __init__(self, content):
                self.message = MockMessage(content)

        class MockMessage:
            def __init__(self, content):
                self.content = content
                self.role = "assistant"

        return MockResponse(mock_response)

    def _generate_mock_response(
        self, system_prompt: str, user_prompt: str
    ) -> str:
        """Generate mock response based on prompts"""
        if "b-roll" in user_prompt.lower() or "видео" in user_prompt.lower():
            return json.dumps(
                {
                    "segments": [
                        {
                            "start_time": 0.0,
                            "end_time": 5.0,
                            "text_context": "Professional business content discussion",
                            "theme_description": "Modern office environment",
                            "importance_score": 8.5,
                            "keywords": ["business", "professional", "office"],
                            "image_prompt": "Professional business content scene 1 with modern office environment, professional lighting, and collaborative atmosphere",
                            "video_prompt": "You are a world-class cinematic director. Create a 16x9 5-second professional business sequence 1. – Opening shot: Modern office environment with professional lighting. – Medium shot: People collaborating and discussing ideas. – Close-up sequence: Focus on technology and innovation. – Camera movement: Smooth dolly movements and subtle tracking shots. – Final shot: Professional success and achievement. Style & tone: Modern, professional, high-quality; blue and white color palette; clean visual effects; ambient business sounds.",
                        },
                        {
                            "start_time": 9.69,
                            "end_time": 14.69,
                            "text_context": "Content creation and innovation",
                            "theme_description": "Creative workspace",
                            "importance_score": 7.8,
                            "keywords": [
                                "innovation",
                                "creativity",
                                "technology",
                            ],
                            "image_prompt": "Professional business content scene 2 with modern office environment, professional lighting, and collaborative atmosphere",
                            "video_prompt": "You are a world-class cinematic director. Create a 16x9 5-second professional business sequence 2. – Opening shot: Modern office environment with professional lighting. – Medium shot: People collaborating and discussing ideas. – Close-up sequence: Focus on technology and innovation. – Camera movement: Smooth dolly movements and subtle tracking shots. – Final shot: Professional success and achievement. Style & tone: Modern, professional, high-quality; blue and white color palette; clean visual effects; ambient business sounds.",
                        },
                        {
                            "start_time": 19.38,
                            "end_time": 24.38,
                            "text_context": "Future of digital content",
                            "theme_description": "Digital transformation",
                            "importance_score": 9.2,
                            "keywords": [
                                "digital",
                                "future",
                                "transformation",
                            ],
                            "image_prompt": "Professional business content scene 3 with modern office environment, professional lighting, and collaborative atmosphere",
                            "video_prompt": "You are a world-class cinematic director. Create a 16x9 5-second professional business sequence 3. – Opening shot: Modern office environment with professional lighting. – Medium shot: People collaborating and discussing ideas. – Close-up sequence: Focus on technology and innovation. – Camera movement: Smooth dolly movements and subtle tracking shots. – Final shot: Professional success and achievement. Style & tone: Modern, professional, high-quality; blue and white color palette; clean visual effects; ambient business sounds.",
                        },
                    ]
                },
                ensure_ascii=False,
                indent=2,
            )

        return "Mock response based on your request"


class MockRequests:
    """Mock requests module for testing"""

    @staticmethod
    def get(url: str, **kwargs) -> "MockResponse":
        """Mock GET request"""
        print(f"🔧 [MOCK] GET request to: {url}")
        return MockResponse(200, {"Content-Type": "application/json"})

    @staticmethod
    def post(url: str, **kwargs) -> "MockResponse":
        """Mock POST request"""
        print(f"🔧 [MOCK] POST request to: {url}")
        return MockResponse(200, {"Content-Type": "application/json"})


class MockResponse:
    """Mock response object"""

    def __init__(self, status_code: int, headers: Dict[str, str]):
        self.status_code = status_code
        self.headers = headers
        self._content = b'{"status": "success", "mock": true}'
        # Mock video content (small fake video data)
        self._video_content = b"fake_video_content_for_testing"

    @property
    def content(self) -> bytes:
        return self._content

    def json(self) -> Dict[str, Any]:
        return {"status": "success", "mock": True}

    def iter_content(self, chunk_size: int = 8192):
        """Mock streaming content for video download"""
        content = self._video_content
        for i in range(0, len(content), chunk_size):
            yield content[i : i + chunk_size]

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"Mock HTTP error: {self.status_code}")


# Mock modules
mock_fal_client = MockFALClient()
mock_openai_client = MockOpenAI
mock_requests = MockRequests()
mock_image_generator = MockImageGenerator
mock_kling_image_to_video_generator = MockKlingImageToVideoGenerator
