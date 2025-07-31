#!/usr/bin/env python3
"""
Tests for workflow logic using mock API

This module contains comprehensive tests for the UnifiedWorkflow class
using mock API implementations to avoid real API calls during testing.
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from mock_api import (
    MockImageGenerator,
    MockKlingImageToVideoGenerator,
    MockOpenAIClient,
)
from workflow import UnifiedWorkflow


class TestUnifiedWorkflow(unittest.TestCase):
    """Test cases for UnifiedWorkflow class"""

    def setUp(self):
        """Set up test environment"""
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir) / "data" / "video_generation"
        self.test_transcript_dir = self.test_data_dir / "audio_transcript"
        self.test_prompts_dir = self.test_data_dir / "broll_prompts"
        self.test_images_dir = Path(self.temp_dir) / "images"
        self.test_videos_dir = Path(self.temp_dir) / "videos"

        # Create test directories
        for directory in [
            self.test_transcript_dir,
            self.test_prompts_dir,
            self.test_images_dir,
            self.test_videos_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        # Create sample transcript data
        self.sample_transcript = {
            "transcript": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "Welcome to our video about artificial intelligence.",
                },
                {
                    "start": 5.0,
                    "end": 10.0,
                    "text": "AI is transforming the world around us.",
                },
                {
                    "start": 10.0,
                    "end": 15.0,
                    "text": "Machine learning algorithms are becoming more sophisticated.",
                },
            ],
            "metadata": {
                "duration": 15.0,
                "language": "en",
                "speaker_count": 1,
            },
        }

        # Save sample transcript
        self.transcript_file = (
            self.test_transcript_dir / "test_transcript.json"
        )
        with open(self.transcript_file, "w", encoding="utf-8") as f:
            json.dump(self.sample_transcript, f, ensure_ascii=False, indent=2)

    def tearDown(self):
        """Clean up test environment"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("workflow.BRollAnalyzer")
    @patch("workflow.ImageGenerator")
    @patch("workflow.KlingImageToVideoGenerator")
    def test_workflow_initialization(
        self, mock_video_gen, mock_img_gen, mock_analyzer
    ):
        """Test workflow initialization with mock components"""
        # Mock the components
        mock_analyzer.return_value = Mock()
        mock_img_gen.return_value = Mock()
        mock_video_gen.return_value = Mock()

        # Create workflow instance
        workflow = UnifiedWorkflow(max_segments=3)

        # Verify components are initialized
        self.assertEqual(workflow.max_segments, 3)
        self.assertIsNotNone(workflow.broll_analyzer)
        self.assertIsNotNone(workflow.image_generator)
        self.assertIsNotNone(workflow.video_generator)

        # Verify directories are created
        self.assertTrue(workflow.transcript_dir.exists())
        self.assertTrue(workflow.prompts_dir.exists())
        self.assertTrue(workflow.images_dir.exists())
        self.assertTrue(workflow.videos_dir.exists())

    @patch("workflow.BRollAnalyzer")
    @patch("workflow.ImageGenerator")
    @patch("workflow.KlingImageToVideoGenerator")
    def test_process_transcript_to_prompts(
        self, mock_video_gen, mock_img_gen, mock_analyzer
    ):
        """Test transcript processing to prompts generation"""
        # Mock analyzer responses
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.load_transcript.return_value = (
            self.sample_transcript
        )

        mock_segments = [
            {
                "start_time": 0.0,
                "end_time": 5.0,
                "text_context": "Welcome to our video about artificial intelligence.",
                "image_prompt": "A modern computer screen displaying AI algorithms",
                "video_prompt": "Smooth camera movement over AI visualization",
            },
            {
                "start_time": 5.0,
                "end_time": 10.0,
                "text_context": "AI is transforming the world around us.",
                "image_prompt": "Futuristic city with AI technology",
                "video_prompt": "Dynamic cityscape with technological elements",
            },
        ]

        mock_analyzer_instance.analyze_transcript_with_ai.return_value = (
            mock_segments
        )
        mock_analyzer_instance.generate_output_json.return_value = {
            "broll_segments": mock_segments,
            "transcript_info": self.sample_transcript["metadata"],
        }

        mock_analyzer.return_value = mock_analyzer_instance
        mock_img_gen.return_value = Mock()
        mock_video_gen.return_value = Mock()

        # Create workflow and test
        workflow = UnifiedWorkflow(max_segments=2)
        result = workflow.process_transcript_to_prompts(
            str(self.transcript_file)
        )

        # Verify results
        self.assertIn("broll_segments", result)
        self.assertEqual(len(result["broll_segments"]), 2)
        self.assertIn("transcript_info", result)

        # Verify prompts file was created
        prompts_file = workflow.prompts_dir / "workflow_generated_prompts.json"
        self.assertTrue(prompts_file.exists())

    @patch("workflow.BRollAnalyzer")
    @patch("workflow.ImageGenerator")
    @patch("workflow.KlingImageToVideoGenerator")
    def test_generate_images_from_prompts(
        self, mock_video_gen, mock_img_gen, mock_analyzer
    ):
        """Test image generation from prompts"""
        # Mock image generator
        mock_img_instance = Mock()
        mock_img_instance.generate_image.return_value = {
            "image_url": "https://mock-api.com/test-image.png",
            "width": 1024,
            "height": 1024,
            "seed": 42,
        }
        mock_img_instance.download_image.return_value = True

        mock_img_gen.return_value = mock_img_instance
        mock_analyzer.return_value = Mock()
        mock_video_gen.return_value = Mock()

        # Test data
        prompts_data = {
            "broll_segments": [
                {
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "image_prompt": "Test image prompt 1",
                },
                {
                    "start_time": 5.0,
                    "end_time": 10.0,
                    "image_prompt": "Test image prompt 2",
                },
            ]
        }

        # Create workflow and test
        workflow = UnifiedWorkflow(max_segments=2)
        result = workflow.generate_images_from_prompts(prompts_data)

        # Verify results
        self.assertEqual(len(result), 2)
        for img_result in result:
            self.assertIn("image_url", img_result)
            self.assertIn("download_success", img_result)
            self.assertTrue(img_result["download_success"])

    @patch("workflow.BRollAnalyzer")
    @patch("workflow.ImageGenerator")
    @patch("workflow.KlingImageToVideoGenerator")
    def test_create_videos_from_images(
        self, mock_video_gen, mock_img_gen, mock_analyzer
    ):
        """Test video creation from images"""
        # Mock video generator
        mock_video_instance = Mock()
        mock_video_instance.generate_video_from_image.return_value = {
            "video_url": "https://mock-api.com/test-video.mp4",
            "duration": 5.0,
            "fps": 30,
            "resolution": "1920x1080",
        }
        mock_video_instance.download_video.return_value = True

        mock_video_gen.return_value = mock_video_instance
        mock_img_gen.return_value = Mock()
        mock_analyzer.return_value = Mock()

        # Create test image file
        test_image_path = self.test_images_dir / "test_image.png"
        with open(test_image_path, "w") as f:
            f.write("mock image content")

        # Test data
        image_results = [
            {
                "local_filename": "test_image.png",
                "image_url": "https://mock-api.com/test-image.png",
                "segment_info": {
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "video_prompt": "Test video prompt",
                },
            }
        ]

        # Create workflow and test
        workflow = UnifiedWorkflow(max_segments=1)
        workflow.images_dir = self.test_images_dir
        workflow.videos_dir = self.test_videos_dir

        result = workflow.create_videos_from_images(image_results)

        # Verify results
        self.assertEqual(len(result), 1)
        for video_result in result:
            self.assertIn("video_url", video_result)
            self.assertIn("download_success", video_result)
            self.assertTrue(video_result["download_success"])

    @patch("workflow.BRollAnalyzer")
    @patch("workflow.ImageGenerator")
    @patch("workflow.KlingImageToVideoGenerator")
    def test_complete_workflow_integration(
        self, mock_video_gen, mock_img_gen, mock_analyzer
    ):
        """Test complete workflow integration"""
        # Mock all components
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.load_transcript.return_value = (
            self.sample_transcript
        )

        mock_segments = [
            {
                "start_time": 0.0,
                "end_time": 5.0,
                "text_context": "Test context",
                "image_prompt": "Test image prompt",
                "video_prompt": "Test video prompt",
            },
        ]

        mock_analyzer_instance.analyze_transcript_with_ai.return_value = (
            mock_segments
        )
        mock_analyzer_instance.generate_output_json.return_value = {
            "broll_segments": mock_segments,
            "transcript_info": self.sample_transcript["metadata"],
        }

        mock_img_instance = Mock()
        mock_img_instance.generate_image.return_value = {
            "image_url": "https://mock-api.com/test-image.png",
            "width": 1024,
            "height": 1024,
        }
        mock_img_instance.download_image.return_value = True

        mock_video_instance = Mock()
        mock_video_instance.generate_video_from_image.return_value = {
            "video_url": "https://mock-api.com/test-video.mp4",
            "duration": 5.0,
        }
        mock_video_instance.download_video.return_value = True

        mock_analyzer.return_value = mock_analyzer_instance
        mock_img_gen.return_value = mock_img_instance
        mock_video_gen.return_value = mock_video_instance

        # Create workflow and run complete workflow
        workflow = UnifiedWorkflow(max_segments=1)
        workflow.data_dir = self.test_data_dir
        workflow.images_dir = self.test_images_dir
        workflow.videos_dir = self.test_videos_dir

        result = workflow.run_complete_workflow(str(self.transcript_file))

        # Verify complete workflow results
        self.assertIn("workflow_info", result)
        self.assertIn("prompts_data", result)
        self.assertIn("image_results", result)
        self.assertIn("video_results", result)
        self.assertIn("summary", result)

        # Verify summary statistics
        summary = result["summary"]
        self.assertEqual(summary["total_segments"], 1)
        self.assertEqual(summary["images_generated"], 1)
        self.assertEqual(summary["videos_created"], 1)

        # Verify report file was created
        report_file = workflow.data_dir / "workflow_complete_report.json"
        self.assertTrue(report_file.exists())

    def test_workflow_with_mock_api_components(self):
        """Test workflow using actual mock API components"""
        # Create workflow with mock components
        workflow = UnifiedWorkflow(max_segments=2)

        # Replace components with mock implementations
        workflow.image_generator = MockImageGenerator()
        workflow.video_generator = MockKlingImageToVideoGenerator()

        # Mock the analyzer to return test data
        workflow.broll_analyzer = Mock()
        workflow.broll_analyzer.load_transcript.return_value = (
            self.sample_transcript
        )

        mock_segments = [
            {
                "start_time": 0.0,
                "end_time": 5.0,
                "text_context": "Test context 1",
                "image_prompt": "Test image prompt 1",
                "video_prompt": "Test video prompt 1",
            },
            {
                "start_time": 5.0,
                "end_time": 10.0,
                "text_context": "Test context 2",
                "image_prompt": "Test image prompt 2",
                "video_prompt": "Test video prompt 2",
            },
        ]

        workflow.broll_analyzer.analyze_transcript_with_ai.return_value = (
            mock_segments
        )
        workflow.broll_analyzer.generate_output_json.return_value = {
            "broll_segments": mock_segments,
            "transcript_info": self.sample_transcript["metadata"],
        }

        # Update directories to use test paths
        workflow.data_dir = self.test_data_dir
        workflow.images_dir = self.test_images_dir
        workflow.videos_dir = self.test_videos_dir

        # Run complete workflow
        result = workflow.run_complete_workflow(str(self.transcript_file))

        # Verify results
        self.assertIn("workflow_info", result)
        self.assertIn("prompts_data", result)
        self.assertIn("image_results", result)
        self.assertIn("video_results", result)

        # Verify that mock API calls were made
        self.assertEqual(len(result["image_results"]), 2)
        self.assertEqual(len(result["video_results"]), 2)

    def test_workflow_error_handling(self):
        """Test workflow error handling"""
        # Create workflow
        workflow = UnifiedWorkflow(max_segments=1)

        # Test with non-existent transcript file
        with self.assertRaises(Exception):
            workflow.process_transcript_to_prompts("non_existent_file.json")

    def test_workflow_with_different_segment_counts(self):
        """Test workflow with different segment counts"""
        segment_counts = [1, 3, 5]

        for count in segment_counts:
            with self.subTest(segment_count=count):
                workflow = UnifiedWorkflow(max_segments=count)
                self.assertEqual(workflow.max_segments, count)

    @patch("workflow.BRollAnalyzer")
    @patch("workflow.ImageGenerator")
    @patch("workflow.KlingImageToVideoGenerator")
    def test_workflow_summary_printing(
        self, mock_video_gen, mock_img_gen, mock_analyzer
    ):
        """Test workflow summary printing"""
        # Mock components
        mock_analyzer.return_value = Mock()
        mock_img_gen.return_value = Mock()
        mock_video_gen.return_value = Mock()

        # Create workflow
        workflow = UnifiedWorkflow(max_segments=2)

        # Test data
        test_results = {
            "workflow_info": {
                "total_duration": 10.5,
                "max_segments": 2,
            },
            "summary": {
                "total_segments": 2,
                "images_generated": 2,
                "videos_created": 2,
                "success_rate": {
                    "images": "2/2",
                    "videos": "2/2",
                },
            },
        }

        # Test summary printing (should not raise exceptions)
        try:
            workflow.print_workflow_summary(test_results)
        except Exception as e:
            self.fail(f"print_workflow_summary raised an exception: {e}")


class TestWorkflowWithMockAPI(unittest.TestCase):
    """Test cases using actual mock API implementations"""

    def setUp(self):
        """Set up test environment with mock API"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_dir = Path(self.temp_dir)

        # Create test directories
        (self.test_dir / "images").mkdir()
        (self.test_dir / "videos").mkdir()

    def tearDown(self):
        """Clean up test environment"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_mock_image_generation(self):
        """Test mock image generation"""
        mock_generator = MockImageGenerator()

        # Test image generation
        result = mock_generator.generate_image(
            prompt="Test prompt", aspect_ratio="16:9", seed=42
        )

        self.assertIsNotNone(result)
        self.assertIn("image_url", result)
        self.assertIn("fal_result", result)
        self.assertIn("images", result["fal_result"])
        self.assertIn("width", result["fal_result"]["images"][0])
        self.assertIn("height", result["fal_result"]["images"][0])

        # Test image download
        success = mock_generator.download_image(
            result["image_url"], "test_image.png"
        )

        self.assertTrue(success)

    def test_mock_video_generation(self):
        """Test mock video generation"""
        mock_generator = MockKlingImageToVideoGenerator()

        # Create test image file
        test_image_path = self.test_dir / "images" / "test_image.png"
        with open(test_image_path, "w") as f:
            f.write("mock image content")

        # Test video generation
        result = mock_generator.generate_video_from_image(
            image_path=str(test_image_path),
            prompt="Test video prompt",
            aspect_ratio="16:9",
        )

        self.assertIsNotNone(result)
        self.assertIn("video_url", result)
        self.assertIn("duration", result)

        # Test video download
        success = mock_generator.download_video(
            result["video_url"], "test_video.mp4"
        )

        self.assertTrue(success)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
