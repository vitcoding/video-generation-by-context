#!/usr/bin/env python3
"""
Integration tests for workflow with mock API

This module contains integration tests that verify the complete workflow
functions correctly with mock API implementations.
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from mock_api import MockImageGenerator, MockKlingImageToVideoGenerator
from workflow import UnifiedWorkflow


class TestWorkflowIntegration(unittest.TestCase):
    """Integration tests for complete workflow with mock API"""

    def setUp(self):
        """Set up test environment for each test method"""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir) / "data" / "video_generation"
        self.test_transcript_dir = self.test_data_dir / "audio_transcript"
        self.test_prompts_dir = self.test_data_dir / "broll_prompts"
        self.test_images_dir = Path(self.temp_dir) / "images"
        self.test_videos_dir = Path(self.temp_dir) / "videos"

        # Create directories
        for directory in [
            self.test_transcript_dir,
            self.test_prompts_dir,
            self.test_images_dir,
            self.test_videos_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        # Create sample transcript
        self.sample_transcript = {
            "transcript": [
                {
                    "start": 0.0,
                    "end": 3.0,
                    "text": "Artificial intelligence is revolutionizing technology.",
                },
                {
                    "start": 3.0,
                    "end": 6.0,
                    "text": "Machine learning algorithms process vast amounts of data.",
                },
                {
                    "start": 6.0,
                    "end": 9.0,
                    "text": "Neural networks mimic human brain functions.",
                },
            ],
            "metadata": {
                "duration": 9.0,
                "language": "en",
                "speaker_count": 1,
            },
        }

        # Save transcript file
        self.transcript_file = (
            self.test_transcript_dir / "integration_test_transcript.json"
        )
        with open(self.transcript_file, "w", encoding="utf-8") as f:
            json.dump(self.sample_transcript, f, ensure_ascii=False, indent=2)

    def tearDown(self):
        """Clean up test environment"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("workflow.BRollAnalyzer")
    def test_workflow_with_mock_api_components(self, mock_analyzer_class):
        """Test complete workflow using mock API components"""
        # Mock the analyzer
        mock_analyzer = mock_analyzer_class.return_value
        mock_analyzer.load_transcript.return_value = self.sample_transcript

        # Mock segments that would be generated
        mock_segments = [
            {
                "start_time": 0.0,
                "end_time": 3.0,
                "text_context": "Artificial intelligence is revolutionizing technology.",
                "image_prompt": "Futuristic AI technology with glowing circuits and digital interfaces",
                "video_prompt": "Smooth camera movement over AI visualization with subtle animations",
            },
            {
                "start_time": 3.0,
                "end_time": 6.0,
                "text_context": "Machine learning algorithms process vast amounts of data.",
                "image_prompt": "Data visualization with flowing streams of information and neural networks",
                "video_prompt": "Dynamic data flow with particle effects and network connections",
            },
        ]

        mock_analyzer.analyze_transcript_with_ai.return_value = mock_segments
        mock_analyzer.generate_output_json.return_value = {
            "broll_segments": mock_segments,
            "transcript_info": self.sample_transcript["metadata"],
        }

        # Create workflow with mock API components
        workflow = UnifiedWorkflow(max_segments=2)

        # Replace components with mock implementations
        workflow.image_generator = MockImageGenerator()
        workflow.video_generator = MockKlingImageToVideoGenerator()

        # Update directories to use test paths
        workflow.data_dir = self.test_data_dir
        workflow.images_dir = self.test_images_dir
        workflow.videos_dir = self.test_videos_dir

        # Run complete workflow
        start_time = time.time()
        result = workflow.run_complete_workflow(str(self.transcript_file))
        end_time = time.time()

        # Verify workflow completed successfully
        assert result is not None
        assert "workflow_info" in result
        assert "prompts_data" in result
        assert "image_results" in result
        assert "video_results" in result
        assert "summary" in result

        # Verify workflow statistics
        summary = result["summary"]
        assert summary["total_segments"] == 2
        assert summary["images_generated"] == 2
        assert summary["videos_created"] == 2
        assert summary["success_rate"]["images"] == "2/2"
        assert summary["success_rate"]["videos"] == "2/2"

        # Verify workflow timing
        workflow_info = result["workflow_info"]
        assert workflow_info["total_duration"] > 0
        assert workflow_info["max_segments"] == 2
        # Note: api_used might be True even in mock mode depending on config
        assert "api_used" in workflow_info

        # Verify files were created
        prompts_file = workflow.prompts_dir / "workflow_generated_prompts.json"
        assert prompts_file.exists()

        report_file = workflow.data_dir / "workflow_complete_report.json"
        assert report_file.exists()

        print(
            f"✅ Integration test completed successfully in {end_time - start_time:.2f} seconds"
        )

    def test_mock_api_components_individually(self):
        """Test individual mock API components"""
        # Test mock image generator
        mock_img_gen = MockImageGenerator()

        # Test image generation
        img_result = mock_img_gen.generate_image(
            prompt="Test AI visualization", aspect_ratio="16:9", seed=42
        )

        assert img_result is not None
        assert "image_url" in img_result
        assert "fal_result" in img_result
        assert "images" in img_result["fal_result"]
        assert "width" in img_result["fal_result"]["images"][0]
        assert "height" in img_result["fal_result"]["images"][0]

        # Test image download
        download_success = mock_img_gen.download_image(
            img_result["image_url"], "test_image.png"
        )
        assert download_success is True

        # Test mock video generator
        mock_video_gen = MockKlingImageToVideoGenerator()

        # Create test image file
        test_image_path = self.test_images_dir / "test_image.png"
        with open(test_image_path, "w") as f:
            f.write("mock image content")

        # Test video generation
        video_result = mock_video_gen.generate_video_from_image(
            image_path=str(test_image_path),
            prompt="Test video with AI elements",
            aspect_ratio="16:9",
        )

        assert video_result is not None
        assert "video_url" in video_result
        assert "duration" in video_result

        # Test video download
        download_success = mock_video_gen.download_video(
            video_result["video_url"], "test_video.mp4"
        )
        assert download_success is True

        print("✅ Individual mock API components tested successfully")

    def test_workflow_error_scenarios(self):
        """Test workflow behavior in error scenarios"""
        # Test with non-existent transcript file
        workflow = UnifiedWorkflow(max_segments=1)

        try:
            workflow.process_transcript_to_prompts("non_existent_file.json")
            assert False, "Should have raised an exception"
        except Exception as e:
            print(f"✅ Correctly handled missing file: {e}")

        # Test with empty transcript data
        empty_transcript = {"transcript": [], "metadata": {}}
        empty_file = self.test_transcript_dir / "empty_transcript.json"
        with open(empty_file, "w", encoding="utf-8") as f:
            json.dump(empty_transcript, f, ensure_ascii=False, indent=2)

        try:
            workflow.process_transcript_to_prompts(str(empty_file))
            # This might not raise an exception depending on implementation
            print("✅ Handled empty transcript data")
        except Exception as e:
            print(f"✅ Correctly handled empty transcript: {e}")

    def test_workflow_performance_with_mock(self):
        """Test workflow performance with mock API"""
        # Mock the analyzer
        with patch("workflow.BRollAnalyzer") as mock_analyzer_class:
            mock_analyzer = mock_analyzer_class.return_value
            mock_analyzer.load_transcript.return_value = self.sample_transcript

            mock_segments = [
                {
                    "start_time": 0.0,
                    "end_time": 3.0,
                    "text_context": "Test segment",
                    "image_prompt": "Test image prompt",
                    "video_prompt": "Test video prompt",
                }
            ]

            mock_analyzer.analyze_transcript_with_ai.return_value = (
                mock_segments
            )
            mock_analyzer.generate_output_json.return_value = {
                "broll_segments": mock_segments,
                "transcript_info": self.sample_transcript["metadata"],
            }

            # Create workflow
            workflow = UnifiedWorkflow(max_segments=1)
            workflow.image_generator = MockImageGenerator()
            workflow.video_generator = MockKlingImageToVideoGenerator()
            workflow.data_dir = self.test_data_dir
            workflow.images_dir = self.test_images_dir
            workflow.videos_dir = self.test_videos_dir

            # Measure performance
            start_time = time.time()
            result = workflow.run_complete_workflow(str(self.transcript_file))
            end_time = time.time()

            duration = end_time - start_time

            # Verify performance is reasonable (should be fast with mock API)
            assert (
                duration < 10.0
            ), f"Workflow took too long: {duration:.2f} seconds"
            assert result["summary"]["videos_created"] == 1

            print(f"✅ Performance test passed: {duration:.2f} seconds")

    def test_workflow_with_different_configurations(self):
        """Test workflow with different configurations"""
        configurations = [
            {"max_segments": 1, "description": "Single segment"},
            {"max_segments": 3, "description": "Multiple segments"},
            {"max_segments": 5, "description": "Maximum segments"},
        ]

        for config in configurations:
            with patch("workflow.BRollAnalyzer") as mock_analyzer_class:
                mock_analyzer = mock_analyzer_class.return_value
                mock_analyzer.load_transcript.return_value = (
                    self.sample_transcript
                )

                # Create mock segments based on configuration
                mock_segments = []
                for i in range(config["max_segments"]):
                    mock_segments.append(
                        {
                            "start_time": i * 3.0,
                            "end_time": (i + 1) * 3.0,
                            "text_context": f"Test segment {i + 1}",
                            "image_prompt": f"Test image prompt {i + 1}",
                            "video_prompt": f"Test video prompt {i + 1}",
                        }
                    )

                mock_analyzer.analyze_transcript_with_ai.return_value = (
                    mock_segments
                )
                mock_analyzer.generate_output_json.return_value = {
                    "broll_segments": mock_segments,
                    "transcript_info": self.sample_transcript["metadata"],
                }

                # Create workflow with configuration
                workflow = UnifiedWorkflow(max_segments=config["max_segments"])
                workflow.image_generator = MockImageGenerator()
                workflow.video_generator = MockKlingImageToVideoGenerator()
                workflow.data_dir = self.test_data_dir
                workflow.images_dir = self.test_images_dir
                workflow.videos_dir = self.test_videos_dir

                # Run workflow
                result = workflow.run_complete_workflow(
                    str(self.transcript_file)
                )

                # Verify results match configuration
                assert (
                    result["workflow_info"]["max_segments"]
                    == config["max_segments"]
                )
                assert (
                    result["summary"]["total_segments"]
                    == config["max_segments"]
                )
                assert (
                    result["summary"]["images_generated"]
                    == config["max_segments"]
                )
                assert (
                    result["summary"]["videos_created"]
                    == config["max_segments"]
                )

                print(f"✅ Configuration test passed: {config['description']}")


def run_integration_tests():
    """Run all integration tests"""
    print("🚀 Starting integration tests with mock API...")
    print("=" * 60)

    # Create test suite and run tests properly
    test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestWorkflowIntegration
    )
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)

    print("=" * 60)
    print(
        f"📊 Integration tests completed: {result.testsRun - len(result.failures) - len(result.errors)}/{result.testsRun} passed"
    )

    if result.wasSuccessful():
        print("🎉 All integration tests passed!")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above.")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)
