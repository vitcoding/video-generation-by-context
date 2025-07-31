# Tests for Workflow with Mock API

This document describes tests for workflow logic using mock API to avoid real API calls during testing.

## Test Structure

### 1. `test_workflow.py` - Main unit tests

Contains comprehensive tests for the `UnifiedWorkflow` class:

- **TestUnifiedWorkflow** - tests for main workflow components
- **TestWorkflowWithMockAPI** - tests using real mock API components

#### Main test cases:

1. **test_workflow_initialization** - workflow initialization check
2. **test_process_transcript_to_prompts** - test transcript processing into prompts
3. **test_generate_images_from_prompts** - test image generation
4. **test_create_videos_from_images** - test video creation from images
5. **test_complete_workflow_integration** - complete workflow integration test
6. **test_workflow_error_handling** - error handling test
7. **test_workflow_with_different_segment_counts** - test with different segment counts
8. **test_workflow_summary_printing** - summary output test

### 2. `test_integration.py` - Integration tests

Contains full workflow integration tests with mock API:

- **TestWorkflowIntegration** - integration tests with mock API
- **run_integration_tests()** - function to run all integration tests

#### Main test cases:

1. **test_workflow_with_mock_api_components** - complete workflow with mock API
2. **test_mock_api_components_individually** - test individual mock API components
3. **test_workflow_error_scenarios** - error scenario tests
4. **test_workflow_performance_with_mock** - performance test
5. **test_workflow_with_different_configurations** - test different configurations

## Running Tests

### Running unit tests

```bash
# Run all unit tests
python test_workflow.py

# Run specific test
python -m unittest test_workflow.TestUnifiedWorkflow.test_workflow_initialization

# Run with verbose output
python -m unittest test_workflow -v
```

### Running integration tests

```bash
# Run integration tests
python test_integration.py

# Run with additional debug information
python test_integration.py --verbose
```

### Running all tests

```bash
# Run all tests via unittest
python -m unittest discover -p "test_*.py" -v
```

## Mock API Components

Tests use the following mock components from `mock_api.py`:

### MockImageGenerator
- Simulates image generation via fal.ai API
- Returns mock image URLs
- Simulates file downloads

### MockKlingImageToVideoGenerator
- Simulates image to video conversion
- Returns mock video URLs
- Simulates video file downloads

### MockOpenAIClient
- Simulates OpenAI API for transcript analysis
- Generates mock prompts for b-roll content

## Test Data Structure

Tests create temporary directories with the following structure:

```
temp_dir/
├── data/
│   └── video_generation/
│       ├── audio_transcript/
│       │   └── test_transcript.json
│       └── broll_prompts/
├── images/
└── videos/
```

## Test Data Examples

### Sample Transcript
```json
{
  "transcript": [
    {
      "start": 0.0,
      "end": 5.0,
      "text": "Welcome to our video about artificial intelligence."
    },
    {
      "start": 5.0,
      "end": 10.0,
      "text": "AI is transforming the world around us."
    }
  ],
  "metadata": {
    "duration": 10.0,
    "language": "en",
    "speaker_count": 1
  }
}
```

### Expected Workflow Results
```json
{
  "workflow_info": {
    "input_transcript": "path/to/transcript.json",
    "total_duration": 15.5,
    "timestamp": "2024-01-01 12:00:00",
    "api_used": false,
    "max_segments": 2
  },
  "summary": {
    "total_segments": 2,
    "images_generated": 2,
    "videos_created": 2,
    "success_rate": {
      "images": "2/2",
      "videos": "2/2"
    }
  }
}
```

## Result Verification

### Successful test should show:

1. ✅ All workflow components initialized correctly
2. ✅ Transcript processed into prompts
3. ✅ Images generated from prompts
4. ✅ Videos created from images
5. ✅ Files saved in appropriate directories
6. ✅ Workflow summary created correctly

### Performance metrics:

- Workflow execution time: < 10 seconds (with mock API)
- Image generation success rate: 100%
- Video creation success rate: 100%
- Result structure correctness

## Test Debugging

### Enabling debug mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Checking mock API calls

```python
# In tests you can verify that mock API was called
mock_img_gen.generate_image.assert_called()
mock_video_gen.generate_video_from_image.assert_called()
```

### Checking created files

```python
# Checking file existence
assert prompts_file.exists()
assert report_file.exists()
assert len(list(images_dir.glob("*.png"))) > 0
assert len(list(videos_dir.glob("*.mp4"))) > 0
```

## Additional Features

### Testing with different configurations

```python
# Test with different segment counts
for max_segments in [1, 3, 5]:
    workflow = UnifiedWorkflow(max_segments=max_segments)
    # ... testing
```

### Testing error handling

```python
# Test with non-existent file
with self.assertRaises(Exception):
    workflow.process_transcript_to_prompts("non_existent.json")
```

### Performance testing

```python
import time
start_time = time.time()
result = workflow.run_complete_workflow(transcript_file)
duration = time.time() - start_time
assert duration < 10.0  # Should be fast with mock API
```

## Conclusion

Tests provide complete coverage of workflow logic without the need for real API calls, which allows:

1. Quick testing of code changes
2. Avoiding API call costs
3. Ensuring test stability
4. Verifying error handling
5. Measuring workflow performance 