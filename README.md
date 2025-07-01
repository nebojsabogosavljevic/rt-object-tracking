# Real-Time Object Tracking Simulation

A Python-based object tracking system that detects and tracks circles and rectangles in video sequences, handling temporary disappearances and providing comprehensive visualization.

## Features

- **Multi-object Detection**: Detects circles and rectangles using OpenCV
- **Robust Tracking**: Handles temporary object disappearances (2.5% probability)
- **Comprehensive Visualization**: Multiple output formats including video, images, and statistics
- **Real-time Preview**: Optional live preview during processing
- **Detailed Statistics**: Track length analysis and object type distribution

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Process a video file with default settings:

```bash
python main.py path/to/your/video.mp4
```

### Advanced Usage

```bash
# Process with real-time preview
python main.py luxonis_task_video.mp4 --preview

# Specify custom output directory
python main.py luxonis_task_video.mp4 --output my_results

# Show tracking statistics
python main.py luxonis_task_video.mp4 --stats

# Combine multiple options
python main.py luxonis_task_video.mp4 --preview --stats --output results
```

### Command Line Arguments

- `video_path`: Path to the input video file (required)
- `--output, -o`: Output directory (default: "output")
- `--preview, -p`: Show real-time preview during processing
- `--stats, -s`: Display tracking statistics after processing

## Output Files

The application generates several output files in the specified output directory:

1. **tracking_summary.png**: Visual summary showing all object tracks
2. **tracking_result.mp4**: Video with tracking visualization overlaid
3. **tracking_data.txt**: Detailed text file with all tracking coordinates
4. **Statistics plots**: Interactive matplotlib plots showing tracking performance

## Architecture

### Object Detector (`object_detector.py`)

The `ObjectDetector` class provides:

- **Circle Detection**: Uses Hough Circle Transform with configurable parameters
- **Rectangle Detection**: Uses contour detection with polygon approximation
- **Color Analysis**: Extracts dominant colors from detected objects

**Usage:**
```python
from object_detector import ObjectDetector

detector = ObjectDetector()
detections = detector.detect_objects(frame)
```

### Object Tracker (`object_tracker.py`)

The `ObjectTracker` class provides:

- **Multi-object Tracking**: Maintains separate tracks for each detected object
- **Disappearance Handling**: Simulates 2.5% chance of temporary disappearance
- **Track Association**: Uses distance-based matching for object continuity
- **Track History**: Maintains complete position history for each object

**Usage:**
```python
from object_tracker import ObjectTracker

tracker = ObjectTracker()
tracked_objects = tracker.update(detections, frame_number)
```

### Visualizer (`visualizer.py`)

The `TrackingVisualizer` class provides:

- **Real-time Visualization**: Draws tracking information on video frames
- **Summary Generation**: Creates comprehensive tracking overview images
- **Statistical Analysis**: Generates plots for tracking performance analysis
- **Video Output**: Creates annotated video with tracking visualization

**Usage:**
```python
from visualizer import TrackingVisualizer

visualizer = TrackingVisualizer(frame_width, frame_height)
result_frame = visualizer.draw_tracks_on_frame(frame, tracked_objects)
```

## Configuration

### Detector Parameters

You can adjust detection sensitivity by modifying parameters in `ObjectDetector`:

```python
# Circle detection parameters
circle_params = {
    'dp': 1,           # Inverse ratio of accumulator resolution
    'minDist': 50,     # Minimum distance between circles
    'param1': 50,      # Upper threshold for edge detection
    'param2': 30,      # Threshold for center detection
    'minRadius': 20,   # Minimum circle radius
    'maxRadius': 100   # Maximum circle radius
}

# Rectangle detection parameters
contour_params = {
    'epsilon_factor': 0.02,  # Approximation accuracy
    'min_area': 500,         # Minimum contour area
    'max_area': 10000        # Maximum contour area
}
```

### Tracker Parameters

Adjust tracking behavior in `ObjectTracker`:

```python
tracker = ObjectTracker(
    max_disappeared_frames=3,    # Frames to keep track after disappearance
    distance_threshold=100.0     # Maximum distance for object association
)
```

## Performance Considerations

- **Processing Speed**: Detection and tracking are optimized for real-time performance
- **Memory Usage**: Track history is maintained in memory; consider clearing old tracks for very long videos
- **Accuracy vs Speed**: Adjust detection parameters based on your specific use case

## Troubleshooting

### Common Issues

1. **No objects detected**: Adjust detection parameters (min/max radius, area thresholds)
2. **Poor tracking**: Reduce distance threshold or increase max_disappeared_frames
3. **Slow performance**: Reduce frame resolution or adjust detection parameters

### Debug Mode

For debugging, you can enable more verbose output by modifying the main script:

```python
# Add debug prints in main.py
print(f"Detected {len(detections)} objects in frame {frame_number}")
```

## Example Output

The system generates comprehensive tracking results:

```
Processing video frames...
Progress: 25.0% (750/3000)
Progress: 50.0% (1500/3000)
Progress: 75.0% (2250/3000)
Processing complete. Tracked 5 objects.

Tracking Statistics:
==============================
Total Objects: 5
Circle Count: 3
Rectangle Count: 2
Avg Track Length: 245.6
Max Track Length: 300
Min Track Length: 180
```

## License

This project is created for the Luxonis interview task. Feel free to use and modify as needed.

## Contributing

This is a demonstration project, but suggestions for improvements are welcome! 