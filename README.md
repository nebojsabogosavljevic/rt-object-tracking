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
