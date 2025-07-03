#!/usr/bin/env python3
"""
Real-Time Object Tracking Simulation

This script processes a video file to detect and track circles and rectangles,
handling temporary disappearances and providing comprehensive visualization.
"""

import cv2
import numpy as np
import argparse
import os
from typing import List, Dict, Any

from object_detector import ObjectDetector
from object_tracker import ObjectTracker
from visualizer import TrackingVisualizer


class ObjectTrackingApp:
    """
    Main application class for object tracking simulation.
    """
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.detector = ObjectDetector()
        self.tracker = ObjectTracker()
        self.visualizer = None
        self.tracked_objects = []
        
    def process_video(self, output_dir: str = "output", show_preview: bool = False) -> None:
        """
        Process the video and perform object tracking.
        
        Args:
            output_dir: Directory to save output files
            show_preview: Whether to show real-time preview
        """
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
        
        self.visualizer = TrackingVisualizer(frame_width, frame_height)
        
        self.frame_tracking_history = []
        
        frame_number = 0
        print("Processing video frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = self.detector.detect_objects(frame)
            
            for detection in detections:
                color = self.detector.get_object_color(frame, detection['center'])
                detection['color'] = color
            
            self.tracked_objects = self.tracker.update(detections, frame_number)
            
            frame_objects = []
            for obj in self.tracked_objects:
                frame_obj = type('FrameObject', (), {})()
                frame_obj.object_id = obj.object_id
                frame_obj.object_type = obj.object_type
                
                current_history = [(f, pos) for f, pos in obj.track_history if f <= frame_number]
                frame_obj.track_history = current_history
                
                frame_obj.color = obj.color
                frame_obj.disappeared_frames = obj.disappeared_frames
                
                def get_current_position(obj_history=current_history):
                    if obj_history:
                        return obj_history[-1][1]
                    return None
                
                frame_obj.get_last_position = get_current_position
                frame_objects.append(frame_obj)
            
            self.frame_tracking_history.append(frame_objects)
            
            if show_preview:
                result_frame = self.visualizer.draw_tracks_on_frame(frame, self.tracked_objects)
                cv2.imshow('Object Tracking', result_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if frame_number % 30 == 0:
                progress = (frame_number / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_number}/{total_frames})")
            
            frame_number += 1
        
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        self.tracked_objects = self.tracker.get_all_tracks()
        print(f"Processing complete. Tracked {len(self.tracked_objects)} objects.")
        
        self._generate_outputs(output_dir)
    
    def _generate_outputs(self, output_dir: str) -> None:
        """
        Generate all output visualizations and statistics.
        
        Args:
            output_dir: Directory to save outputs
        """
        print("Generating output visualizations...")
        
        summary_image = self.visualizer.create_tracking_summary(
            self.tracked_objects, 
            self.visualizer.frame_width, 
            self.visualizer.frame_height
        )
        
        summary_path = os.path.join(output_dir, "tracking_summary.png")
        cv2.imwrite(summary_path, summary_image)
        print(f"Tracking summary saved to: {summary_path}")
        
        video_output_path = os.path.join(output_dir, "tracking_result.mp4")
        self.visualizer.save_tracking_video_with_history(self.video_path, video_output_path, self.frame_tracking_history)
        
        self.visualizer.plot_track_statistics(self.tracked_objects)
        
        self._save_tracking_data(output_dir)
    
    def _save_tracking_data(self, output_dir: str) -> None:
        """
        Save tracking data to a text file.
        
        Args:
            output_dir: Directory to save the data file
        """
        data_path = os.path.join(output_dir, "tracking_data.txt")
        
        with open(data_path, 'w') as f:
            f.write("Object Tracking Results\n")
            f.write("=" * 50 + "\n\n")
            
            for obj in self.tracked_objects:
                f.write(f"Object ID: {obj.object_id}\n")
                f.write(f"Type: {obj.object_type}\n")
                f.write(f"Track Length: {obj.get_track_length()} frames\n")
                f.write("Track History:\n")
                
                for frame_num, pos in obj.track_history:
                    f.write(f"  Frame {frame_num}: ({pos[0]}, {pos[1]})\n")
                
                f.write("\n" + "-" * 30 + "\n\n")
        
        print(f"Tracking data saved to: {data_path}")
    
    def get_tracking_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive tracking statistics.
        
        Returns:
            Dictionary containing tracking statistics
        """
        if not self.tracked_objects:
            return {}
        
        stats = {
            'total_objects': len(self.tracked_objects),
            'circle_count': sum(1 for obj in self.tracked_objects if obj.object_type == 'circle'),
            'rectangle_count': sum(1 for obj in self.tracked_objects if obj.object_type == 'rectangle'),
            'avg_track_length': np.mean([obj.get_track_length() for obj in self.tracked_objects]),
            'max_track_length': max([obj.get_track_length() for obj in self.tracked_objects]),
            'min_track_length': min([obj.get_track_length() for obj in self.tracked_objects]),
        }
        
        return stats


def main():
    """Main function to run the object tracking application."""
    parser = argparse.ArgumentParser(description="Real-Time Object Tracking Simulation")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--output", "-o", default="output", help="Output directory (default: output)")
    parser.add_argument("--preview", "-p", action="store_true", help="Show real-time preview")
    parser.add_argument("--stats", "-s", action="store_true", help="Show tracking statistics")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found.")
        return
    
    try:
        app = ObjectTrackingApp(args.video_path)
        app.process_video(output_dir=args.output, show_preview=args.preview)
        
        if args.stats:
            stats = app.get_tracking_statistics()
            print("\nTracking Statistics:")
            print("=" * 30)
            for key, value in stats.items():
                print(f"{key.replace('_', ' ').title()}: {value}")
    
    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main() 