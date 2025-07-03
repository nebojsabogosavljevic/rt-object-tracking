import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from object_tracker import TrackedObject


class TrackingVisualizer:
    """
    Visualizer for object tracking results.
    """
    
    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 0),    # Dark Blue
            (0, 128, 0),    # Dark Green
            (0, 0, 128),    # Dark Red
            (128, 128, 0),  # Olive
        ]
    
    def _validate_color(self, color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Validate and ensure color is in correct BGR format."""
        if not isinstance(color, tuple) or len(color) != 3:
            return (0, 0, 0)
        
        try:
            b, g, r = color
            b = max(0, min(255, int(b)))
            g = max(0, min(255, int(g)))
            r = max(0, min(255, int(r)))
            
            return (b, g, r)
        except (ValueError, TypeError):
            return (0, 0, 0)
    
    def _bgr_to_rgb(self, bgr_color: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert BGR color to RGB for matplotlib (0-1 range)."""
        try:
            b, g, r = bgr_color
            return (r / 255.0, g / 255.0, b / 255.0)
        except (ValueError, TypeError):
            return (0.0, 0.0, 0.0)
    
    def draw_tracks_on_frame(self, frame: np.ndarray, tracked_objects: List[TrackedObject],
                           show_history: bool = True, show_ids: bool = True) -> np.ndarray:
        """
        Draw tracking information on a frame.
        
        Args:
            frame: Input frame
            tracked_objects: List of tracked objects
            show_history: Whether to show track history
            show_ids: Whether to show object IDs
            
        Returns:
            Frame with tracking visualization
        """
        result_frame = frame.copy()
        
        for obj in tracked_objects:
            if obj.disappeared_frames > 0:
                continue 
            
            color = self._validate_color(self.colors[obj.object_id % len(self.colors)])
            current_pos = obj.get_last_position()
            
            if current_pos is None:
                continue
            
            if obj.object_type == 'circle':
                cv2.circle(result_frame, current_pos, 25, color, 3)
            else:
                x, y = current_pos
                cv2.rectangle(result_frame, (x-20, y-20), (x+20, y+20), color, 3)
            
            if show_ids:
                cv2.putText(result_frame, f"ID:{obj.object_id}", 
                           (current_pos[0] + 30, current_pos[1] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
            
            if show_history and len(obj.track_history) > 1:
                points = [pos for _, pos in obj.track_history]
                for i in range(1, len(points)):
                    cv2.line(result_frame, points[i-1], points[i], color, 3)
        
        return result_frame
    
    def create_tracking_summary(self, tracked_objects: List[TrackedObject], 
                              frame_width: int, frame_height: int) -> np.ndarray:
        """
        Create a summary visualization showing all object tracks.
        
        Args:
            tracked_objects: List of tracked objects
            frame_width: Width of the original video
            frame_height: Height of the original video
            
        Returns:
            Summary image showing all tracks
        """
        summary = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
        
        for obj in tracked_objects:
            if len(obj.track_history) < 2:
                continue
            
            color = self._validate_color(self.colors[obj.object_id % len(self.colors)])
            points = [pos for _, pos in obj.track_history]
            
            for i in range(1, len(points)):
                cv2.line(summary, points[i-1], points[i], color, 3)
            
            if points:
                cv2.circle(summary, points[0], 8, (0, 255, 0), -1)
                cv2.circle(summary, points[-1], 8, (0, 0, 255), -1)
                
                cv2.putText(summary, f"ID:{obj.object_id}", 
                           (points[-1][0] + 10, points[-1][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        legend_y = 30
        cv2.putText(summary, "Tracking Summary", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        legend_y += 30
        cv2.putText(summary, "Green: Start points", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        legend_y += 20
        cv2.putText(summary, "Red: End points", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        legend_y += 20
        cv2.putText(summary, f"Total objects tracked: {len(tracked_objects)}", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return summary
    
    def plot_track_statistics(self, tracked_objects: List[TrackedObject]) -> None:
        """
        Create statistical plots of tracking performance.
        
        Args:
            tracked_objects: List of tracked objects
        """
        if not tracked_objects:
            print("No tracked objects to plot statistics for.")
            return
        
        track_lengths = [obj.get_track_length() for obj in tracked_objects]
        object_types = [obj.object_type for obj in tracked_objects]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        ax1.hist(track_lengths, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Track Length (frames)')
        ax1.set_ylabel('Number of Objects')
        ax1.set_title('Track Length Distribution')
        ax1.grid(True, alpha=0.3)
        
        type_counts = {}
        for obj_type in object_types:
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
        
        ax2.bar(type_counts.keys(), type_counts.values(), color=['red', 'blue'], alpha=0.7)
        ax2.set_xlabel('Object Type')
        ax2.set_ylabel('Count')
        ax2.set_title('Object Type Distribution')
        ax2.grid(True, alpha=0.3)
        
        circle_lengths = [obj.get_track_length() for obj in tracked_objects if obj.object_type == 'circle']
        rect_lengths = [obj.get_track_length() for obj in tracked_objects if obj.object_type == 'rectangle']
        
        ax3.boxplot([circle_lengths, rect_lengths], labels=['Circle', 'Rectangle'])
        ax3.set_ylabel('Track Length (frames)')
        ax3.set_title('Track Length by Object Type')
        ax3.grid(True, alpha=0.3)
        
        for obj in tracked_objects:
            if len(obj.track_history) > 1:
                x_coords = [pos[0] for _, pos in obj.track_history]
                y_coords = [pos[1] for _, pos in obj.track_history]
                bgr_color = self.colors[obj.object_id % len(self.colors)]
                rgb_color = self._bgr_to_rgb(bgr_color)
                ax4.plot(x_coords, y_coords, color=rgb_color, alpha=0.7, linewidth=2)
        
        ax4.set_xlabel('X Coordinate')
        ax4.set_ylabel('Y Coordinate')
        ax4.set_title('Object Movement Patterns')
        ax4.grid(True, alpha=0.3)
        ax4.invert_yaxis()
        
        plt.tight_layout()
        plt.show()
    
    def save_tracking_video(self, video_path: str, output_path: str, 
                          tracked_objects: List[TrackedObject]) -> None:
        """
        Create a video with tracking visualization.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            tracked_objects: List of tracked objects
        """
        cap = cv2.VideoCapture(video_path)
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame_objects = []
            for obj in tracked_objects:
                current_history = [(f, pos) for f, pos in obj.track_history if f <= frame_number]
                
                if current_history:
                    class TempTrackedObject:
                        def __init__(self, obj_id, obj_type, history, color):
                            self.object_id = obj_id
                            self.object_type = obj_type
                            self.track_history = history
                            self.color = color
                            self.disappeared_frames = 0
                        
                        def get_last_position(self):
                            if self.track_history:
                                return self.track_history[-1][1]
                            return None
                    
                    is_active = current_history and current_history[-1][0] == frame_number
                    
                    temp_obj = TempTrackedObject(
                        obj.object_id,
                        obj.object_type,
                        current_history,
                        obj.color
                    )
                    
                    if not is_active:
                        temp_obj.disappeared_frames = 1
                    
                    current_frame_objects.append(temp_obj)
            
            result_frame = self.draw_tracks_on_frame(
                frame, 
                current_frame_objects, 
                show_history=True, 
                show_ids=True
            )
            
            out.write(result_frame)
            frame_number += 1
        
        cap.release()
        out.release()
        print(f"Tracking video saved to: {output_path}")
    
    def save_tracking_video_with_history(self, video_path: str, output_path: str, 
                                       frame_tracking_history: List[List]) -> None:
        """
        Create a video with tracking visualization using frame-by-frame tracking history.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            frame_tracking_history: List of tracked objects for each frame
        """
        cap = cv2.VideoCapture(video_path)
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_number < len(frame_tracking_history):
                current_frame_objects = frame_tracking_history[frame_number]
            else:
                current_frame_objects = []
            
            result_frame = self.draw_tracks_on_frame(
                frame, 
                current_frame_objects, 
                show_history=True, 
                show_ids=True
            )
            
            out.write(result_frame)
            frame_number += 1
        
        cap.release()
        out.release()
        print(f"Tracking video saved to: {output_path}") 