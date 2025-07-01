import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import random


class TrackedObject:
    """
    Represents a tracked object with its history and properties.
    """
    
    def __init__(self, object_id: int, initial_detection: Dict[str, Any], frame_number: int):
        self.object_id = object_id
        self.object_type = initial_detection['type']
        self.track_history = [(frame_number, initial_detection['center'])]
        self.last_seen = frame_number
        self.disappeared_frames = 0
        self.color = initial_detection.get('color', (0, 0, 0))
        self.properties = initial_detection
    
    def update(self, detection: Dict[str, Any], frame_number: int):
        """Update the tracked object with new detection."""
        self.track_history.append((frame_number, detection['center']))
        self.last_seen = frame_number
        self.disappeared_frames = 0
        self.properties = detection
    
    def mark_missing(self, frame_number: int):
        """Mark the object as missing for this frame."""
        self.disappeared_frames += 1
    
    def get_last_position(self) -> Optional[Tuple[int, int]]:
        """Get the last known position of the object."""
        if self.track_history:
            return self.track_history[-1][1]
        return None
    
    def get_track_length(self) -> int:
        """Get the number of frames this object has been tracked."""
        return len(self.track_history)


class ObjectTracker:
    """
    Multi-object tracker that can handle temporary disappearances.
    """
    
    def __init__(self, max_disappeared_frames: int = 3, distance_threshold: float = 100.0):
        self.tracked_objects: List[TrackedObject] = []
        self.next_object_id = 0
        self.max_disappeared_frames = max_disappeared_frames
        self.distance_threshold = distance_threshold
        self.disappearance_probability = 0.025  # 2.5% chance
    
    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _find_best_match(self, detection: Dict[str, Any], available_objects: List[TrackedObject]) -> Optional[TrackedObject]:
        """Find the best matching tracked object for a detection."""
        if not available_objects:
            return None
        
        best_match = None
        min_distance = float('inf')
        
        for obj in available_objects:
            last_pos = obj.get_last_position()
            if last_pos is None:
                continue
            
            distance = self._calculate_distance(detection['center'], last_pos)
            
            # Check if object types match
            if obj.object_type == detection['type'] and distance < min_distance:
                min_distance = distance
                best_match = obj
        
        # Only return match if distance is within threshold
        if best_match and min_distance <= self.distance_threshold:
            return best_match
        
        return None
    
    def update(self, detections: List[Dict[str, Any]], frame_number: int) -> List[TrackedObject]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detected objects in current frame
            frame_number: Current frame number
            
        Returns:
            List of currently tracked objects
        """
        # Mark all objects as missing initially
        for obj in self.tracked_objects:
            obj.mark_missing(frame_number)
        
        # Try to match detections with existing tracked objects
        matched_detection_indices = set()
        available_objects = [obj for obj in self.tracked_objects 
                           if obj.disappeared_frames <= self.max_disappeared_frames]
        
        for i, detection in enumerate(detections):
            # Simulate 2.5% chance of object disappearing
            if random.random() < self.disappearance_probability:
                continue
            
            best_match = self._find_best_match(detection, available_objects)
            
            if best_match:
                best_match.update(detection, frame_number)
                matched_detection_indices.add(i)
                available_objects.remove(best_match)
        
        # Create new tracked objects for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detection_indices:
                new_object = TrackedObject(self.next_object_id, detection, frame_number)
                self.tracked_objects.append(new_object)
                self.next_object_id += 1
        
        # Remove objects that have been missing for too long
        self.tracked_objects = [obj for obj in self.tracked_objects 
                              if obj.disappeared_frames <= self.max_disappeared_frames]
        
        return self.tracked_objects
    
    def get_active_tracks(self) -> List[TrackedObject]:
        """Get all currently active tracked objects."""
        return [obj for obj in self.tracked_objects if obj.disappeared_frames == 0]
    
    def get_all_tracks(self) -> List[TrackedObject]:
        """Get all tracked objects including those that disappeared recently."""
        return self.tracked_objects
    
    def get_track_history(self, object_id: int) -> List[Tuple[int, Tuple[int, int]]]:
        """Get the complete track history for a specific object."""
        for obj in self.tracked_objects:
            if obj.object_id == object_id:
                return obj.track_history
        return [] 