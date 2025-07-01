import cv2
import numpy as np
from typing import List, Tuple, Dict, Any


class ObjectDetector:
    """
    Object detector for circles and rectangles using OpenCV.
    """
    
    def __init__(self):
        self.circle_params = {
            'dp': 1,
            'minDist': 30,
            'param1': 50,
            'param2': 25,
            'minRadius': 15,
            'maxRadius': 100
        }
        
        self.contour_params = {
            'epsilon_factor': 0.02,
            'min_area': 200,
            'max_area': 15000
        }
    
    def detect_circles(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect circles in the frame using Hough Circle Transform.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of dictionaries containing circle information
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            **self.circle_params
        )
        
        detected_circles = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                detected_circles.append({
                    'type': 'circle',
                    'center': (int(x), int(y)),
                    'radius': int(r),
                    'bbox': (int(x - r), int(y - r), int(x + r), int(y + r))
                })
        
        return detected_circles
    
    def detect_rectangles(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect rectangles in the frame using contour detection.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of dictionaries containing rectangle information
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_rectangles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.contour_params['min_area'] or area > self.contour_params['max_area']:
                continue
            
            # Approximate the contour to a polygon
            epsilon = self.contour_params['epsilon_factor'] * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's a rectangle (4 vertices)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / h
                
                # Filter based on aspect ratio to ensure it's roughly rectangular
                if 0.3 <= aspect_ratio <= 3.0:
                    detected_rectangles.append({
                        'type': 'rectangle',
                        'center': (int(x + w // 2), int(y + h // 2)),
                        'width': int(w),
                        'height': int(h),
                        'bbox': (int(x), int(y), int(x + w), int(y + h))
                    })
        
        return detected_rectangles
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect both circles and rectangles in the frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of dictionaries containing all detected objects
        """
        circles = self.detect_circles(frame)
        rectangles = self.detect_rectangles(frame)
        
        return circles + rectangles
    
    def get_object_color(self, frame: np.ndarray, center: Tuple[int, int], radius: int = 10) -> Tuple[int, int, int]:
        """
        Get the dominant color of an object at the given center.
        
        Args:
            frame: Input frame
            center: Center coordinates (x, y)
            radius: Radius around center to sample color
            
        Returns:
            BGR color tuple
        """
        x, y = center
        x1, y1 = max(0, x - radius), max(0, y - radius)
        x2, y2 = min(frame.shape[1], x + radius), min(frame.shape[0], y + radius)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return (0, 0, 0)
        
        # Get the mean color of the ROI
        mean_color = np.mean(roi, axis=(0, 1))
        return tuple(map(int, mean_color)) 