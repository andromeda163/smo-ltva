"""
Long-term point tracker with per-frame new-point initialization.

This module implements the complete point tracking pipeline from Ochs et al.,
including:
- Forward-backward consistency checking for occlusion detection
- Per-frame new-point initialization for disocclusion handling
- Trajectory management with active/occluded/lost states
- Dense coverage maintenance throughout the video

The key innovation is handling disocclusions: when a point becomes visible
again after being occluded, we initialize a new tracking point in that region.
"""

from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from point_selection import select_grid_points, select_best_points, compute_corner_response
from optical_flow import ForwardBackwardTracker


class PointState(Enum):
    """State of a tracking point."""
    ACTIVE = 'active'           # Currently being tracked successfully
    OCCLUDED = 'occluded'        # Temporarily lost (may reappear)
    LOST = 'lost'               # Permanently lost


@dataclass
class TrackedPoint:
    """
    A single tracked point with its trajectory and state.
    
    Attributes:
        point_id: Unique identifier for this point.
        trajectory: List of (x, y) positions, one per frame.
        states: List of PointState, one per frame.
        fb_errors: List of forward-backward errors, one per frame.
        birth_frame: Frame when this point was initialized.
        last_active_frame: Last frame where point was successfully tracked.
    """
    point_id: int
    trajectory: List[np.ndarray] = field(default_factory=list)
    states: List[PointState] = field(default_factory=list)
    fb_errors: List[float] = field(default_factory=list)
    birth_frame: int = 0
    last_active_frame: int = 0
    
    def get_position(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get position at a specific frame (absolute frame index)."""
        # Convert absolute frame index to relative index
        rel_idx = frame_idx - self.birth_frame
        if 0 <= rel_idx < len(self.trajectory):
            return self.trajectory[rel_idx]
        return None
    
    def get_state(self, frame_idx: int) -> PointState:
        """Get state at a specific frame (absolute frame index)."""
        # Convert absolute frame index to relative index
        rel_idx = frame_idx - self.birth_frame
        if 0 <= rel_idx < len(self.states):
            return self.states[rel_idx]
        return PointState.LOST
    
    def is_active_at(self, frame_idx: int) -> bool:
        """Check if point was active at a specific frame (absolute frame index)."""
        return self.get_state(frame_idx) == PointState.ACTIVE


class PointTracker:
    """
    Long-term point tracker with disocclusion handling.
    
    This tracker maintains a set of tracked points throughout a video,
    initializing new points when:
    - Existing points are lost
    - New regions become visible (disocclusions)
    - Coverage becomes sparse
    
    The tracker uses:
    - Forward-backward consistency for reliable tracking
    - Grid-based point selection for even coverage
    - Minimum distance constraints to avoid overlapping points
    """
    
    def __init__(
        self,
        fb_threshold: float = 1.0,
        window_size: int = 15,
        max_level: int = 3,
        min_distance: int = 10,
        grid_spacing: int = 20,
        min_corner_response: float = 2.0,
        border: int = 10,
        max_occluded_frames: int = 5,
        min_active_ratio: float = 0.3
    ):
        """
        Initialize the point tracker.
        
        Args:
            fb_threshold: Maximum forward-backward error for valid tracks.
            window_size: Lucas-Kanade window size.
            max_level: Number of pyramid levels.
            min_distance: Minimum distance between points.
            grid_spacing: Spacing for grid-based point selection.
            min_corner_response: Minimum corner response for new points.
            border: Border margin to exclude points.
            max_occluded_frames: Maximum frames a point can be occluded before lost.
            min_active_ratio: Minimum ratio of active points to maintain.
        """
        self.tracker = ForwardBackwardTracker(
            fb_threshold=fb_threshold,
            window_size=window_size,
            max_level=max_level
        )
        self.min_distance = min_distance
        self.grid_spacing = grid_spacing
        self.min_corner_response = min_corner_response
        self.border = border
        self.max_occluded_frames = max_occluded_frames
        self.min_active_ratio = min_active_ratio
        
        # Tracking state
        self.points: Dict[int, TrackedPoint] = {}
        self.next_point_id: int = 0
        self.current_frame: int = 0
        
        # Active point tracking
        self.active_point_ids: Set[int] = set()
        self.occluded_point_ids: Set[int] = set()
    
    def _get_next_id(self) -> int:
        """Get the next unique point ID."""
        point_id = self.next_point_id
        self.next_point_id += 1
        return point_id
    
    def _initialize_points(
        self,
        frame: np.ndarray,
        existing_points: np.ndarray,
        num_points: int
    ) -> np.ndarray:
        """
        Initialize new tracking points in uncovered regions.
        
        Args:
            frame: Current frame.
            existing_points: Array of existing point positions (N, 2).
            num_points: Number of new points to initialize.
            
        Returns:
            Array of new point positions (M, 2) where M <= num_points.
        """
        H, W = frame.shape[:2]
        
        # Compute corner response
        response = compute_corner_response(frame)
        
        # Create exclusion mask for existing points
        exclusion_mask = np.zeros((H, W), dtype=bool)
        
        for pt in existing_points:
            x, y = int(pt[0]), int(pt[1])
            # Exclude circular region around each existing point
            y_min = max(0, y - self.min_distance)
            y_max = min(H, y + self.min_distance + 1)
            x_min = max(0, x - self.min_distance)
            x_max = min(W, x + self.min_distance + 1)
            
            for yi in range(y_min, y_max):
                for xi in range(x_min, x_max):
                    if (xi - x) ** 2 + (yi - y) ** 2 <= self.min_distance ** 2:
                        exclusion_mask[yi, xi] = True
        
        # Also exclude border
        exclusion_mask[:self.border, :] = True
        exclusion_mask[-self.border:, :] = True
        exclusion_mask[:, :self.border] = True
        exclusion_mask[:, -self.border:] = True
        
        # Mask out excluded regions
        masked_response = response.copy()
        masked_response[exclusion_mask] = -np.inf
        
        # Select best points
        new_points = []
        
        for _ in range(num_points):
            # Find global maximum
            idx = np.argmax(masked_response)
            y, x = np.unravel_index(idx, masked_response.shape)
            
            if masked_response[y, x] < self.min_corner_response:
                break
            
            new_points.append([x, y])
            
            # Exclude region around this new point
            y_min = max(0, y - self.min_distance)
            y_max = min(H, y + self.min_distance + 1)
            x_min = max(0, x - self.min_distance)
            x_max = min(W, x + self.min_distance + 1)
            
            for yi in range(y_min, y_max):
                for xi in range(x_min, x_max):
                    if (xi - x) ** 2 + (yi - y) ** 2 <= self.min_distance ** 2:
                        masked_response[yi, xi] = -np.inf
        
        if len(new_points) == 0:
            return np.array([]).reshape(0, 2)
        
        return np.array(new_points)
    
    def _get_active_positions(self) -> np.ndarray:
        """Get positions of all currently active points."""
        positions = []
        for point_id in self.active_point_ids:
            point = self.points[point_id]
            if point.trajectory:
                positions.append(point.trajectory[-1])
        
        if len(positions) == 0:
            return np.array([]).reshape(0, 2)
        
        return np.array(positions)
    
    def initialize(self, frame: np.ndarray, num_points: int = 100) -> int:
        """
        Initialize tracking with a set of points from the first frame.
        
        Args:
            frame: First frame of the video.
            num_points: Number of initial points to select.
            
        Returns:
            Number of points initialized.
        """
        # Reset state
        self.points = {}
        self.next_point_id = 0
        self.current_frame = 0
        self.active_point_ids = set()
        self.occluded_point_ids = set()
        
        # Select initial points
        initial_positions, _ = select_best_points(
            frame,
            num_points=num_points,
            min_distance=self.min_distance,
            border=self.border
        )
        
        # Create tracked points
        for pos in initial_positions:
            point_id = self._get_next_id()
            point = TrackedPoint(
                point_id=point_id,
                trajectory=[pos.copy()],
                states=[PointState.ACTIVE],
                fb_errors=[0.0],
                birth_frame=0,
                last_active_frame=0
            )
            self.points[point_id] = point
            self.active_point_ids.add(point_id)
        
        return len(initial_positions)
    
    def track_frame(
        self,
        frame_prev: np.ndarray,
        frame_curr: np.ndarray,
        add_new_points: bool = True,
        target_num_points: Optional[int] = None,
        verbose: bool = False
    ) -> dict:
        """
        Track points from previous frame to current frame.
        
        This method:
        1. Tracks existing active points using FB validation
        2. Updates point states (active/occluded/lost)
        3. Optionally initializes new points in uncovered regions
        
        Args:
            frame_prev: Previous frame.
            frame_curr: Current frame.
            add_new_points: Whether to add new points for disocclusions.
            target_num_points: Target number of active points. If None, uses initial count.
            verbose: Print debug information.
            
        Returns:
            Dictionary with tracking statistics for this frame.
        """
        self.current_frame += 1
        frame_idx = self.current_frame
        
        # Get active points to track
        active_ids = list(self.active_point_ids)
        
        if len(active_ids) > 0:
            # Get positions of active points
            active_positions = np.array([
                self.points[pid].trajectory[-1] for pid in active_ids
            ])
            
            # Track with FB validation
            new_positions, status, fb_errors, _ = self.tracker.track(
                frame_prev, frame_curr, active_positions
            )
            
            # Update tracked points
            for i, point_id in enumerate(active_ids):
                point = self.points[point_id]
                
                if status[i]:
                    # Successfully tracked
                    point.trajectory.append(new_positions[i].copy())
                    point.states.append(PointState.ACTIVE)
                    point.fb_errors.append(fb_errors[i])
                    point.last_active_frame = frame_idx
                else:
                    # Tracking failed - mark as occluded
                    last_pos = point.trajectory[-1].copy()
                    point.trajectory.append(last_pos)
                    point.states.append(PointState.OCCLUDED)
                    point.fb_errors.append(float('inf'))
                    
                    # Move from active to occluded
                    self.active_point_ids.discard(point_id)
                    self.occluded_point_ids.add(point_id)
        else:
            status = np.array([])
            fb_errors = np.array([])
        
        # Check occluded points - mark as lost if too long
        lost_ids = []
        for point_id in list(self.occluded_point_ids):
            point = self.points[point_id]
            frames_occluded = frame_idx - point.last_active_frame
            
            if frames_occluded > self.max_occluded_frames:
                point.states[-1] = PointState.LOST
                lost_ids.append(point_id)
        
        for point_id in lost_ids:
            self.occluded_point_ids.discard(point_id)
        
        # Add new points if needed
        num_new_points = 0
        if add_new_points:
            # Determine target number of points
            if target_num_points is None:
                target_num_points = len(self.points)
            
            current_active = len(self.active_point_ids)
            points_needed = target_num_points - current_active
            
            if points_needed > 0:
                # Get all current point positions (active + occluded)
                all_positions = []
                for point_id in self.active_point_ids | self.occluded_point_ids:
                    all_positions.append(self.points[point_id].trajectory[-1])
                
                existing_positions = np.array(all_positions) if all_positions else np.array([]).reshape(0, 2)
                
                # Initialize new points
                new_positions = self._initialize_points(
                    frame_curr, existing_positions, points_needed
                )
                
                # Add new tracked points
                for pos in new_positions:
                    point_id = self._get_next_id()
                    point = TrackedPoint(
                        point_id=point_id,
                        trajectory=[pos.copy()],
                        states=[PointState.ACTIVE],
                        fb_errors=[0.0],
                        birth_frame=frame_idx,
                        last_active_frame=frame_idx
                    )
                    self.points[point_id] = point
                    self.active_point_ids.add(point_id)
                    num_new_points += 1
        
        # Compile statistics
        stats = {
            'frame': frame_idx,
            'active': len(self.active_point_ids),
            'occluded': len(self.occluded_point_ids),
            'lost_this_frame': len(lost_ids),
            'new_points': num_new_points,
            'total_points': len(self.points),
            'avg_fb_error': float(np.mean(fb_errors[status])) if len(status) > 0 and np.any(status) else 0.0
        }
        
        if verbose:
            print(f"  Frame {frame_idx}: active={stats['active']}, "
                  f"occluded={stats['occluded']}, new={stats['new_points']}, "
                  f"avg_fb={stats['avg_fb_error']:.3f}")
        
        return stats
    
    def track_sequence(
        self,
        frames: List[np.ndarray],
        initial_points: int = 100,
        add_new_points: bool = True,
        verbose: bool = False
    ) -> dict:
        """
        Track points through an entire video sequence.
        
        Args:
            frames: List of video frames.
            initial_points: Number of points to initialize.
            add_new_points: Whether to add new points for disocclusions.
            verbose: Print progress information.
            
        Returns:
            Dictionary with complete tracking results and statistics.
        """
        num_frames = len(frames)
        
        if verbose:
            print(f"Tracking {num_frames} frames with {initial_points} initial points")
        
        # Initialize on first frame
        num_init = self.initialize(frames[0], num_points=initial_points)
        
        if verbose:
            print(f"  Initialized {num_init} points")
        
        # Track through remaining frames
        frame_stats = []
        for i in range(1, num_frames):
            stats = self.track_frame(
                frames[i - 1], frames[i],
                add_new_points=add_new_points,
                target_num_points=initial_points,
                verbose=verbose
            )
            frame_stats.append(stats)
        
        # Compile final statistics
        total_trajectories = len(self.points)
        survived_all = sum(
            1 for p in self.points.values()
            if p.get_state(num_frames - 1) == PointState.ACTIVE
        )
        
        avg_lifetime = np.mean([
            p.last_active_frame - p.birth_frame + 1
            for p in self.points.values()
        ])
        
        return {
            'trajectories': self.points,
            'frame_stats': frame_stats,
            'num_frames': num_frames,
            'total_trajectories': total_trajectories,
            'survived_all_frames': survived_all,
            'avg_lifetime': avg_lifetime
        }
    
    def get_trajectory_array(self) -> np.ndarray:
        """
        Get all trajectories as a 3D array.
        
        Returns:
            Array of shape (num_points, num_frames, 2) with (x, y) positions.
            Points that don't exist at a frame have NaN values.
        """
        if len(self.points) == 0:
            return np.array([])
        
        num_frames = self.current_frame + 1
        num_points = len(self.points)
        
        trajectories = np.full((num_points, num_frames, 2), np.nan)
        
        for i, point in enumerate(self.points.values()):
            for f, pos in enumerate(point.trajectory):
                trajectories[i, f] = pos
        
        return trajectories
    
    def get_active_positions_at(self, frame_idx: int) -> np.ndarray:
        """
        Get positions of all active points at a specific frame.
        
        Args:
            frame_idx: Frame index.
            
        Returns:
            Array of shape (N, 2) with active point positions.
        """
        positions = []
        for point in self.points.values():
            if point.is_active_at(frame_idx):
                pos = point.get_position(frame_idx)
                if pos is not None:
                    positions.append(pos)
        
        if len(positions) == 0:
            return np.array([]).reshape(0, 2)
        
        return np.array(positions)


def create_tracker_video(
    video_path: str,
    output_path: str,
    num_points: int = 100,
    fb_threshold: float = 1.0,
    min_distance: int = 10,
    add_new_points: bool = True,
    trail_length: int = 20,
    fps: Optional[int] = None,
    verbose: bool = True
) -> dict:
    """
    Create a video showing point tracking with disocclusion handling.
    
    Args:
        video_path: Path to input video directory.
        output_path: Path for output video file.
        num_points: Number of initial points to track.
        fb_threshold: Maximum forward-backward error.
        min_distance: Minimum distance between points.
        add_new_points: Whether to add new points for disocclusions.
        trail_length: Length of trajectory trail to draw.
        fps: Output video FPS.
        verbose: Print progress information.
        
    Returns:
        Dictionary with tracking statistics.
    """
    from video_loader import VideoLoader
    
    # Load video
    video = VideoLoader(video_path)
    num_frames = len(video)
    
    if verbose:
        print(f"Loading video: {video_path}")
        print(f"  Frames: {num_frames}")
        print(f"  Shape: {video.frame_shape}")
        print(f"  Add new points: {add_new_points}")
    
    # Initialize tracker
    tracker = PointTracker(
        fb_threshold=fb_threshold,
        min_distance=min_distance,
        grid_spacing=min_distance * 2
    )
    
    # Load all frames
    frames = [video[i] for i in range(num_frames)]
    
    # Track sequence
    result = tracker.track_sequence(
        frames,
        initial_points=num_points,
        add_new_points=add_new_points,
        verbose=verbose
    )
    
    # Create output video
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    H, W = video.frame_shape[:2]
    if fps is None:
        fps = 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
    
    if verbose:
        print(f"Creating output video: {output_path}")
    
    # Color palette for trajectories
    np.random.seed(42)
    colors = {}
    
    # Process each frame
    for frame_idx in range(num_frames):
        frame = frames[frame_idx].copy()
        
        # Draw all trajectories
        for point_id, point in tracker.points.items():
            if point_id not in colors:
                colors[point_id] = tuple(
                    int(c) for c in np.random.randint(50, 255, 3, dtype=np.uint8)
                )
            
            color = colors[point_id]
            state = point.get_state(frame_idx)
            
            if state == PointState.LOST:
                continue
            
            pos = point.get_position(frame_idx)
            if pos is None:
                continue
            
            # Draw trail
            start_frame = max(0, frame_idx - trail_length + 1)
            trail_points = []
            
            for f in range(start_frame, frame_idx + 1):
                if f < len(point.trajectory):
                    s = point.states[f] if f < len(point.states) else PointState.LOST
                    if s != PointState.LOST:
                        trail_points.append(point.trajectory[f])
            
            if len(trail_points) >= 2:
                for j in range(len(trail_points) - 1):
                    alpha = (j + 1) / len(trail_points)
                    fade_color = tuple(int(c * alpha) for c in color)
                    pt1 = tuple(trail_points[j].astype(int))
                    pt2 = tuple(trail_points[j + 1].astype(int))
                    cv2.line(frame, pt1, pt2, fade_color, 2)
            
            # Draw current point with state color
            pt = tuple(pos.astype(int))
            if state == PointState.ACTIVE:
                cv2.circle(frame, pt, 4, color, -1)
                cv2.circle(frame, pt, 5, (255, 255, 255), 1)
            elif state == PointState.OCCLUDED:
                # Draw with transparency effect for occluded
                cv2.circle(frame, pt, 3, color, 1)
        
        # Draw frame info
        active = sum(1 for p in tracker.points.values() if p.is_active_at(frame_idx))
        occluded = sum(1 for p in tracker.points.values() 
                      if frame_idx < len(p.states) and p.states[frame_idx] == PointState.OCCLUDED)
        
        info_text = f"Frame: {frame_idx + 1}/{num_frames}"
        info_text2 = f"Active: {active}, Occluded: {occluded}, Total: {len(tracker.points)}"
        
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, info_text2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    
    if verbose:
        print(f"\nTracking Statistics:")
        print(f"  Total trajectories: {result['total_trajectories']}")
        print(f"  Survived all frames: {result['survived_all_frames']}")
        print(f"  Average lifetime: {result['avg_lifetime']:.1f} frames")
        print(f"  Output saved to: {output_path}")
    
    return result


def create_comparison_video(
    video_path: str,
    output_path: str,
    num_points: int = 100,
    fb_threshold: float = 1.0,
    min_distance: int = 10,
    trail_length: int = 20,
    fps: Optional[int] = None,
    verbose: bool = True
) -> dict:
    """
    Create a side-by-side comparison video showing tracking with and without disocclusion handling.
    
    Left side: Without new point initialization (points fade away)
    Right side: With new point initialization (coverage maintained)
    
    Visual markers:
    - Solid circle: Active points (tracked successfully)
    - Hollow circle (outline): Occluded points
    - X marker: Newly initialized points (shown for first 10 frames)
    
    Args:
        video_path: Path to input video directory.
        output_path: Path for output video file.
        num_points: Number of initial points to track.
        fb_threshold: Maximum forward-backward error.
        min_distance: Minimum distance between points.
        trail_length: Length of trajectory trail to draw.
        fps: Output video FPS.
        verbose: Print progress information.
        
    Returns:
        Dictionary with comparison statistics.
    """
    from video_loader import VideoLoader
    
    # Load video
    video = VideoLoader(video_path)
    num_frames = len(video)
    
    if verbose:
        print(f"Loading video: {video_path}")
        print(f"  Frames: {num_frames}")
        print(f"  Shape: {video.frame_shape}")
    
    # Initialize two trackers
    tracker_static = PointTracker(
        fb_threshold=fb_threshold,
        min_distance=min_distance,
        grid_spacing=min_distance * 2
    )
    tracker_dynamic = PointTracker(
        fb_threshold=fb_threshold,
        min_distance=min_distance,
        grid_spacing=min_distance * 2
    )
    
    # Load all frames
    frames = [video[i] for i in range(num_frames)]
    
    # Track sequences
    if verbose:
        print("\nTracking WITHOUT new point initialization...")
    result_static = tracker_static.track_sequence(
        frames,
        initial_points=num_points,
        add_new_points=False,
        verbose=False
    )
    
    if verbose:
        print("Tracking WITH new point initialization...")
    result_dynamic = tracker_dynamic.track_sequence(
        frames,
        initial_points=num_points,
        add_new_points=True,
        verbose=False
    )
    
    # Create output video
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    H, W = video.frame_shape[:2]
    # Side by side: 2x width + divider
    out_W = W * 2 + 4
    if fps is None:
        fps = 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_W, H))
    
    if verbose:
        print(f"Creating comparison video: {output_path}")
    
    # Color palette - use consistent colors for initial points
    np.random.seed(42)
    initial_colors = np.random.randint(50, 255, size=(num_points, 3), dtype=np.uint8)
    
    # Process each frame
    for frame_idx in range(num_frames):
        # Create side-by-side frame
        frame_left = frames[frame_idx].copy()
        frame_right = frames[frame_idx].copy()
        
        # Draw static tracker (left side)
        _draw_frame_trajectories(
            frame_left, tracker_static, frame_idx, trail_length,
            initial_colors, num_points, is_static=True
        )
        
        # Draw dynamic tracker (right side)
        _draw_frame_trajectories(
            frame_right, tracker_dynamic, frame_idx, trail_length,
            initial_colors, num_points, is_static=False
        )
        
        # Add labels
        cv2.putText(frame_left, "WITHOUT Disocclusion Handling", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_right, "WITH Disocclusion Handling", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add stats
        active_static = sum(1 for p in tracker_static.points.values() if p.is_active_at(frame_idx))
        active_dynamic = sum(1 for p in tracker_dynamic.points.values() if p.is_active_at(frame_idx))
        new_dynamic = sum(1 for p in tracker_dynamic.points.values() 
                         if p.birth_frame == frame_idx)
        
        cv2.putText(frame_left, f"Active: {active_static}", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame_right, f"Active: {active_dynamic}, New: {new_dynamic}", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Combine frames side by side with divider
        divider = np.ones((H, 4, 3), dtype=np.uint8) * 128
        combined = np.hstack([frame_left, divider, frame_right])
        
        frame_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    
    # Compile comparison statistics
    stats = {
        'video': video_path,
        'num_frames': num_frames,
        'initial_points': num_points,
        'static': {
            'total_trajectories': result_static['total_trajectories'],
            'survived_all_frames': result_static['survived_all_frames'],
            'avg_lifetime': result_static['avg_lifetime'],
            'final_active': sum(1 for p in tracker_static.points.values() 
                               if p.is_active_at(num_frames - 1))
        },
        'dynamic': {
            'total_trajectories': result_dynamic['total_trajectories'],
            'survived_all_frames': result_dynamic['survived_all_frames'],
            'avg_lifetime': result_dynamic['avg_lifetime'],
            'final_active': sum(1 for p in tracker_dynamic.points.values() 
                               if p.is_active_at(num_frames - 1)),
            'new_points_added': result_dynamic['total_trajectories'] - num_points
        },
        'output_path': str(output_path)
    }
    
    if verbose:
        print(f"\nComparison Statistics:")
        print(f"  WITHOUT disocclusion handling:")
        print(f"    Total trajectories: {stats['static']['total_trajectories']}")
        print(f"    Survived all frames: {stats['static']['survived_all_frames']}")
        print(f"    Final active points: {stats['static']['final_active']}")
        print(f"    Average lifetime: {stats['static']['avg_lifetime']:.1f} frames")
        print(f"  WITH disocclusion handling:")
        print(f"    Total trajectories: {stats['dynamic']['total_trajectories']}")
        print(f"    New points added: {stats['dynamic']['new_points_added']}")
        print(f"    Survived all frames: {stats['dynamic']['survived_all_frames']}")
        print(f"    Final active points: {stats['dynamic']['final_active']}")
        print(f"    Average lifetime: {stats['dynamic']['avg_lifetime']:.1f} frames")
        print(f"  Output saved to: {stats['output_path']}")
    
    return stats


def _draw_frame_trajectories(
    frame: np.ndarray,
    tracker: 'PointTracker',
    frame_idx: int,
    trail_length: int,
    initial_colors: np.ndarray,
    num_initial: int,
    is_static: bool
):
    """Draw trajectories for a single frame with appropriate markers."""
    
    for point_id, point in tracker.points.items():
        state = point.get_state(frame_idx)
        
        if state == PointState.LOST and frame_idx >= len(point.states):
            continue
        
        # Get position
        pos = point.get_position(frame_idx)
        if pos is None:
            continue
        
        # Determine color
        if point_id < num_initial:
            # Initial point - use consistent color
            color = tuple(int(c) for c in initial_colors[point_id])
        else:
            # New point - use a distinct color based on ID
            np.random.seed(point_id + 1000)
            color = tuple(int(c) for c in np.random.randint(50, 255, 3, dtype=np.uint8))
        
        # Draw trail
        start_frame = max(0, frame_idx - trail_length + 1)
        trail_points = []
        
        for f in range(start_frame, frame_idx + 1):
            if f < len(point.trajectory):
                s = point.states[f] if f < len(point.states) else PointState.LOST
                if s != PointState.LOST:
                    trail_points.append(point.trajectory[f])
        
        if len(trail_points) >= 2:
            for j in range(len(trail_points) - 1):
                alpha = (j + 1) / len(trail_points)
                fade_color = tuple(int(c * alpha) for c in color)
                pt1 = tuple(trail_points[j].astype(int))
                pt2 = tuple(trail_points[j + 1].astype(int))
                cv2.line(frame, pt1, pt2, fade_color, 2)
        
        # Draw point with appropriate marker
        pt = tuple(pos.astype(int))
        
        # Check if this is a newly initialized point (within first 10 frames of birth)
        is_new = (frame_idx - point.birth_frame) < 10 and point.birth_frame > 0
        
        if state == PointState.ACTIVE:
            if is_new:
                # Draw X marker for new points
                size = 6
                cv2.line(frame, (pt[0] - size, pt[1] - size), (pt[0] + size, pt[1] + size), 
                        color, 2)
                cv2.line(frame, (pt[0] + size, pt[1] - size), (pt[0] - size, pt[1] + size), 
                        color, 2)
            else:
                # Solid circle for active points
                cv2.circle(frame, pt, 4, color, -1)
                cv2.circle(frame, pt, 5, (255, 255, 255), 1)
        
        elif state == PointState.OCCLUDED:
            # Hollow circle for occluded points (outline only)
            cv2.circle(frame, pt, 5, color, 2)
            # Add small dot in center to distinguish
            cv2.circle(frame, pt, 1, color, -1)


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description='Long-term point tracking with disocclusion handling.'
    )
    parser.add_argument('--video', '-v', default='data/bear01', help='Input video path')
    parser.add_argument('--output', '-o', default=None, help='Output video path')
    parser.add_argument('--points', '-n', type=int, default=100, help='Initial points')
    parser.add_argument('--fb-threshold', type=float, default=1.0, help='FB threshold')
    parser.add_argument('--no-new-points', action='store_true', help='Disable new point initialization')
    parser.add_argument('--comparison', '-c', action='store_true', help='Create side-by-side comparison video')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')
    
    args = parser.parse_args()
    
    if not HAS_CV2:
        print("Error: OpenCV is required.")
        exit(1)
    
    video_name = Path(args.video).name
    
    if args.comparison:
        # Create comparison video
        if args.output is None:
            args.output = f'outputs/trajectories/{video_name}_comparison.mp4'
        create_comparison_video(
            video_path=args.video,
            output_path=args.output,
            num_points=args.points,
            fb_threshold=args.fb_threshold,
            verbose=not args.quiet
        )
    else:
        # Create single tracker video
        if args.output is None:
            suffix = '_tracker' if not args.no_new_points else '_tracker_static'
            args.output = f'outputs/trajectories/{video_name}{suffix}.mp4'
        create_tracker_video(
            video_path=args.video,
            output_path=args.output,
            num_points=args.points,
            fb_threshold=args.fb_threshold,
            add_new_points=not args.no_new_points,
            verbose=not args.quiet
        )
