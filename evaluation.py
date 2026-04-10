"""
Evaluation module for tracking and segmentation quality assessment.

This module implements evaluation protocols for the FBMS dataset, including:
- Trajectory-to-region assignment based on ground truth
- Consistency tracking across annotated frames
- Coverage and survival metrics

The evaluation approach measures how well trajectories maintain their
association with the ground truth regions they were initially assigned to.
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TrajectoryEvaluation:
    """Evaluation results for a single trajectory."""
    trajectory_id: int
    assigned_region: int           # Region assigned at first annotated frame (-1 if unassigned)
    birth_frame: int               # Frame when trajectory started
    death_frame: int               # Last frame with valid position (-1 if still active)
    total_frames: int              # Total frames trajectory exists
    valid_frames: int = 0          # Frames with valid (in-bounds) position at GT frames
    consistent_frames: int = 0     # Frames on assigned region at GT frames
    consistency: float = 0.0       # consistent_frames / valid_frames
    survived: bool = False         # Reached last annotated frame


@dataclass
class RegionEvaluation:
    """Evaluation summary for a ground truth region."""
    region_id: int
    num_trajectories: int = 0      # Trajectories assigned to this region
    total_pixels: int = 0          # Total pixels in this region
    covered_pixels: int = 0        # Pixels with at least one trajectory
    coverage: float = 0.0          # covered_pixels / total_pixels
    mean_consistency: float = 0.0  # Average consistency for trajectories in this region


@dataclass
class FrameMetrics:
    """Metrics for a single annotated frame."""
    frame_idx: int
    num_active_trajectories: int = 0
    num_consistent: int = 0
    num_inconsistent: int = 0
    num_occluded: int = 0
    num_unassigned: int = 0
    consistency_rate: float = 0.0


@dataclass
class TrackingEvaluationResult:
    """Complete evaluation results for tracking quality."""
    # Overall metrics
    num_trajectories: int
    num_assigned: int              # Trajectories assigned to a valid region
    num_unassigned: int            # Trajectories starting on unlabeled area
    mean_consistency: float        # Average consistency across assigned trajectories
    survival_rate: float           # Fraction surviving to last annotated frame
    overall_coverage: float        # Overall GT coverage by trajectories
    
    # Per-trajectory details
    trajectories: List[TrajectoryEvaluation]
    
    # Per-region summary
    regions: Dict[int, RegionEvaluation]
    
    # Frame-by-frame breakdown (only for annotated frames)
    frame_metrics: Dict[int, FrameMetrics]


class TrajectoryEvaluator:
    """
    Evaluator for tracking quality using ground truth regions.
    
    This evaluator measures how consistently trajectories stay on the
    ground truth regions they were initially assigned to.
    
    The evaluation process:
    1. Assign each trajectory to the GT region at its position in the first annotated frame
    2. Check consistency at each subsequent annotated frame
    3. Compute per-trajectory, per-region, and overall metrics
    """
    
    def __init__(self, min_valid_frames: int = 1):
        """
        Initialize the evaluator.
        
        Args:
            min_valid_frames: Minimum number of valid frames for a trajectory to be
                              included in consistency metrics (default: 1)
        """
        self.min_valid_frames = min_valid_frames
    
    def _get_position_at_frame(
        self,
        trajectory_positions: Dict[int, np.ndarray],  # frame_idx -> position
        frame_idx: int,
        frame_shape: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """
        Get the position of a trajectory at a specific frame.
        
        Args:
            trajectory_positions: Dict mapping frame_idx to (x, y) position
            frame_idx: Frame to get position at
            frame_shape: (H, W) shape of the frame for bounds checking
            
        Returns:
            Tuple of (y, x) as integer coordinates, or None if invalid/out of bounds
        """
        if frame_idx not in trajectory_positions:
            return None
        
        pos = trajectory_positions[frame_idx]
        x, y = pos[0], pos[1]
        
        # Check bounds
        H, W = frame_shape
        if x < 0 or x >= W or y < 0 or y >= H:
            return None
        
        return (int(y), int(x))
    
    def _get_region_at_position(
        self,
        gt_mask: np.ndarray,
        position: Tuple[int, int]
    ) -> int:
        """
        Get the ground truth region at a position.
        
        Args:
            gt_mask: Ground truth mask with region IDs (H, W)
            position: (y, x) position
            
        Returns:
            Region ID, or -1 if unlabeled
        """
        y, x = position
        return int(gt_mask[y, x])
    
    def _extract_trajectory_positions(
        self,
        trajectory
    ) -> Dict[int, np.ndarray]:
        """
        Extract positions from a trajectory object.
        
        Handles both TrackedPoint objects from point_tracker.py and
        simple dict/list formats.
        
        Args:
            trajectory: Trajectory object with positions over time
            
        Returns:
            Dict mapping frame_idx to (x, y) position array
        """
        positions = {}
        
        # Check if it's a TrackedPoint object
        if hasattr(trajectory, 'trajectory') and hasattr(trajectory, 'birth_frame'):
            birth_frame = trajectory.birth_frame
            for i, pos in enumerate(trajectory.trajectory):
                frame_idx = birth_frame + i
                positions[frame_idx] = pos
        # Check if it's a dict with frame keys
        elif isinstance(trajectory, dict):
            if 'positions' in trajectory:
                # Format: {'positions': {frame_idx: (x, y), ...}}
                positions = trajectory['positions']
            elif 'trajectory' in trajectory:
                # Format: {'trajectory': [(x, y), ...], 'birth_frame': int}
                birth_frame = trajectory.get('birth_frame', 0)
                for i, pos in enumerate(trajectory['trajectory']):
                    positions[birth_frame + i] = np.array(pos)
        # Check if it's a list/array of positions starting at frame 0
        elif isinstance(trajectory, (list, np.ndarray)):
            for i, pos in enumerate(trajectory):
                if pos is not None and not (isinstance(pos, float) and np.isnan(pos)):
                    positions[i] = np.array(pos)
        
        return positions
    
    def assign_trajectories_to_regions(
        self,
        trajectories: List,
        gt_mask: np.ndarray,
        frame_idx: int = 0,
        frame_shape: Optional[Tuple[int, int]] = None
    ) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        """
        Assign each trajectory to a ground truth region based on its position.
        
        This is Step 1 of the evaluation: determine which region each trajectory
        belongs to at the first annotated frame.
        
        Args:
            trajectories: List of trajectory objects
            gt_mask: Ground truth mask with region IDs, shape (H, W)
            frame_idx: Frame index to use for assignment (default: 0)
            frame_shape: Optional frame shape for bounds checking.
                        If None, uses gt_mask.shape.
        
        Returns:
            Tuple of:
                - traj_to_region: Dict mapping trajectory index to region ID
                - region_to_trajectories: Dict mapping region ID to list of trajectory indices
        """
        if frame_shape is None:
            frame_shape = gt_mask.shape
        
        traj_to_region: Dict[int, int] = {}
        region_to_trajectories: Dict[int, List[int]] = {}
        
        for traj_idx, trajectory in enumerate(trajectories):
            # Extract positions
            positions = self._extract_trajectory_positions(trajectory)
            
            # Get position at assignment frame
            pos = self._get_position_at_frame(positions, frame_idx, frame_shape)
            
            if pos is None:
                # Trajectory doesn't exist at this frame
                region_id = -1
            else:
                # Look up region at this position
                region_id = self._get_region_at_position(gt_mask, pos)
            
            traj_to_region[traj_idx] = region_id
            
            if region_id not in region_to_trajectories:
                region_to_trajectories[region_id] = []
            region_to_trajectories[region_id].append(traj_idx)
        
        return traj_to_region, region_to_trajectories
    
    def evaluate_trajectory_consistency(
        self,
        trajectory,
        assigned_region: int,
        gt_masks: Dict[int, np.ndarray],
        frame_shape: Tuple[int, int]
    ) -> TrajectoryEvaluation:
        """
        Evaluate a single trajectory's consistency with its assigned region.
        
        Args:
            trajectory: Trajectory object
            assigned_region: Region ID assigned at first frame (-1 if unassigned)
            gt_masks: Dict mapping frame_idx to GT mask
            frame_shape: (H, W) shape for bounds checking
            
        Returns:
            TrajectoryEvaluation with detailed metrics
        """
        # Extract positions
        positions = self._extract_trajectory_positions(trajectory)
        
        if len(positions) == 0:
            return TrajectoryEvaluation(
                trajectory_id=-1,
                assigned_region=assigned_region,
                birth_frame=-1,
                death_frame=-1,
                total_frames=0
            )
        
        # Determine trajectory ID and birth/death frames
        frame_indices = sorted(positions.keys())
        birth_frame = frame_indices[0]
        death_frame = frame_indices[-1]
        total_frames = len(frame_indices)
        
        # Get trajectory ID if available
        traj_id = getattr(trajectory, 'point_id', -1)
        if traj_id == -1 and isinstance(trajectory, dict):
            traj_id = trajectory.get('id', -1)
        
        # For unassigned trajectories, return early
        if assigned_region < 0:
            return TrajectoryEvaluation(
                trajectory_id=traj_id,
                assigned_region=assigned_region,
                birth_frame=birth_frame,
                death_frame=death_frame,
                total_frames=total_frames
            )
        
        # Check consistency at each annotated frame
        annotated_frames = sorted(gt_masks.keys())
        valid_frames = 0
        consistent_frames = 0
        
        for frame_idx in annotated_frames:
            # Get position at this frame
            pos = self._get_position_at_frame(positions, frame_idx, frame_shape)
            
            if pos is None:
                # Trajectory doesn't exist at this frame (not yet born or already dead)
                continue
            
            # Check if position is valid (in bounds)
            gt_mask = gt_masks[frame_idx]
            y, x = pos
            
            if y < 0 or y >= gt_mask.shape[0] or x < 0 or x >= gt_mask.shape[1]:
                # Out of bounds
                continue
            
            valid_frames += 1
            
            # Check consistency
            current_region = self._get_region_at_position(gt_mask, pos)
            if current_region == assigned_region:
                consistent_frames += 1
        
        # Compute metrics
        consistency = consistent_frames / valid_frames if valid_frames > 0 else 0.0
        
        # Check survival (reached last annotated frame with valid position)
        last_annotated = max(annotated_frames)
        survived = self._get_position_at_frame(positions, last_annotated, frame_shape) is not None
        
        return TrajectoryEvaluation(
            trajectory_id=traj_id,
            assigned_region=assigned_region,
            birth_frame=birth_frame,
            death_frame=death_frame,
            total_frames=total_frames,
            valid_frames=valid_frames,
            consistent_frames=consistent_frames,
            consistency=consistency,
            survived=survived
        )
    
    def evaluate(
        self,
        trajectories: List,
        ground_truth,
        frame_shape: Optional[Tuple[int, int]] = None
    ) -> TrackingEvaluationResult:
        """
        Perform complete evaluation of tracking quality.
        
        Args:
            trajectories: List of trajectory objects (from PointTracker)
            ground_truth: GroundTruth object from video_loader
            frame_shape: Optional frame shape. If None, uses first GT mask shape.
            
        Returns:
            TrackingEvaluationResult with complete metrics
        """
        # Get annotated frames and load GT masks
        annotated_frames = sorted(ground_truth.frame_indices)
        if len(annotated_frames) == 0:
            raise ValueError("Ground truth has no annotated frames")
        
        # Load all GT masks
        gt_masks: Dict[int, np.ndarray] = {}
        for frame_idx in annotated_frames:
            gt_mask = ground_truth.load_ground_truth(frame_idx)
            if gt_mask is None:
                raise ValueError(f"Failed to load GT for frame {frame_idx}")
            gt_masks[frame_idx] = gt_mask
        
        # Determine frame shape
        if frame_shape is None:
            frame_shape = gt_masks[annotated_frames[0]].shape
        
        # Step 1: Assign trajectories to regions at first annotated frame
        first_frame = annotated_frames[0]
        traj_to_region, region_to_trajectories = self.assign_trajectories_to_regions(
            trajectories,
            gt_masks[first_frame],
            frame_idx=first_frame,
            frame_shape=frame_shape
        )
        
        # Step 2: Evaluate each trajectory
        trajectory_evaluations: List[TrajectoryEvaluation] = []
        for traj_idx, trajectory in enumerate(trajectories):
            assigned_region = traj_to_region[traj_idx]
            eval_result = self.evaluate_trajectory_consistency(
                trajectory,
                assigned_region,
                gt_masks,
                frame_shape
            )
            eval_result.trajectory_id = traj_idx  # Ensure consistent ID
            trajectory_evaluations.append(eval_result)
        
        # Step 3: Compute per-region metrics
        region_evaluations: Dict[int, RegionEvaluation] = {}
        
        for region_id in range(ground_truth.num_regions):
            region_eval = RegionEvaluation(region_id=region_id)
            
            # Get trajectories assigned to this region
            region_traj_indices = region_to_trajectories.get(region_id, [])
            region_eval.num_trajectories = len(region_traj_indices)
            
            # Count pixels in this region (from first GT mask)
            first_gt = gt_masks[first_frame]
            region_pixels = np.sum(first_gt == region_id)
            region_eval.total_pixels = int(region_pixels)
            
            # Count covered pixels (pixels with at least one trajectory)
            covered_positions = set()
            for traj_idx in region_traj_indices:
                positions = self._extract_trajectory_positions(trajectories[traj_idx])
                if first_frame in positions:
                    pos = positions[first_frame]
                    covered_positions.add((int(pos[1]), int(pos[0])))  # (y, x)
            
            region_eval.covered_pixels = len(covered_positions)
            region_eval.coverage = (
                region_eval.covered_pixels / region_eval.total_pixels
                if region_eval.total_pixels > 0 else 0.0
            )
            
            # Compute mean consistency for trajectories in this region
            consistencies = [
                trajectory_evaluations[i].consistency
                for i in region_traj_indices
                if trajectory_evaluations[i].valid_frames >= self.min_valid_frames
            ]
            region_eval.mean_consistency = np.mean(consistencies) if consistencies else 0.0
            
            region_evaluations[region_id] = region_eval
        
        # Also handle unassigned trajectories (region_id = -1)
        unassigned_indices = region_to_trajectories.get(-1, [])
        
        # Step 4: Compute frame-by-frame metrics
        frame_metrics: Dict[int, FrameMetrics] = {}
        
        for frame_idx in annotated_frames:
            metrics = FrameMetrics(frame_idx=frame_idx)
            gt_mask = gt_masks[frame_idx]
            
            for traj_idx, trajectory in enumerate(trajectories):
                positions = self._extract_trajectory_positions(trajectory)
                pos = self._get_position_at_frame(positions, frame_idx, frame_shape)
                
                if pos is None:
                    # Trajectory doesn't exist at this frame
                    continue
                
                metrics.num_active_trajectories += 1
                assigned_region = traj_to_region[traj_idx]
                
                if assigned_region < 0:
                    metrics.num_unassigned += 1
                else:
                    current_region = self._get_region_at_position(gt_mask, pos)
                    if current_region == assigned_region:
                        metrics.num_consistent += 1
                    else:
                        metrics.num_inconsistent += 1
            
            # Compute consistency rate for this frame
            total_assigned = metrics.num_consistent + metrics.num_inconsistent
            metrics.consistency_rate = (
                metrics.num_consistent / total_assigned
                if total_assigned > 0 else 0.0
            )
            
            frame_metrics[frame_idx] = metrics
        
        # Step 5: Compute overall metrics
        num_trajectories = len(trajectories)
        num_assigned = sum(1 for r in traj_to_region.values() if r >= 0)
        num_unassigned = num_trajectories - num_assigned
        
        # Mean consistency (only for assigned trajectories with enough valid frames)
        consistencies = [
            e.consistency for e in trajectory_evaluations
            if e.assigned_region >= 0 and e.valid_frames >= self.min_valid_frames
        ]
        mean_consistency = np.mean(consistencies) if consistencies else 0.0
        
        # Survival rate
        survived = sum(1 for e in trajectory_evaluations if e.survived)
        survival_rate = survived / num_trajectories if num_trajectories > 0 else 0.0
        
        # Overall coverage (weighted by region size)
        total_gt_pixels = sum(r.total_pixels for r in region_evaluations.values())
        total_covered = sum(r.covered_pixels for r in region_evaluations.values())
        overall_coverage = total_covered / total_gt_pixels if total_gt_pixels > 0 else 0.0
        
        return TrackingEvaluationResult(
            num_trajectories=num_trajectories,
            num_assigned=num_assigned,
            num_unassigned=num_unassigned,
            mean_consistency=mean_consistency,
            survival_rate=survival_rate,
            overall_coverage=overall_coverage,
            trajectories=trajectory_evaluations,
            regions=region_evaluations,
            frame_metrics=frame_metrics
        )


def evaluate_tracking(
    trajectories: List,
    ground_truth,
    frame_shape: Optional[Tuple[int, int]] = None
) -> TrackingEvaluationResult:
    """
    Convenience function to evaluate tracking quality.
    
    Args:
        trajectories: List of trajectory objects
        ground_truth: GroundTruth object
        frame_shape: Optional frame shape for bounds checking
        
    Returns:
        TrackingEvaluationResult with complete metrics
    """
    evaluator = TrajectoryEvaluator()
    return evaluator.evaluate(trajectories, ground_truth, frame_shape)


def print_evaluation_result(result: TrackingEvaluationResult, verbose: bool = True):
    """
    Print evaluation results in a human-readable format.
    
    Args:
        result: TrackingEvaluationResult to print
        verbose: If True, print per-trajectory details
    """
    print("=" * 60)
    print("TRACKING EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nOverall Metrics:")
    print(f"  Total trajectories: {result.num_trajectories}")
    print(f"  Assigned to region: {result.num_assigned}")
    print(f"  Unassigned: {result.num_unassigned}")
    print(f"  Mean consistency: {result.mean_consistency:.3f}")
    print(f"  Survival rate: {result.survival_rate:.3f}")
    print(f"  Overall coverage: {result.overall_coverage:.3f}")
    
    print(f"\nPer-Region Metrics:")
    for region_id, region_eval in sorted(result.regions.items()):
        print(f"  Region {region_id}:")
        print(f"    Trajectories: {region_eval.num_trajectories}")
        print(f"    Coverage: {region_eval.coverage:.3f} ({region_eval.covered_pixels}/{region_eval.total_pixels} pixels)")
        print(f"    Mean consistency: {region_eval.mean_consistency:.3f}")
    
    print(f"\nPer-Frame Metrics:")
    for frame_idx, metrics in sorted(result.frame_metrics.items()):
        print(f"  Frame {frame_idx}:")
        print(f"    Active: {metrics.num_active_trajectories}")
        print(f"    Consistent: {metrics.num_consistent}")
        print(f"    Inconsistent: {metrics.num_inconsistent}")
        print(f"    Unassigned: {metrics.num_unassigned}")
        print(f"    Consistency rate: {metrics.consistency_rate:.3f}")
    
    if verbose:
        print(f"\nPer-Trajectory Details (first 20):")
        for i, traj_eval in enumerate(result.trajectories[:20]):
            status = "SURVIVED" if traj_eval.survived else "DIED"
            region_str = f"region {traj_eval.assigned_region}" if traj_eval.assigned_region >= 0 else "UNASSIGNED"
            print(f"  Traj {traj_eval.trajectory_id}: {region_str}, "
                  f"consistency={traj_eval.consistency:.2f} ({traj_eval.consistent_frames}/{traj_eval.valid_frames}), "
                  f"{status}")
        
        if len(result.trajectories) > 20:
            print(f"  ... and {len(result.trajectories) - 20} more trajectories")


def visualize_frame_evaluation(
    frame: np.ndarray,
    trajectories: List,
    traj_to_region: Dict[int, int],
    gt_mask: np.ndarray,
    frame_idx: int,
    frame_metrics: Optional[FrameMetrics] = None,
    region_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    show_gt_overlay: bool = True,
    gt_alpha: float = 0.3
) -> np.ndarray:
    """
    Create a visualization of trajectories with evaluation status on a frame.
    
    Args:
        frame: RGB frame as (H, W, 3) array
        trajectories: List of trajectory objects
        traj_to_region: Dict mapping trajectory index to assigned region
        gt_mask: Ground truth mask for this frame
        frame_idx: Current frame index
        frame_metrics: Optional FrameMetrics to display
        region_colors: Optional dict mapping region_id to RGB color
        show_gt_overlay: Whether to overlay GT regions
        gt_alpha: Transparency for GT overlay
        
    Returns:
        Visualization image as (H, W, 3) RGB array
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required for visualization")
    
    H, W = frame.shape[:2]
    vis = frame.copy()
    
    # Default region colors
    if region_colors is None:
        region_colors = {
            0: (200, 200, 200),   # Background: light gray
            1: (0, 255, 255),     # Foreground: cyan
            2: (255, 0, 255),     # Magenta
            3: (255, 255, 0),     # Yellow
            4: (0, 165, 255),     # Orange
            5: (255, 0, 128),     # Rose
        }
    
    # Overlay GT regions
    if show_gt_overlay:
        for region_id, color in region_colors.items():
            mask = gt_mask == region_id
            if np.any(mask):
                for c in range(3):
                    vis[:, :, c][mask] = (
                        (1 - gt_alpha) * vis[:, :, c][mask] + gt_alpha * color[c]
                    )
    
    # Draw trajectories
    evaluator = TrajectoryEvaluator()
    
    for traj_idx, trajectory in enumerate(trajectories):
        positions = evaluator._extract_trajectory_positions(trajectory)
        
        if frame_idx not in positions:
            continue
        
        pos = positions[frame_idx]
        x, y = int(pos[0]), int(pos[1])
        
        # Check bounds
        if x < 0 or x >= W or y < 0 or y >= H:
            continue
        
        assigned_region = traj_to_region.get(traj_idx, -1)
        current_region = int(gt_mask[y, x])
        
        # Determine color based on consistency
        if assigned_region < 0:
            # Unassigned: gray
            color = (128, 128, 128)
            marker_type = 'hollow'
        elif current_region == assigned_region:
            # Consistent: green
            color = (0, 255, 0)
            marker_type = 'solid'
        else:
            # Inconsistent: red
            color = (255, 0, 0)
            marker_type = 'cross'
        
        # Draw marker
        if marker_type == 'solid':
            cv2.circle(vis, (x, y), 4, color, -1)
            cv2.circle(vis, (x, y), 5, (255, 255, 255), 1)
        elif marker_type == 'hollow':
            cv2.circle(vis, (x, y), 4, color, 1)
        elif marker_type == 'cross':
            cv2.drawMarker(vis, (x, y), color, cv2.MARKER_CROSS, 8, 2)
    
    # Draw metrics text
    if frame_metrics is not None:
        y_offset = 30
        texts = [
            f"Frame: {frame_idx}",
            f"Active: {frame_metrics.num_active_trajectories}",
            f"Consistent: {frame_metrics.num_consistent}",
            f"Inconsistent: {frame_metrics.num_inconsistent}",
            f"Consistency: {frame_metrics.consistency_rate:.1%}"
        ]
        
        for text in texts:
            cv2.putText(vis, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
    
    return vis


def create_evaluation_video(
    video_path: str,
    result: TrackingEvaluationResult,
    trajectories: List,
    ground_truth,
    output_path: str,
    fps: int = 10,
    show_gt_overlay: bool = True,
    verbose: bool = True
) -> str:
    """
    Create a video showing frame-by-frame evaluation visualization.
    
    Args:
        video_path: Path to video directory
        result: TrackingEvaluationResult from evaluate_tracking()
        trajectories: List of trajectory objects used in evaluation
        ground_truth: GroundTruth object
        output_path: Path for output video file
        fps: Output video FPS
        show_gt_overlay: Whether to overlay GT regions on annotated frames
        verbose: Print progress information
        
    Returns:
        Path to the created video file
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required for video creation")
    
    from video_loader import VideoLoader
    
    # Load video
    video = VideoLoader(video_path)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    H, W = video.frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
    
    if verbose:
        print(f"Creating evaluation video: {video_path}")
        print(f"  Frames: {len(video)}")
    
    # Get trajectory-to-region mapping
    evaluator = TrajectoryEvaluator()
    first_annotated = min(ground_truth.frame_indices)
    first_gt = ground_truth.load_ground_truth(first_annotated)
    traj_to_region, _ = evaluator.assign_trajectories_to_regions(
        trajectories, first_gt, frame_idx=first_annotated
    )
    
    # Region colors
    region_colors = {}
    for region_id in range(ground_truth.num_regions):
        region_colors[region_id] = ground_truth.regions[region_id].color
    
    # Process each frame
    for frame_idx in range(len(video)):
        frame = video[frame_idx]
        
        if ground_truth.has_ground_truth(frame_idx):
            gt_mask = ground_truth.load_ground_truth(frame_idx)
            frame_metrics = result.frame_metrics.get(frame_idx)
            
            vis = visualize_frame_evaluation(
                frame, trajectories, traj_to_region, gt_mask, frame_idx,
                frame_metrics=frame_metrics,
                region_colors=region_colors,
                show_gt_overlay=show_gt_overlay
            )
        else:
            # No GT for this frame - just show trajectories
            vis = frame.copy()
            
            for traj_idx, trajectory in enumerate(trajectories):
                positions = evaluator._extract_trajectory_positions(trajectory)
                
                if frame_idx not in positions:
                    continue
                
                pos = positions[frame_idx]
                x, y = int(pos[0]), int(pos[1])
                
                if x < 0 or x >= W or y < 0 or y >= H:
                    continue
                
                assigned_region = traj_to_region.get(traj_idx, -1)
                
                if assigned_region < 0:
                    color = (128, 128, 128)
                else:
                    color = region_colors.get(assigned_region, (0, 255, 0))
                
                cv2.circle(vis, (x, y), 3, color, -1)
            
            # Show frame number
            cv2.putText(vis, f"Frame: {frame_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Convert RGB to BGR for OpenCV
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        out.write(vis_bgr)
    
    out.release()
    
    if verbose:
        print(f"  Output saved to: {output_path}")
    
    return str(output_path)


def create_evaluation_summary_image(
    result: TrackingEvaluationResult,
    output_path: str,
    video_name: str = "Video"
) -> str:
    """
    Create a summary image with evaluation metrics.
    
    Args:
        result: TrackingEvaluationResult
        output_path: Path for output image
        video_name: Name of the video for display
        
    Returns:
        Path to the created image
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required for visualization")
    
    # Create image
    W, H = 800, 600
    img = np.ones((H, W, 3), dtype=np.uint8) * 30  # Dark background
    
    # Title
    cv2.putText(img, f"Tracking Evaluation: {video_name}", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Overall metrics
    y = 80
    metrics_text = [
        f"Total Trajectories: {result.num_trajectories}",
        f"Assigned: {result.num_assigned}",
        f"Unassigned: {result.num_unassigned}",
        f"",
        f"Mean Consistency: {result.mean_consistency:.1%}",
        f"Survival Rate: {result.survival_rate:.1%}",
        f"Coverage: {result.overall_coverage:.1%}",
    ]
    
    for text in metrics_text:
        cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        y += 30
    
    # Per-region metrics
    y += 20
    cv2.putText(img, "Per-Region Metrics:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    y += 35
    
    for region_id, region_eval in sorted(result.regions.items()):
        text = f"Region {region_id}: {region_eval.num_trajectories} traj, " \
               f"coverage={region_eval.coverage:.1%}, consistency={region_eval.mean_consistency:.1%}"
        cv2.putText(img, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        y += 25
    
    # Per-frame metrics (show as bar chart on right side)
    if len(result.frame_metrics) > 0:
        # Draw consistency rate over time
        chart_x = 400
        chart_y = 100
        chart_w = 350
        chart_h = 200
        
        # Background
        cv2.rectangle(img, (chart_x, chart_y), (chart_x + chart_w, chart_y + chart_h), 
                     (50, 50, 50), -1)
        cv2.putText(img, "Consistency Over Time", (chart_x + 10, chart_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Plot bars
        frames = sorted(result.frame_metrics.keys())
        if len(frames) > 0:
            bar_w = max(5, (chart_w - 20) // len(frames))
            
            for i, frame_idx in enumerate(frames):
                metrics = result.frame_metrics[frame_idx]
                bar_h = int(metrics.consistency_rate * (chart_h - 20))
                x = chart_x + 10 + i * bar_w
                
                # Color based on consistency
                if metrics.consistency_rate >= 0.9:
                    color = (0, 200, 0)  # Green
                elif metrics.consistency_rate >= 0.7:
                    color = (0, 200, 200)  # Yellow
                else:
                    color = (0, 0, 200)  # Red
                
                cv2.rectangle(img, (x, chart_y + chart_h - bar_h - 10),
                             (x + bar_w - 2, chart_y + chart_h - 10), color, -1)
        
        # Y-axis labels
        cv2.putText(img, "100%", (chart_x - 45, chart_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(img, "50%", (chart_x - 40, chart_y + chart_h // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(img, "0%", (chart_x - 35, chart_y + chart_h - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    # Legend
    y = chart_y + chart_h + 50
    cv2.putText(img, "Legend:", (chart_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y += 25
    
    # Consistent marker
    cv2.circle(img, (chart_x + 10, y), 4, (0, 255, 0), -1)
    cv2.putText(img, "Consistent", (chart_x + 25, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    # Inconsistent marker
    y += 25
    cv2.drawMarker(img, (chart_x + 10, y), (255, 0, 0), cv2.MARKER_CROSS, 8, 2)
    cv2.putText(img, "Inconsistent", (chart_x + 25, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    # Unassigned marker
    y += 25
    cv2.circle(img, (chart_x + 10, y), 4, (128, 128, 128), 1)
    cv2.putText(img, "Unassigned", (chart_x + 25, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    # Save image
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    
    return str(output_path)


# Need to import Path for the functions above
from pathlib import Path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate tracking quality against ground truth.'
    )
    parser.add_argument(
        '--video', '-v',
        type=str,
        required=True,
        help='Path to video directory'
    )
    parser.add_argument(
        '--num-points', '-n',
        type=int,
        default=100,
        help='Number of initial points to track (default: 100)'
    )
    parser.add_argument(
        '--fb-threshold',
        type=float,
        default=1.0,
        help='Forward-backward error threshold (default: 1.0)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print per-trajectory details'
    )
    parser.add_argument(
        '--output-video', '-o',
        type=str,
        default=None,
        help='Path for evaluation visualization video'
    )
    parser.add_argument(
        '--output-summary', '-s',
        type=str,
        default=None,
        help='Path for evaluation summary image'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='FPS for output video (default: 10)'
    )
    parser.add_argument(
        '--no-gt-overlay',
        action='store_true',
        help='Do not overlay GT regions on visualization'
    )
    
    args = parser.parse_args()
    
    # Import here to avoid circular imports
    from video_loader import VideoLoader, GroundTruth
    from point_tracker import PointTracker
    
    # Load video and ground truth
    print(f"Loading video: {args.video}")
    video = VideoLoader(args.video)
    gt = GroundTruth(args.video)
    
    print(f"  Frames: {len(video)}")
    print(f"  Annotated frames: {len(gt.annotated_frames)}")
    print(f"  Regions: {gt.num_regions}")
    
    # Run tracking
    print(f"\nRunning tracking with {args.num_points} points...")
    tracker = PointTracker(fb_threshold=args.fb_threshold)
    frames = [video[i] for i in range(len(video))]
    tracking_result = tracker.track_sequence(
        frames,
        initial_points=args.num_points,
        add_new_points=True,
        verbose=False
    )
    
    print(f"  Total trajectories: {tracking_result['total_trajectories']}")
    
    # Evaluate tracking
    print(f"\nEvaluating tracking quality...")
    result = evaluate_tracking(
        list(tracker.points.values()),
        gt,
        frame_shape=video.frame_shape[:2]
    )
    
    # Print results
    print_evaluation_result(result, verbose=args.verbose)
    
    # Create visual outputs if requested
    video_name = Path(args.video).name
    
    if args.output_video:
        create_evaluation_video(
            args.video,
            result,
            list(tracker.points.values()),
            gt,
            args.output_video,
            fps=args.fps,
            show_gt_overlay=not args.no_gt_overlay
        )
    elif args.output_summary or (not args.output_video and not args.output_summary):
        # Default: create both outputs
        output_dir = Path('outputs/evaluation')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_output = output_dir / f'{video_name}_eval_video.mp4'
        summary_output = output_dir / f'{video_name}_eval_summary.png'
        
        create_evaluation_video(
            args.video,
            result,
            list(tracker.points.values()),
            gt,
            str(video_output),
            fps=args.fps,
            show_gt_overlay=not args.no_gt_overlay
        )
        
        create_evaluation_summary_image(
            result,
            str(summary_output),
            video_name=video_name
        )
    
    if args.output_summary:
        create_evaluation_summary_image(
            result,
            args.output_summary,
            video_name=video_name
        )
