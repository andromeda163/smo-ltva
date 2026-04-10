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
