#!/usr/bin/env python3
"""
Demo script for point tracking with trajectory visualization.

This script demonstrates the optical flow tracking pipeline:
1. Load a video sequence
2. Select interest points using structure tensor analysis
3. Track points through the video using Lucas-Kanade optical flow
4. Visualize trajectories and save as a video

Usage:
    python demo_trajectories.py [--video VIDEO] [--output OUTPUT] [--points N]

Examples:
    python demo_trajectories.py --video data/bear01 --output outputs/trajectories/bear01_tracks.mp4
    python demo_trajectories.py --video data/cars2 --points 100
    python demo_trajectories.py --video data/horses03 --use-fb --fb-threshold 0.5
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not found. Video output will not be available.")

from video_loader import VideoLoader
from point_selection import select_best_points, PointSelector
from optical_flow import OpticalFlowTracker, ForwardBackwardTracker, compute_fb_statistics


def create_trajectory_video(
    video_path: str,
    output_path: str,
    num_points: int = 100,
    min_distance: int = 20,
    window_size: int = 15,
    max_level: int = 3,
    trail_length: int = 20,
    fps: Optional[int] = None,
    use_fb: bool = False,
    fb_threshold: float = 1.0,
    verbose: bool = True
) -> dict:
    """
    Create a video showing point trajectories.
    
    Args:
        video_path: Path to input video directory.
        output_path: Path for output video file.
        num_points: Number of points to track.
        min_distance: Minimum distance between points.
        window_size: Lucas-Kanade window size.
        max_level: Number of pyramid levels.
        trail_length: Length of trajectory trail to draw.
        fps: Output video FPS. If None, uses input video frame rate.
        use_fb: Use forward-backward consistency checking.
        fb_threshold: Maximum FB error for valid tracks.
        verbose: Print progress information.
        
    Returns:
        Dictionary with tracking statistics.
    """
    # Load video
    video = VideoLoader(video_path)
    num_frames = len(video)
    
    if verbose:
        print(f"Loading video: {video_path}")
        print(f"  Frames: {num_frames}")
        print(f"  Shape: {video.frame_shape}")
        if use_fb:
            print(f"  Using FB validation with threshold: {fb_threshold}")
    
    # Initialize tracker
    if use_fb:
        tracker = ForwardBackwardTracker(
            fb_threshold=fb_threshold,
            window_size=window_size,
            max_level=max_level
        )
    else:
        tracker = OpticalFlowTracker(
            backend='opencv',
            window_size=window_size,
            max_level=max_level
        )
    
    # Select initial points
    frame0 = video[0]
    points, responses = select_best_points(
        frame0,
        num_points=num_points,
        min_distance=min_distance
    )
    
    if verbose:
        print(f"Selected {len(points)} initial points")
    
    # Track through all frames
    if use_fb:
        # Load all frames for FB tracker
        frames = [video[i] for i in range(num_frames)]
        result = tracker.track_sequence(frames, points, verbose=verbose)
        trajectories = result['trajectories']
        statuses = result['statuses']
        fb_errors = result['fb_errors']
    else:
        trajectories = [points.copy()]
        statuses = [np.ones(len(points), dtype=bool)]
        current_points = points.copy()
        
        for i in range(1, num_frames):
            frame_prev = video[i - 1]
            frame_curr = video[i]
            
            points_next, status, _ = tracker.track(frame_prev, frame_curr, current_points)
            
            trajectories.append(points_next.copy())
            statuses.append(status.copy())
            
            current_points = points_next
            
            if verbose and i % 10 == 0:
                active = np.sum(status)
                print(f"  Frame {i}/{num_frames}: {active} points active")
        
        fb_errors = None
    
    # Create output video
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    H, W = video.frame_shape[:2]
    if fps is None:
        fps = 30  # Default FPS
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
    
    if verbose:
        print(f"Creating output video: {output_path}")
    
    # Color palette for trajectories
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(len(points), 3), dtype=np.uint8)
    
    # Process each frame
    for frame_idx in range(num_frames):
        frame = video[frame_idx].copy()
        
        # Draw trajectories up to this frame
        for pt_idx in range(len(points)):
            # Check if this point was active at this frame
            if not statuses[frame_idx][pt_idx]:
                continue
            
            # Draw trail
            start_frame = max(0, frame_idx - trail_length + 1)
            trail_points = []
            
            for f in range(start_frame, frame_idx + 1):
                if statuses[f][pt_idx]:
                    trail_points.append(trajectories[f][pt_idx])
            
            if len(trail_points) >= 2:
                # Draw trajectory line
                color = tuple(int(c) for c in colors[pt_idx])
                for j in range(len(trail_points) - 1):
                    # Fade effect: older points are more transparent
                    alpha = (j + 1) / len(trail_points)
                    fade_color = tuple(int(c * alpha) for c in color)
                    
                    pt1 = tuple(trail_points[j].astype(int))
                    pt2 = tuple(trail_points[j + 1].astype(int))
                    cv2.line(frame, pt1, pt2, fade_color, 2)
            
            # Draw current point
            if len(trail_points) > 0:
                pt = tuple(trail_points[-1].astype(int))
                color = tuple(int(c) for c in colors[pt_idx])
                cv2.circle(frame, pt, 4, color, -1)
                cv2.circle(frame, pt, 5, (255, 255, 255), 1)
        
        # Draw frame info
        info_text = f"Frame: {frame_idx + 1}/{num_frames}"
        active_count = np.sum(statuses[frame_idx])
        info_text2 = f"Active points: {active_count}"
        if use_fb:
            info_text2 += f" (FB threshold: {fb_threshold})"
        
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, info_text2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    
    # Compute statistics
    survived_all = sum(
        1 for pt_idx in range(len(points))
        if all(statuses[f][pt_idx] for f in range(num_frames))
    )
    
    avg_lifetime = np.mean([
        sum(1 for f in range(num_frames) if statuses[f][pt_idx])
        for pt_idx in range(len(points))
    ])
    
    stats = {
        'num_frames': num_frames,
        'num_initial_points': len(points),
        'survived_all_frames': survived_all,
        'survival_rate': survived_all / len(points) * 100,
        'avg_lifetime_frames': avg_lifetime,
        'output_path': str(output_path),
        'use_fb': use_fb,
        'fb_threshold': fb_threshold if use_fb else None
    }
    
    # Add FB statistics if available
    if use_fb and fb_errors is not None:
        fb_stats = compute_fb_statistics(fb_errors, statuses)
        stats['fb_stats'] = fb_stats
    
    if verbose:
        print(f"\nTracking Statistics:")
        print(f"  Initial points: {stats['num_initial_points']}")
        print(f"  Survived all frames: {stats['survived_all_frames']} ({stats['survival_rate']:.1f}%)")
        print(f"  Average lifetime: {stats['avg_lifetime_frames']:.1f} frames")
        if use_fb and 'fb_stats' in stats:
            print(f"  FB error mean: {stats['fb_stats']['mean']:.3f} pixels")
            print(f"  FB error median: {stats['fb_stats']['median']:.3f} pixels")
        print(f"  Output saved to: {stats['output_path']}")
    
    return stats


def main():
    """Main entry point for the demo script."""
    parser = argparse.ArgumentParser(
        description='Track points through a video and visualize trajectories.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo_trajectories.py --video data/bear01
    python demo_trajectories.py --video data/cars2 --points 50 --output outputs/trajectories/cars2.mp4
    python demo_trajectories.py --video data/horses03 --points 200 --trail-length 30
    python demo_trajectories.py --video data/bear01 --use-fb --fb-threshold 0.5
        """
    )
    
    parser.add_argument(
        '--video', '-v',
        type=str,
        default='data/bear01',
        help='Path to input video directory (default: data/bear01)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path for output video file (default: outputs/trajectories/<video_name>_tracks.mp4)'
    )
    
    parser.add_argument(
        '--points', '-n',
        type=int,
        default=100,
        help='Number of points to track (default: 100)'
    )
    
    parser.add_argument(
        '--min-distance', '-d',
        type=int,
        default=20,
        help='Minimum distance between points (default: 20)'
    )
    
    parser.add_argument(
        '--window-size', '-w',
        type=int,
        default=15,
        help='Lucas-Kanade window size (default: 15)'
    )
    
    parser.add_argument(
        '--pyramid-levels', '-l',
        type=int,
        default=3,
        help='Number of pyramid levels (default: 3)'
    )
    
    parser.add_argument(
        '--trail-length', '-t',
        type=int,
        default=20,
        help='Length of trajectory trail to draw (default: 20)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=None,
        help='Output video FPS (default: 30)'
    )
    
    parser.add_argument(
        '--use-fb',
        action='store_true',
        help='Use forward-backward consistency checking for occlusion detection'
    )
    
    parser.add_argument(
        '--fb-threshold',
        type=float,
        default=1.0,
        help='Maximum forward-backward error in pixels (default: 1.0)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output is None:
        video_name = Path(args.video).name
        suffix = '_fb' if args.use_fb else ''
        args.output = f'outputs/trajectories/{video_name}_tracks{suffix}.mp4'
    
    # Run tracking
    stats = create_trajectory_video(
        video_path=args.video,
        output_path=args.output,
        num_points=args.points,
        min_distance=args.min_distance,
        window_size=args.window_size,
        max_level=args.pyramid_levels,
        trail_length=args.trail_length,
        fps=args.fps,
        use_fb=args.use_fb,
        fb_threshold=args.fb_threshold,
        verbose=not args.quiet
    )
    
    return stats


if __name__ == '__main__':
    if not HAS_CV2:
        print("Error: OpenCV is required for video output.")
        print("Install with: pip install opencv-python")
        exit(1)
    
    stats = main()
