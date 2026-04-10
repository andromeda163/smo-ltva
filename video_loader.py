"""
Video loader for the BMF (Berkeley Motion Segmentation) format.

This module provides utilities to load video sequences from the BMF dataset format,
where a .bmf manifest file lists frame filenames and actual images are stored
as JPEG files.

Also provides ground truth loading for the FBMS dataset, where segmentation
annotations are provided for selected frames (typically every 20th frame).
"""

import os
from pathlib import Path
from typing import Iterator, Optional, Tuple, List, Dict
from dataclasses import dataclass
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class VideoLoader:
    """
    Loader for BMF format video sequences.
    
    The BMF format consists of:
    - A .bmf manifest file with format:
        Line 1: <num_frames> <num_videos>
        Lines 2+: <frame_filename> (one per frame)
    - Image files (typically .jpg) in the same directory
    
    Attributes:
        path: Path to the video directory or .bmf file
        frames: List of frame filenames from the manifest
        num_frames: Total number of frames
        frame_shape: Shape of frames (height, width, channels)
    """
    
    def __init__(self, path: str):
        """
        Initialize the video loader.
        
        Args:
            path: Path to either:
                - A directory containing a .bmf file and images
                - A .bmf manifest file directly
        """
        self.path = Path(path)
        
        # Determine if path is directory or file
        if self.path.is_dir():
            # Find .bmf file in directory
            bmf_files = list(self.path.glob("*.bmf"))
            if not bmf_files:
                raise FileNotFoundError(f"No .bmf file found in {self.path}")
            self.bmf_path = bmf_files[0]
            self.directory = self.path
        else:
            self.bmf_path = self.path
            self.directory = self.path.parent
        
        # Parse manifest
        self.frames, self.num_frames = self._parse_manifest()
        
        # Cache for frame shape (computed on first load)
        self._frame_shape: Optional[Tuple[int, int, int]] = None
    
    def _parse_manifest(self) -> Tuple[List[str], int]:
        """
        Parse the .bmf manifest file.
        
        Returns:
            Tuple of (list of frame filenames, number of frames)
        """
        with open(self.bmf_path, 'r') as f:
            lines = f.read().strip().split('\n')
        
        # First line: num_frames num_videos
        header = lines[0].split()
        num_frames = int(header[0])
        
        # Remaining lines: frame filenames
        frames = [line.strip() for line in lines[1:] if line.strip()]
        
        if len(frames) != num_frames:
            raise ValueError(
                f"Manifest declares {num_frames} frames but lists {len(frames)}"
            )
        
        return frames, num_frames
    
    def _resolve_frame_path(self, frame_name: str) -> Path:
        """
        Resolve the actual path to a frame file.
        
        The manifest may reference .ppm files but actual files are .jpg.
        This method handles the conversion.
        
        Args:
            frame_name: Filename from the manifest
            
        Returns:
            Path to the actual image file
        """
        # Try the filename as-is first
        direct_path = self.directory / frame_name
        if direct_path.exists():
            return direct_path
        
        # Try replacing .ppm with .jpg (common in this dataset)
        if frame_name.endswith('.ppm'):
            jpg_name = frame_name[:-4] + '.jpg'
            jpg_path = self.directory / jpg_name
            if jpg_path.exists():
                return jpg_path
        
        # Try other common extensions
        base_name = frame_name.rsplit('.', 1)[0]
        for ext in ['.jpg', '.jpeg', '.png', '.ppm', '.pgm']:
            alt_path = self.directory / (base_name + ext)
            if alt_path.exists():
                return alt_path
        
        raise FileNotFoundError(f"Cannot find frame file: {frame_name}")
    
    def _load_image(self, path: Path) -> np.ndarray:
        """
        Load an image file as a numpy array.
        
        Args:
            path: Path to the image file
            
        Returns:
            Image as numpy array with shape (H, W, 3) in RGB format
        """
        if HAS_CV2:
            # OpenCV loads as BGR, convert to RGB
            img = cv2.imread(str(path))
            if img is None:
                raise IOError(f"Failed to load image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif HAS_PIL:
            img = np.array(Image.open(path))
            if len(img.shape) == 2:
                # Grayscale to RGB
                img = np.stack([img, img, img], axis=2)
            elif img.shape[2] == 4:
                # RGBA to RGB
                img = img[:, :, :3]
        else:
            raise ImportError(
                "Either OpenCV (cv2) or PIL is required to load images. "
                "Install with: pip install opencv-python or pip install Pillow"
            )
        
        return img
    
    @property
    def frame_shape(self) -> Tuple[int, int, int]:
        """Get the shape of frames (height, width, channels)."""
        if self._frame_shape is None:
            first_frame = self[0]
            self._frame_shape = first_frame.shape
        return self._frame_shape
    
    def __len__(self) -> int:
        """Return the number of frames."""
        return self.num_frames
    
    def __getitem__(self, idx: int) -> np.ndarray:
        """
        Get a frame by index.
        
        Args:
            idx: Frame index (0-based)
            
        Returns:
            Frame as numpy array with shape (H, W, 3)
        """
        if not 0 <= idx < self.num_frames:
            raise IndexError(
                f"Frame index {idx} out of range [0, {self.num_frames})"
            )
        
        frame_name = self.frames[idx]
        frame_path = self._resolve_frame_path(frame_name)
        return self._load_image(frame_path)
    
    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over all frames."""
        for idx in range(self.num_frames):
            yield self[idx]
    
    def get_frame_path(self, idx: int) -> Path:
        """
        Get the path to a frame file.
        
        Args:
            idx: Frame index (0-based)
            
        Returns:
            Path to the frame file
        """
        if not 0 <= idx < self.num_frames:
            raise IndexError(
                f"Frame index {idx} out of range [0, {self.num_frames})"
            )
        return self._resolve_frame_path(self.frames[idx])
    
    def get_frame_name(self, idx: int) -> str:
        """
        Get the original frame name from the manifest.
        
        Args:
            idx: Frame index (0-based)
            
        Returns:
            Frame filename as listed in manifest
        """
        if not 0 <= idx < self.num_frames:
            raise IndexError(
                f"Frame index {idx} out of range [0, {self.num_frames})"
            )
        return self.frames[idx]
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"VideoLoader(path={self.directory}, "
            f"num_frames={self.num_frames}, "
            f"frame_shape={self._frame_shape})"
        )


class VideoDataset:
    """
    Dataset class for loading multiple video sequences.
    
    Provides access to multiple video sequences from a parent directory.
    """
    
    def __init__(self, root_path: str):
        """
        Initialize the dataset.
        
        Args:
            root_path: Path to directory containing video subdirectories
        """
        self.root_path = Path(root_path)
        self.videos = self._discover_videos()
    
    def _discover_videos(self) -> List[str]:
        """Discover all video sequences in the root directory."""
        videos = []
        for item in sorted(self.root_path.iterdir()):
            if item.is_dir():
                # Check if directory contains a .bmf file
                if list(item.glob("*.bmf")):
                    videos.append(item.name)
        return videos
    
    def __len__(self) -> int:
        """Return number of video sequences."""
        return len(self.videos)
    
    def __getitem__(self, idx: int) -> VideoLoader:
        """
        Get a video loader by index.
        
        Args:
            idx: Video index (0-based)
            
        Returns:
            VideoLoader for the selected video
        """
        if not 0 <= idx < len(self.videos):
            raise IndexError(
                f"Video index {idx} out of range [0, {len(self.videos)})"
            )
        return VideoLoader(self.root_path / self.videos[idx])
    
    def get_video_by_name(self, name: str) -> VideoLoader:
        """
        Get a video loader by name.
        
        Args:
            name: Video directory name
            
        Returns:
            VideoLoader for the selected video
        """
        if name not in self.videos:
            raise KeyError(f"Video '{name}' not found. Available: {self.videos}")
        return VideoLoader(self.root_path / name)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"VideoDataset(root={self.root_path}, videos={self.videos})"


def load_video(path: str) -> VideoLoader:
    """
    Convenience function to load a single video.
    
    Args:
        path: Path to video directory or .bmf file
        
    Returns:
        VideoLoader instance
    """
    return VideoLoader(path)


def load_dataset(root_path: str) -> VideoDataset:
    """
    Convenience function to load a dataset of videos.
    
    Args:
        root_path: Path to directory containing video subdirectories
        
    Returns:
        VideoDataset instance
    """
    return VideoDataset(root_path)


@dataclass
class GroundTruthRegion:
    """Information about a single ground truth region."""
    region_id: int
    color: Tuple[int, int, int]  # RGB color
    scale: int  # Original scale value from Def.dat


@dataclass
class GroundTruthFrame:
    """Ground truth annotation for a single frame."""
    frame_idx: int  # 0-based frame index
    filename: str   # GT filename (e.g., bear01_0001_gt.ppm)
    input_filename: str  # Original frame filename


class GroundTruth:
    """
    Loader for FBMS ground truth annotations.
    
    The FBMS ground truth format consists of:
    - A *Def.dat file with metadata:
        - Number of regions and their color scales
        - List of annotated frames with filenames
    - PPM or PGM image files with color-coded segmentation masks
    
    Each unique color in the GT image corresponds to a different region.
    Region 0 is typically the background.
    
    Attributes:
        num_regions: Number of labeled regions
        regions: Dict mapping region_id to GroundTruthRegion
        annotated_frames: List of GroundTruthFrame entries
        frame_indices: Set of frame indices that have ground truth
    """
    
    def __init__(self, video_path: str):
        """
        Initialize ground truth loader.
        
        Args:
            video_path: Path to video directory (should contain GroundTruth/ subdirectory)
        """
        self.video_path = Path(video_path)
        self.gt_directory = self.video_path / "GroundTruth"
        
        # Find the Def.dat file
        def_files = list(self.gt_directory.glob("*Def.dat"))
        if not def_files:
            raise FileNotFoundError(
                f"No *Def.dat file found in {self.gt_directory}"
            )
        
        self.def_path = def_files[0]
        self.video_name = self.def_path.stem.replace("Def", "")
        
        # Parse definition file
        self.num_regions, self.regions, self.annotated_frames = self._parse_def_file()
        self.frame_indices = {f.frame_idx for f in self.annotated_frames}
        
        # Build color to region mapping
        self._color_to_region: Dict[Tuple[int, int, int], int] = {
            tuple(r.color): r.region_id for r in self.regions.values()
        }
    
    def _parse_def_file(self) -> Tuple[int, Dict[int, GroundTruthRegion], List[GroundTruthFrame]]:
        """
        Parse the Def.dat definition file.
        
        Returns:
            Tuple of (num_regions, regions dict, annotated frames list)
        """
        with open(self.def_path, 'r') as f:
            lines = [l.strip() for l in f.readlines()]
        
        # Parse using line-by-line approach
        i = 0
        regions: Dict[int, GroundTruthRegion] = {}
        annotated_frames: List[GroundTruthFrame] = []
        
        # Skip header
        while i < len(lines) and "Total number of regions" not in lines[i]:
            i += 1
        i += 1
        
        # Number of regions
        num_regions = int(lines[i])
        i += 1
        
        # Parse each region's scale
        for region_id in range(num_regions):
            # Find "Scale of region X:" line
            while i < len(lines) and not lines[i].startswith("Scale of region"):
                i += 1
            i += 1  # Move to the value line
            
            scale = int(lines[i])
            i += 1
            
            # Convert scale to RGB color
            # Scale is either 24-bit RGB (for PPM) or 8-bit grayscale (for PGM)
            if scale > 255:
                # 24-bit RGB: R = (scale >> 16) & 255, G = (scale >> 8) & 255, B = scale & 255
                r = (scale >> 16) & 255
                g = (scale >> 8) & 255
                b = scale & 255
                color = (r, g, b)
            else:
                # 8-bit grayscale: use as all three channels
                color = (scale, scale, scale)
            
            regions[region_id] = GroundTruthRegion(
                region_id=region_id,
                color=color,
                scale=scale
            )
        
        # Find total frames info
        while i < len(lines) and "Total number of frames in this shot" not in lines[i]:
            i += 1
        i += 1
        total_frames = int(lines[i])
        i += 1
        
        # Number of labeled frames
        while i < len(lines) and "Total number of labeled frames" not in lines[i]:
            i += 1
        i += 1
        num_labeled = int(lines[i])
        i += 1
        
        # Parse each labeled frame
        for _ in range(num_labeled):
            # Frame number
            while i < len(lines) and "Frame number:" not in lines[i]:
                i += 1
            i += 1
            frame_idx = int(lines[i])
            i += 1
            
            # File name
            while i < len(lines) and "File name:" not in lines[i]:
                i += 1
            i += 1
            gt_filename = lines[i]
            i += 1
            
            # Input file name
            while i < len(lines) and "Input file name:" not in lines[i]:
                i += 1
            i += 1
            input_filename = lines[i]
            i += 1
            
            annotated_frames.append(GroundTruthFrame(
                frame_idx=frame_idx,
                filename=gt_filename,
                input_filename=input_filename
            ))
        
        return num_regions, regions, annotated_frames
    
    def _resolve_gt_path(self, gt_filename: str) -> Path:
        """
        Resolve the path to a ground truth image file.
        
        Args:
            gt_filename: Filename from Def.dat
            
        Returns:
            Path to the GT file
        """
        # Try PPM first (color), then PGM (grayscale)
        for ext in ['.ppm', '.pgm']:
            # Try exact filename
            path = self.gt_directory / gt_filename
            if path.exists():
                return path
            
            # Try with different extension
            base = gt_filename.rsplit('.', 1)[0]
            path = self.gt_directory / (base + ext)
            if path.exists():
                return path
        
        raise FileNotFoundError(f"Cannot find ground truth file: {gt_filename}")
    
    def _load_gt_image(self, gt_filename: str) -> np.ndarray:
        """
        Load a ground truth image.
        
        Args:
            gt_filename: Filename from Def.dat
            
        Returns:
            Ground truth image as (H, W, 3) RGB array
        """
        path = self._resolve_gt_path(gt_filename)
        
        if HAS_PIL:
            img = np.array(Image.open(path))
        elif HAS_CV2:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"Failed to load GT image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ImportError("PIL or OpenCV required to load images")
        
        # Ensure RGB format
        if len(img.shape) == 2:
            # Grayscale: expand to RGB
            img = np.stack([img, img, img], axis=2)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        
        return img
    
    def has_ground_truth(self, frame_idx: int) -> bool:
        """Check if a frame has ground truth annotation."""
        return frame_idx in self.frame_indices
    
    def get_ground_truth_frame(self, frame_idx: int) -> Optional[GroundTruthFrame]:
        """Get the GroundTruthFrame info for a frame index."""
        for gtf in self.annotated_frames:
            if gtf.frame_idx == frame_idx:
                return gtf
        return None
    
    def load_ground_truth(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Load ground truth segmentation for a specific frame.
        
        Args:
            frame_idx: 0-based frame index
            
        Returns:
            Ground truth as (H, W) array with region IDs, or None if not annotated
        """
        gtf = self.get_ground_truth_frame(frame_idx)
        if gtf is None:
            return None
        
        # Load the GT image
        gt_rgb = self._load_gt_image(gtf.filename)
        
        # Convert RGB to region IDs
        h, w = gt_rgb.shape[:2]
        region_map = np.full((h, w), -1, dtype=np.int32)
        
        for color, region_id in self._color_to_region.items():
            mask = np.all(gt_rgb == np.array(color), axis=2)
            region_map[mask] = region_id
        
        return region_map
    
    def load_ground_truth_rgb(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Load ground truth as RGB image for visualization.
        
        Args:
            frame_idx: 0-based frame index
            
        Returns:
            Ground truth as (H, W, 3) RGB array, or None if not annotated
        """
        gtf = self.get_ground_truth_frame(frame_idx)
        if gtf is None:
            return None
        
        return self._load_gt_image(gtf.filename)
    
    def __repr__(self) -> str:
        return (
            f"GroundTruth(video={self.video_name}, "
            f"regions={self.num_regions}, "
            f"annotated_frames={len(self.annotated_frames)})"
        )


def overlay_ground_truth(
    frame: np.ndarray,
    gt_mask: np.ndarray,
    alpha: float = 0.5,
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Overlay ground truth segmentation on a frame.
    
    Args:
        frame: RGB frame as (H, W, 3) array
        gt_mask: Ground truth as (H, W) array with region IDs
        alpha: Blending factor (0 = original, 1 = full overlay)
        colors: Optional dict mapping region_id to RGB color.
                If None, uses a default color palette.
    
    Returns:
        Blended RGB image as (H, W, 3) array
    """
    if colors is None:
        # Default color palette for regions
        default_colors = [
            (255, 255, 255),   # Region 0: white (usually background)
            (0, 255, 255),     # Region 1: cyan
            (255, 0, 255),     # Region 2: magenta
            (255, 255, 0),     # Region 3: yellow
            (0, 128, 255),     # Region 4: orange
            (128, 0, 255),     # Region 5: purple
            (0, 255, 128),     # Region 6: spring green
            (255, 0, 128),     # Region 7: rose
            (128, 255, 0),     # Region 8: chartreuse
            (0, 128, 128),     # Region 9: teal
        ]
        colors = {i: default_colors[i % len(default_colors)] for i in range(10)}
    
    # Create overlay image
    overlay = frame.copy()
    
    for region_id, color in colors.items():
        mask = gt_mask == region_id
        if np.any(mask):
            for c in range(3):
                overlay[:, :, c][mask] = (
                    (1 - alpha) * frame[:, :, c][mask] + alpha * color[c]
                )
    
    # Mark unlabeled pixels (region_id == -1) with a distinct pattern
    unlabeled = gt_mask == -1
    if np.any(unlabeled):
        overlay[unlabeled] = [128, 128, 128]  # Gray for unlabeled
    
    return overlay.astype(np.uint8)


def create_gt_overlay_video(
    video_path: str,
    output_path: str,
    fps: int = 10,
    alpha: float = 0.4,
    verbose: bool = True
) -> str:
    """
    Create a video showing frames with ground truth overlay.
    
    Args:
        video_path: Path to video directory
        output_path: Path for output video file
        fps: Output video FPS
        alpha: Overlay transparency (0 = no overlay, 1 = full overlay)
        verbose: Print progress information
        
    Returns:
        Path to the created video file
    """
    if not HAS_CV2:
        raise ImportError("OpenCV is required for video creation")
    
    # Load video and ground truth
    video = VideoLoader(video_path)
    gt = GroundTruth(video_path)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    H, W = video.frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
    
    if verbose:
        print(f"Creating GT overlay video: {video_path}")
        print(f"  Frames: {len(video)}")
        print(f"  Annotated frames: {len(gt.annotated_frames)}")
        print(f"  Regions: {gt.num_regions}")
    
    for frame_idx in range(len(video)):
        frame = video[frame_idx]
        
        if gt.has_ground_truth(frame_idx):
            gt_mask = gt.load_ground_truth(frame_idx)
            frame = overlay_ground_truth(frame, gt_mask, alpha=alpha)
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    
    if verbose:
        print(f"  Output saved to: {output_path}")
    
    return str(output_path)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Video loader with ground truth visualization for FBMS dataset.'
    )
    parser.add_argument(
        '--video', '-v',
        type=str,
        required=True,
        help='Path to video directory'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output video path (default: outputs/gt_overlay/<video_name>.mp4)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Output video FPS (default: 10)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.4,
        help='Overlay transparency 0-1 (default: 0.4)'
    )
    parser.add_argument(
        '--info', '-i',
        action='store_true',
        help='Print ground truth info only, do not create video'
    )
    
    args = parser.parse_args()
    
    if not HAS_CV2:
        print("Error: OpenCV is required for video output.")
        print("Install with: pip install opencv-python")
        exit(1)
    
    # Load video and ground truth
    video = VideoLoader(args.video)
    gt = GroundTruth(args.video)
    
    print(f"Video: {args.video}")
    print(f"  Frames: {len(video)}")
    print(f"  Frame shape: {video.frame_shape}")
    print(f"\nGround Truth:")
    print(f"  Regions: {gt.num_regions}")
    
    for region_id, region in gt.regions.items():
        print(f"    Region {region_id}: color={region.color}, scale={region.scale}")
    
    print(f"  Annotated frames ({len(gt.annotated_frames)}):")
    for gtf in gt.annotated_frames:
        print(f"    Frame {gtf.frame_idx}: {gtf.filename}")
    
    if args.info:
        exit(0)
    
    # Determine output path
    if args.output is None:
        video_name = Path(args.video).name
        args.output = f'outputs/gt_overlay/{video_name}_gt_overlay.mp4'
    
    # Create overlay video
    create_gt_overlay_video(
        args.video,
        args.output,
        fps=args.fps,
        alpha=args.alpha
    )
