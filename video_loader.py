"""
Video loader for the BMF (Berkeley Motion Segmentation) format.

This module provides utilities to load video sequences from the BMF dataset format,
where a .bmf manifest file lists frame filenames and actual images are stored
as JPEG files.
"""

import os
from pathlib import Path
from typing import Iterator, Optional, Tuple, List
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
