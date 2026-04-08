"""
Optical flow estimation for point tracking.

This module implements Lucas-Kanade optical flow for sparse point tracking,
as used in the Ochs et al. long-term point tracking pipeline.

The Lucas-Kanade algorithm assumes brightness constancy within a local window:
    I(x, y, t) ≈ I(x + dx, y + dy, t+1)

The displacement is computed by solving a least-squares system using the
structure tensor (second moment matrix) of image gradients.
"""

from typing import Tuple, Optional, List
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def compute_image_pyramid(
    image: np.ndarray,
    max_level: int,
    sigma: float = 1.0
) -> List[np.ndarray]:
    """
    Compute Gaussian image pyramid.
    
    Args:
        image: Input image (H, W) grayscale.
        max_level: Number of pyramid levels (0 = original only).
        sigma: Gaussian smoothing before downsampling.
        
    Returns:
        List of images from finest to coarsest [level_0, level_1, ...].
    """
    pyramid = [image.astype(np.float64)]
    
    for level in range(max_level):
        # Smooth current level
        if HAS_SCIPY:
            smoothed = gaussian_filter(pyramid[-1], sigma=sigma)
        else:
            smoothed = pyramid[-1]  # No smoothing fallback
        
        # Downsample by factor of 2
        downsampled = smoothed[::2, ::2]
        pyramid.append(downsampled)
    
    return pyramid


def compute_spatial_derivatives(
    image: np.ndarray,
    sigma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spatial derivatives Ix and Iy using Sobel filters.
    
    Args:
        image: Input image (H, W) grayscale.
        sigma: Optional Gaussian smoothing before derivatives.
        
    Returns:
        Tuple of (Ix, Iy) derivatives.
    """
    if HAS_SCIPY:
        if sigma > 0:
            smoothed = gaussian_filter(image.astype(np.float64), sigma=sigma)
        else:
            smoothed = image.astype(np.float64)
        
        # Sobel derivatives
        Ix = ndimage.sobel(smoothed, axis=1, mode='reflect') / 8.0
        Iy = ndimage.sobel(smoothed, axis=0, mode='reflect') / 8.0
    else:
        # Simple finite differences fallback
        img = image.astype(np.float64)
        Ix = np.zeros_like(img)
        Iy = np.zeros_like(img)
        Ix[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2.0
        Iy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2.0
    
    return Ix, Iy


def compute_optical_flow_lk(
    I1: np.ndarray,
    I2: np.ndarray,
    point: Tuple[float, float],
    window_size: int = 15,
    max_iterations: int = 20,
    epsilon: float = 0.01
) -> Tuple[Optional[np.ndarray], float]:
    """
    Compute optical flow for a single point using iterative Lucas-Kanade.
    
    This implements the standard iterative LK algorithm:
    1. Warp image I2 with current displacement estimate
    2. Compute gradient and error
    3. Solve for displacement update
    4. Iterate until convergence
    
    Args:
        I1: First image (reference) as grayscale float array.
        I2: Second image (target) as grayscale float array.
        point: (x, y) coordinates of the point in I1.
        window_size: Size of the integration window.
        max_iterations: Maximum number of iterations.
        epsilon: Convergence threshold for displacement change.
        
    Returns:
        Tuple of (displacement, error) where:
            displacement: (dx, dy) or None if tracking failed
            error: Sum of squared differences in the window
    """
    x, y = point
    half_win = window_size // 2
    
    H, W = I1.shape
    
    # Check if point is too close to border
    if x < half_win or x >= W - half_win or y < half_win or y >= H - half_win:
        return None, float('inf')
    
    # Extract window from I1
    y_start, y_end = int(y) - half_win, int(y) + half_win + 1
    x_start, x_end = int(x) - half_win, int(x) + half_win + 1
    
    patch1 = I1[y_start:y_end, x_start:x_end].astype(np.float64)
    
    # Compute spatial derivatives of I1
    Ix, Iy = compute_spatial_derivatives(I1, sigma=0.5)
    
    # Extract derivative windows
    Ix_win = Ix[y_start:y_end, x_start:x_end]
    Iy_win = Iy[y_start:y_end, x_start:x_end]
    
    # Compute structure tensor components (sum over window)
    Axx = np.sum(Ix_win * Ix_win)
    Axy = np.sum(Ix_win * Iy_win)
    Ayy = np.sum(Iy_win * Iy_win)
    
    # Check if structure tensor is invertible (det > threshold)
    det = Axx * Ayy - Axy * Axy
    if det < 1e-6:
        return None, float('inf')
    
    # Inverse of structure tensor
    inv_det = 1.0 / det
    inv_Axx = Ayy * inv_det
    inv_Axy = -Axy * inv_det
    inv_Ayy = Axx * inv_det
    
    # Initialize displacement
    dx, dy = 0.0, 0.0
    
    for iteration in range(max_iterations):
        # Current position in I2
        x2 = x + dx
        y2 = y + dy
        
        # Check bounds
        if x2 < half_win or x2 >= W - half_win or y2 < half_win or y2 >= H - half_win:
            return None, float('inf')
        
        # Extract window from I2 using bilinear interpolation
        patch2 = extract_patch_bilinear(I2, x2, y2, window_size)
        
        if patch2 is None:
            return None, float('inf')
        
        # Compute temporal derivative (error image)
        It = patch2 - patch1
        
        # Compute gradient-weighted error
        bx = -np.sum(Ix_win * It)
        by = -np.sum(Iy_win * It)
        
        # Solve for displacement update: A * delta = b
        delta_x = inv_Axx * bx + inv_Axy * by
        delta_y = inv_Axy * bx + inv_Ayy * by
        
        # Update displacement
        dx += delta_x
        dy += delta_y
        
        # Check convergence
        if delta_x * delta_x + delta_y * delta_y < epsilon * epsilon:
            break
    
    # Compute final error (SSD)
    x2, y2 = x + dx, y + dy
    if x2 < half_win or x2 >= W - half_win or y2 < half_win or y2 >= H - half_win:
        return None, float('inf')
    
    patch2 = extract_patch_bilinear(I2, x2, y2, window_size)
    if patch2 is None:
        return None, float('inf')
    
    error = np.sum((patch2 - patch1) ** 2)
    
    return np.array([dx, dy]), error


def extract_patch_bilinear(
    image: np.ndarray,
    x: float,
    y: float,
    window_size: int
) -> Optional[np.ndarray]:
    """
    Extract a patch from an image using bilinear interpolation.
    
    Args:
        image: Input image (H, W).
        x, y: Center coordinates (can be subpixel).
        window_size: Size of the patch to extract.
        
    Returns:
        Patch of shape (window_size, window_size) or None if out of bounds.
    """
    half_win = window_size // 2
    H, W = image.shape
    
    # Check bounds with margin
    if x - half_win < 0 or x + half_win >= W - 1:
        return None
    if y - half_win < 0 or y + half_win >= H - 1:
        return None
    
    # Create coordinate grid for the patch
    patch = np.zeros((window_size, window_size), dtype=np.float64)
    
    for i in range(window_size):
        for j in range(window_size):
            # Sample position
            px = x - half_win + j
            py = y - half_win + i
            
            # Bilinear interpolation
            x0, y0 = int(px), int(py)
            x1, y1 = x0 + 1, y0 + 1
            
            # Fractional parts
            fx, fy = px - x0, py - y0
            
            # Interpolate
            patch[i, j] = (
                (1 - fx) * (1 - fy) * image[y0, x0] +
                fx * (1 - fy) * image[y0, x1] +
                (1 - fx) * fy * image[y1, x0] +
                fx * fy * image[y1, x1]
            )
    
    return patch


def track_points_lk(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
    points: np.ndarray,
    window_size: int = 15,
    max_level: int = 3,
    max_iterations: int = 20,
    epsilon: float = 0.01,
    min_eigen_threshold: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Track points from frame_prev to frame_curr using Lucas-Kanade with pyramids.
    
    This implements coarse-to-fine tracking:
    1. Start at the coarsest pyramid level
    2. Track at that level, propagate result to next level
    3. Repeat until finest level
    
    Args:
        frame_prev: Previous frame (H, W, 3) or (H, W).
        frame_curr: Current frame (H, W, 3) or (H, W).
        points: Points to track, shape (N, 2) with (x, y) coordinates.
        window_size: Size of the Lucas-Kanade window.
        max_level: Number of pyramid levels (0 = no pyramid).
        max_iterations: Maximum iterations per level.
        epsilon: Convergence threshold.
        min_eigen_threshold: Minimum eigenvalue for valid tracking.
        
    Returns:
        Tuple of (points_next, status, errors) where:
            points_next: Tracked positions, shape (N, 2)
            status: Boolean mask of successfully tracked points, shape (N,)
            errors: Tracking error for each point, shape (N,)
    """
    # Convert to grayscale if needed
    if len(frame_prev.shape) == 3:
        gray_prev = 0.299 * frame_prev[:, :, 0] + 0.587 * frame_prev[:, :, 1] + 0.114 * frame_prev[:, :, 2]
        gray_curr = 0.299 * frame_curr[:, :, 0] + 0.587 * frame_curr[:, :, 1] + 0.114 * frame_curr[:, :, 2]
    else:
        gray_prev = frame_prev.astype(np.float64)
        gray_curr = frame_curr.astype(np.float64)
    
    # Normalize to [0, 1] range for numerical stability
    if gray_prev.max() > 1:
        gray_prev = gray_prev / 255.0
    if gray_curr.max() > 1:
        gray_curr = gray_curr / 255.0
    
    N = len(points)
    points_next = points.copy()
    status = np.zeros(N, dtype=bool)
    errors = np.full(N, float('inf'))
    
    # Build pyramids
    pyramid_prev = compute_image_pyramid(gray_prev, max_level, sigma=1.0)
    pyramid_curr = compute_image_pyramid(gray_curr, max_level, sigma=1.0)
    
    # Track each point
    for i in range(N):
        point = points[i]
        result = _track_point_pyramid(
            pyramid_prev, pyramid_curr,
            point, window_size, max_iterations, epsilon, min_eigen_threshold
        )
        
        if result is not None:
            new_pos, error = result
            points_next[i] = new_pos
            status[i] = True
            errors[i] = error
    
    return points_next, status, errors


def _track_point_pyramid(
    pyramid_prev: List[np.ndarray],
    pyramid_curr: List[np.ndarray],
    point: np.ndarray,
    window_size: int,
    max_iterations: int,
    epsilon: float,
    min_eigen_threshold: float
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Track a single point through the image pyramid.
    
    Args:
        pyramid_prev: Pyramid of previous frame.
        pyramid_curr: Pyramid of current frame.
        point: (x, y) coordinates in finest level.
        window_size: Lucas-Kanade window size.
        max_iterations: Max iterations per level.
        epsilon: Convergence threshold.
        min_eigen_threshold: Minimum eigenvalue for valid tracking.
        
    Returns:
        Tuple of (new_position, error) or None if tracking failed.
    """
    max_level = len(pyramid_prev) - 1
    
    # Initialize displacement at finest level
    dx_total, dy_total = 0.0, 0.0
    
    # Track from coarse to fine
    for level in range(max_level, -1, -1):
        # Scale factor for this level
        scale = 2 ** level
        
        # Get images at this level
        I1 = pyramid_prev[level]
        I2 = pyramid_curr[level]
        
        H, W = I1.shape
        
        # Scale point coordinates to this level
        x = point[0] / scale
        y = point[1] / scale
        
        # Scale total displacement to this level
        dx = dx_total / scale
        dy = dy_total / scale
        
        half_win = window_size // 2
        
        # Check if point is within bounds (with some margin for displacement)
        margin = half_win + 2
        if x < margin or x >= W - margin or y < margin or y >= H - margin:
            return None
        
        # Compute derivatives at this level
        Ix, Iy = compute_spatial_derivatives(I1, sigma=0.5)
        
        # Extract windows from I1
        y_start, y_end = int(y) - half_win, int(y) + half_win + 1
        x_start, x_end = int(x) - half_win, int(x) + half_win + 1
        
        patch1 = I1[y_start:y_end, x_start:x_end]
        Ix_win = Ix[y_start:y_end, x_start:x_end]
        Iy_win = Iy[y_start:y_end, x_start:x_end]
        
        # Compute structure tensor
        Axx = np.sum(Ix_win * Ix_win)
        Axy = np.sum(Ix_win * Iy_win)
        Ayy = np.sum(Iy_win * Iy_win)
        
        # Check eigenvalue threshold
        trace = Axx + Ayy
        det = Axx * Ayy - Axy * Axy
        lambda_min = (trace - np.sqrt(max(0, trace * trace - 4 * det))) / 2
        
        if lambda_min < min_eigen_threshold:
            return None
        
        # Inverse of structure tensor
        if det < 1e-10:
            return None
        
        inv_det = 1.0 / det
        
        # Iterative refinement at this level
        for _ in range(max_iterations):
            # Current position in I2
            x2 = x + dx
            y2 = y + dy
            
            # Check bounds
            if x2 < half_win or x2 >= W - half_win - 1 or y2 < half_win or y2 >= H - half_win - 1:
                return None
            
            # Extract patch from I2 with bilinear interpolation
            patch2 = _extract_patch_bilinear_fast(I2, x2, y2, window_size)
            
            if patch2 is None:
                return None
            
            # Temporal derivative
            It = patch2 - patch1
            
            # Gradient-weighted error
            bx = -np.sum(Ix_win * It)
            by = -np.sum(Iy_win * It)
            
            # Solve for displacement update
            delta_x = (Ayy * bx - Axy * by) * inv_det
            delta_y = (-Axy * bx + Axx * by) * inv_det
            
            dx += delta_x
            dy += delta_y
            
            # Check convergence
            if delta_x * delta_x + delta_y * delta_y < epsilon * epsilon:
                break
        
        # Accumulate displacement (scale back to finest level)
        dx_total = dx * scale
        dy_total = dy * scale
    
    # Final position in original coordinates
    final_x = point[0] + dx_total
    final_y = point[1] + dy_total
    
    # Compute final error
    H, W = pyramid_prev[0].shape
    half_win = window_size // 2
    
    if final_x < half_win or final_x >= W - half_win - 1 or final_y < half_win or final_y >= H - half_win - 1:
        return None
    
    # Extract final patches for error computation
    I1 = pyramid_prev[0]
    I2 = pyramid_curr[0]
    
    patch1 = _extract_patch_bilinear_fast(I1, point[0], point[1], window_size)
    patch2 = _extract_patch_bilinear_fast(I2, final_x, final_y, window_size)
    
    if patch1 is None or patch2 is None:
        return None
    
    error = np.mean((patch2 - patch1) ** 2)
    
    return np.array([final_x, final_y]), error


def _extract_patch_bilinear_fast(
    image: np.ndarray,
    x: float,
    y: float,
    window_size: int
) -> Optional[np.ndarray]:
    """
    Extract a patch using bilinear interpolation (optimized version).
    
    Args:
        image: Input image (H, W).
        x, y: Center coordinates.
        window_size: Size of the patch.
        
    Returns:
        Patch of shape (window_size, window_size) or None if out of bounds.
    """
    H, W = image.shape
    half_win = window_size // 2
    
    # Check bounds
    if x - half_win < 0 or x + half_win >= W - 1:
        return None
    if y - half_win < 0 or y + half_win >= H - 1:
        return None
    
    # Create output patch
    patch = np.zeros((window_size, window_size), dtype=np.float64)
    
    # Precompute coordinate arrays
    xs = x - half_win + np.arange(window_size)
    ys = y - half_win + np.arange(window_size)
    
    for i, py in enumerate(ys):
        y0 = int(py)
        y1 = y0 + 1
        fy = py - y0
        
        for j, px in enumerate(xs):
            x0 = int(px)
            x1 = x0 + 1
            fx = px - x0
            
            # Bilinear interpolation
            patch[i, j] = (
                (1 - fx) * (1 - fy) * image[y0, x0] +
                fx * (1 - fy) * image[y0, x1] +
                (1 - fx) * fy * image[y1, x0] +
                fx * fy * image[y1, x1]
            )
    
    return patch


def track_points_opencv(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
    points: np.ndarray,
    window_size: Tuple[int, int] = (15, 15),
    max_level: int = 3,
    max_iterations: int = 20,
    epsilon: float = 0.01,
    min_eigen_threshold: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Track points using OpenCV's Lucas-Kanade implementation.
    
    This is a wrapper around cv2.calcOpticalFlowPyrLK with proper
    parameter handling and output formatting.
    
    Args:
        frame_prev: Previous frame (H, W, 3) or (H, W).
        frame_curr: Current frame (H, W, 3) or (H, W).
        points: Points to track, shape (N, 2) with (x, y) coordinates.
        window_size: Size of the search window at each pyramid level.
        max_level: 0-based maximal pyramid level number.
        max_iterations: Maximum iterations per pyramid level.
        epsilon: Convergence criteria for iterative search.
        min_eigen_threshold: Minimum eigenvalue of the 2x2 normal matrix.
        
    Returns:
        Tuple of (points_next, status, errors) where:
            points_next: Tracked positions, shape (N, 2)
            status: Boolean mask of successfully tracked points, shape (N,)
            errors: Tracking error for each point, shape (N,)
    """
    if not HAS_CV2:
        raise ImportError("OpenCV (cv2) is required for this function")
    
    # Convert to grayscale if needed
    if len(frame_prev.shape) == 3:
        gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_RGB2GRAY)
        gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_RGB2GRAY)
    else:
        gray_prev = frame_prev
        gray_curr = frame_curr
    
    # Ensure uint8
    if gray_prev.dtype != np.uint8:
        gray_prev = (gray_prev * 255).astype(np.uint8) if gray_prev.max() <= 1 else gray_prev.astype(np.uint8)
    if gray_curr.dtype != np.uint8:
        gray_curr = (gray_curr * 255).astype(np.uint8) if gray_curr.max() <= 1 else gray_curr.astype(np.uint8)
    
    # Format points for OpenCV (need float32, shape (N, 1, 2))
    points_input = points.reshape(-1, 1, 2).astype(np.float32)
    
    # Lucas-Kanade parameters
    lk_params = dict(
        winSize=window_size,
        maxLevel=max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, epsilon),
        minEigThreshold=min_eigen_threshold
    )
    
    # Compute optical flow
    points_next, status_cv, errors_cv = cv2.calcOpticalFlowPyrLK(
        gray_prev, gray_curr, points_input, None, **lk_params
    )
    
    # Reshape outputs
    points_next = points_next.reshape(-1, 2)
    status = status_cv.flatten().astype(bool)
    errors = errors_cv.flatten()
    
    return points_next, status, errors


class OpticalFlowTracker:
    """
    High-level interface for optical flow tracking.
    
    Provides a unified interface for different tracking backends.
    """
    
    def __init__(
        self,
        backend: str = 'opencv',
        window_size: int = 15,
        max_level: int = 3,
        max_iterations: int = 20,
        epsilon: float = 0.01,
        min_eigen_threshold: float = 1e-4
    ):
        """
        Initialize the optical flow tracker.
        
        Args:
            backend: Tracking backend ('opencv' or 'numpy').
            window_size: Size of the Lucas-Kanade window.
            max_level: Number of pyramid levels.
            max_iterations: Maximum iterations per level.
            epsilon: Convergence threshold.
            min_eigen_threshold: Minimum eigenvalue for valid tracking.
        """
        self.backend = backend
        self.window_size = window_size
        self.max_level = max_level
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.min_eigen_threshold = min_eigen_threshold
    
    def track(
        self,
        frame_prev: np.ndarray,
        frame_curr: np.ndarray,
        points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Track points from frame_prev to frame_curr.
        
        Args:
            frame_prev: Previous frame (H, W, 3) or (H, W).
            frame_curr: Current frame (H, W, 3) or (H, W).
            points: Points to track, shape (N, 2) with (x, y) coordinates.
            
        Returns:
            Tuple of (points_next, status, errors) where:
                points_next: Tracked positions, shape (N, 2)
                status: Boolean mask of successfully tracked points, shape (N,)
                errors: Tracking error for each point, shape (N,)
        """
        if self.backend == 'opencv':
            return track_points_opencv(
                frame_prev, frame_curr, points,
                window_size=(self.window_size, self.window_size),
                max_level=self.max_level,
                max_iterations=self.max_iterations,
                epsilon=self.epsilon,
                min_eigen_threshold=self.min_eigen_threshold
            )
        elif self.backend == 'numpy':
            return track_points_lk(
                frame_prev, frame_curr, points,
                window_size=self.window_size,
                max_level=self.max_level,
                max_iterations=self.max_iterations,
                epsilon=self.epsilon,
                min_eigen_threshold=self.min_eigen_threshold
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def track_sequence(
        self,
        frames: List[np.ndarray],
        initial_points: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Track points through a sequence of frames.
        
        Args:
            frames: List of frames.
            initial_points: Initial points to track, shape (N, 2).
            
        Returns:
            Tuple of (trajectories, statuses) where:
                trajectories: List of point positions at each frame
                statuses: List of status masks at each frame
        """
        trajectories = [initial_points.copy()]
        statuses = [np.ones(len(initial_points), dtype=bool)]
        
        current_points = initial_points.copy()
        
        for i in range(1, len(frames)):
            points_next, status, _ = self.track(
                frames[i - 1], frames[i], current_points
            )
            
            trajectories.append(points_next)
            statuses.append(status)
            
            current_points = points_next
        
        return trajectories, statuses
    
    def track_forward_backward(
        self,
        frame_prev: np.ndarray,
        frame_curr: np.ndarray,
        points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Track points with forward-backward consistency check.
        
        This method:
        1. Tracks points forward from frame_prev to frame_curr
        2. Tracks points backward from frame_curr to frame_prev
        3. Computes forward-backward error
        4. Marks points with large FB error as failed (likely occluded)
        
        Args:
            frame_prev: Previous frame (H, W, 3) or (H, W).
            frame_curr: Current frame (H, W, 3) or (H, W).
            points: Points to track, shape (N, 2) with (x, y) coordinates.
            
        Returns:
            Tuple of (points_next, status, fb_error, track_error) where:
                points_next: Tracked positions, shape (N, 2)
                status: Boolean mask of successfully tracked points, shape (N,)
                fb_error: Forward-backward error for each point, shape (N,)
                track_error: Tracking error (SSD) for each point, shape (N,)
        """
        # Forward tracking
        points_forward, status_forward, errors_forward = self.track(
            frame_prev, frame_curr, points
        )
        
        # Backward tracking
        points_backward, status_backward, errors_backward = self.track(
            frame_curr, frame_prev, points_forward
        )
        
        # Compute forward-backward error
        fb_error = np.linalg.norm(points_backward - points, axis=1)
        
        # Point is valid only if both forward and backward tracking succeeded
        # and FB error is small
        status = status_forward & status_backward
        
        return points_forward, status, fb_error, errors_forward


def track_with_forward_backward(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
    points: np.ndarray,
    fb_threshold: float = 1.0,
    window_size: int = 15,
    max_level: int = 3,
    backend: str = 'opencv'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Track points with forward-backward consistency validation.
    
    This is a convenience function that wraps the tracker class.
    
    Args:
        frame_prev: Previous frame.
        frame_curr: Current frame.
        points: Points to track, shape (N, 2).
        fb_threshold: Maximum allowed forward-backward error.
        window_size: Lucas-Kanade window size.
        max_level: Number of pyramid levels.
        backend: Tracking backend ('opencv' or 'numpy').
        
    Returns:
        Tuple of (points_next, status, fb_error) where:
            points_next: Tracked positions, shape (N, 2)
            status: Boolean mask of valid tracks, shape (N,)
            fb_error: Forward-backward error, shape (N,)
    """
    tracker = OpticalFlowTracker(
        backend=backend,
        window_size=window_size,
        max_level=max_level
    )
    
    points_next, status, fb_error, _ = tracker.track_forward_backward(
        frame_prev, frame_curr, points
    )
    
    # Apply FB threshold
    status = status & (fb_error < fb_threshold)
    
    return points_next, status, fb_error


class ForwardBackwardTracker:
    """
    Tracker with forward-backward consistency checking for robust tracking.
    
    This tracker uses bidirectional optical flow to detect and handle:
    - Occlusions (points that become hidden)
    - Tracking failures (drift, aperture problem)
    - Motion boundaries (points near object edges)
    
    Attributes:
        fb_threshold: Maximum forward-backward error for valid tracks.
        tracker: Underlying optical flow tracker.
    """
    
    def __init__(
        self,
        backend: str = 'opencv',
        window_size: int = 15,
        max_level: int = 3,
        fb_threshold: float = 1.0,
        min_eigen_threshold: float = 1e-4
    ):
        """
        Initialize the forward-backward tracker.
        
        Args:
            backend: Tracking backend ('opencv' or 'numpy').
            window_size: Size of the Lucas-Kanade window.
            max_level: Number of pyramid levels.
            fb_threshold: Maximum forward-backward error in pixels.
            min_eigen_threshold: Minimum eigenvalue for valid tracking.
        """
        self.tracker = OpticalFlowTracker(
            backend=backend,
            window_size=window_size,
            max_level=max_level,
            min_eigen_threshold=min_eigen_threshold
        )
        self.fb_threshold = fb_threshold
    
    def track(
        self,
        frame_prev: np.ndarray,
        frame_curr: np.ndarray,
        points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Track points with forward-backward validation.
        
        Args:
            frame_prev: Previous frame.
            frame_curr: Current frame.
            points: Points to track, shape (N, 2).
            
        Returns:
            Tuple of (points_next, status, fb_error, track_error) where:
                points_next: Tracked positions, shape (N, 2)
                status: Boolean mask of valid tracks, shape (N,)
                fb_error: Forward-backward error, shape (N,)
                track_error: SSD tracking error, shape (N,)
        """
        points_next, status, fb_error, track_error = self.tracker.track_forward_backward(
            frame_prev, frame_curr, points
        )
        
        # Apply FB threshold to status
        status = status & (fb_error <= self.fb_threshold)
        
        return points_next, status, fb_error, track_error
    
    def track_sequence(
        self,
        frames: List[np.ndarray],
        initial_points: np.ndarray,
        verbose: bool = False
    ) -> dict:
        """
        Track points through a sequence with FB validation.
        
        Args:
            frames: List of frames.
            initial_points: Initial points to track, shape (N, 2).
            verbose: Print progress information.
            
        Returns:
            Dictionary containing:
                trajectories: List of point positions at each frame
                statuses: List of status masks at each frame
                fb_errors: List of FB errors at each frame
                track_errors: List of tracking errors at each frame
        """
        num_frames = len(frames)
        num_points = len(initial_points)
        
        trajectories = [initial_points.copy()]
        statuses = [np.ones(num_points, dtype=bool)]
        fb_errors = [np.zeros(num_points)]
        track_errors = [np.zeros(num_points)]
        
        current_points = initial_points.copy()
        
        for i in range(1, num_frames):
            points_next, status, fb_err, track_err = self.track(
                frames[i - 1], frames[i], current_points
            )
            
            trajectories.append(points_next.copy())
            statuses.append(status.copy())
            fb_errors.append(fb_err.copy())
            track_errors.append(track_err.copy())
            
            current_points = points_next
            
            if verbose and i % 10 == 0:
                active = np.sum(status)
                avg_fb = fb_err[status].mean() if np.any(status) else 0
                print(f"  Frame {i}/{num_frames}: {active} points, avg FB error: {avg_fb:.2f}")
        
        return {
            'trajectories': trajectories,
            'statuses': statuses,
            'fb_errors': fb_errors,
            'track_errors': track_errors
        }


def compute_fb_statistics(
    fb_errors: List[np.ndarray],
    statuses: List[np.ndarray]
) -> dict:
    """
    Compute statistics about forward-backward errors.
    
    Args:
        fb_errors: List of FB errors at each frame.
        statuses: List of status masks at each frame.
        
    Returns:
        Dictionary with statistics.
    """
    all_fb_errors = []
    for fb_err, status in zip(fb_errors[1:], statuses[1:]):  # Skip first frame
        if np.any(status):
            all_fb_errors.extend(fb_err[status].tolist())
    
    if len(all_fb_errors) == 0:
        return {
            'mean': 0,
            'median': 0,
            'std': 0,
            'max': 0,
            'min': 0
        }
    
    all_fb_errors = np.array(all_fb_errors)
    
    return {
        'mean': float(np.mean(all_fb_errors)),
        'median': float(np.median(all_fb_errors)),
        'std': float(np.std(all_fb_errors)),
        'max': float(np.max(all_fb_errors)),
        'min': float(np.min(all_fb_errors))
    }
