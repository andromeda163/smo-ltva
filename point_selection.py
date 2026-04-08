"""
Structure tensor computation and interest-point selection for point tracking.

This module implements the smaller eigenvalue filter for selecting good tracking
points, as used in the Ochs et al. long-term point tracking pipeline.

The structure tensor (second moment matrix) captures local image structure:
    M = [Ix²   IxIy]
        [IxIy  Iy² ]

The smaller eigenvalue λ_min indicates the "cornerness" of a point:
- Small λ_min: flat region or edge (poor for tracking)
- Large λ_min: corner or textured region (good for tracking)
"""

from typing import Tuple, Optional
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter, maximum_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def compute_gradients(
    image: np.ndarray,
    sigma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute image gradients using Gaussian derivatives.
    
    Args:
        image: Input image (H, W) or (H, W, C). If multi-channel, converts to grayscale.
        sigma: Standard deviation for Gaussian smoothing.
        
    Returns:
        Tuple of (Ix, Iy) - gradients in x and y directions.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        # Use luminance formula for RGB to grayscale
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        gray = gray.astype(np.float64)
    else:
        gray = image.astype(np.float64)
    
    if HAS_SCIPY:
        # Use Gaussian derivative kernels
        # First smooth, then differentiate
        smoothed = gaussian_filter(gray, sigma=sigma)
        
        # Sobel-like derivative kernels
        Ix = ndimage.sobel(smoothed, axis=1, mode='reflect') / 8.0
        Iy = ndimage.sobel(smoothed, axis=0, mode='reflect') / 8.0
    else:
        # Fallback to simple finite differences
        smoothed = gray
        if sigma > 0:
            # Simple box filter approximation
            kernel_size = int(2 * sigma + 1)
            from numpy.lib.stride_tricks import as_strided
            
        Ix = np.zeros_like(smoothed)
        Iy = np.zeros_like(smoothed)
        
        # Central differences
        Ix[:, 1:-1] = (smoothed[:, 2:] - smoothed[:, :-2]) / 2.0
        Iy[1:-1, :] = (smoothed[2:, :] - smoothed[:-2, :]) / 2.0
    
    return Ix, Iy


def compute_structure_tensor(
    image: np.ndarray,
    sigma_deriv: float = 1.0,
    sigma_integrate: float = 1.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the structure tensor (second moment matrix) for an image.
    
    The structure tensor at each pixel is:
        M = [⟨Ix²⟩   ⟨IxIy⟩]
            [⟨IxIy⟩  ⟨Iy²⟩ ]
    
    where ⟨·⟩ denotes local averaging (Gaussian weighted).
    
    Args:
        image: Input image (H, W) or (H, W, C).
        sigma_deriv: Sigma for gradient computation (smoothing before derivatives).
        sigma_integrate: Sigma for local integration (smoothing the tensor components).
        
    Returns:
        Tuple of (Mxx, Mxy, Myy) - components of the structure tensor.
    """
    # Compute gradients
    Ix, Iy = compute_gradients(image, sigma=sigma_deriv)
    
    # Compute tensor components
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy
    
    # Smooth tensor components (local integration)
    if HAS_SCIPY:
        Mxx = gaussian_filter(Ixx, sigma=sigma_integrate)
        Mxy = gaussian_filter(Ixy, sigma=sigma_integrate)
        Myy = gaussian_filter(Iyy, sigma=sigma_integrate)
    else:
        # Simple box filter fallback
        Mxx = _box_filter(Ixx, sigma_integrate)
        Mxy = _box_filter(Ixy, sigma_integrate)
        Myy = _box_filter(Iyy, sigma_integrate)
    
    return Mxx, Mxy, Myy


def _box_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    """Simple box filter fallback when scipy is not available."""
    size = max(1, int(2 * sigma + 1))
    kernel = np.ones((size, size)) / (size * size)
    
    # Simple convolution using numpy
    from numpy.lib.stride_tricks import as_strided
    
    H, W = image.shape
    padded = np.pad(image, size // 2, mode='reflect')
    
    output = np.zeros_like(image)
    for i in range(size):
        for j in range(size):
            output += kernel[i, j] * padded[i:i+H, j:j+W]
    
    return output


def compute_eigenvalues(
    Mxx: np.ndarray,
    Mxy: np.ndarray,
    Myy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues of the structure tensor.
    
    For a 2x2 symmetric matrix:
        M = [a  b]
            [b  c]
    
    Eigenvalues are:
        λ₁ = (a + c + √((a-c)² + 4b²)) / 2
        λ₂ = (a + c - √((a-c)² + 4b²)) / 2
    
    where λ₁ ≥ λ₂ (λ₂ is the smaller eigenvalue).
    
    Args:
        Mxx, Mxy, Myy: Components of the structure tensor.
        
    Returns:
        Tuple of (lambda_max, lambda_min) - larger and smaller eigenvalues.
    """
    # Trace and determinant
    trace = Mxx + Myy
    diff = Mxx - Myy
    discriminant = np.sqrt(diff * diff + 4 * Mxy * Mxy)
    
    lambda_max = (trace + discriminant) / 2
    lambda_min = (trace - discriminant) / 2
    
    # Ensure non-negative (numerical stability)
    lambda_min = np.maximum(lambda_min, 0)
    lambda_max = np.maximum(lambda_max, lambda_min)
    
    return lambda_max, lambda_min


def compute_corner_response(
    image: np.ndarray,
    sigma_deriv: float = 1.0,
    sigma_integrate: float = 1.5
) -> np.ndarray:
    """
    Compute corner response using the smaller eigenvalue of the structure tensor.
    
    This is the key measure for interest-point selection:
    - Large λ_min indicates a corner or textured region (good for tracking)
    - Small λ_min indicates an edge or flat region (poor for tracking)
    
    Args:
        image: Input image (H, W) or (H, W, C).
        sigma_deriv: Sigma for gradient computation.
        sigma_integrate: Sigma for local integration.
        
    Returns:
        Corner response map (smaller eigenvalue at each pixel).
    """
    Mxx, Mxy, Myy = compute_structure_tensor(image, sigma_deriv, sigma_integrate)
    _, lambda_min = compute_eigenvalues(Mxx, Mxy, Myy)
    return lambda_min


def non_maximum_suppression(
    response: np.ndarray,
    window_size: int = 7,
    threshold: Optional[float] = None
) -> np.ndarray:
    """
    Apply non-maximum suppression to get local maxima.
    
    Args:
        response: Response map (e.g., corner response).
        window_size: Size of the suppression window.
        threshold: Minimum response value. If None, uses adaptive threshold.
        
    Returns:
        Binary mask with True at local maxima positions.
    """
    if HAS_SCIPY:
        # Use maximum filter for efficient non-maximum suppression
        local_max = maximum_filter(response, size=window_size, mode='reflect')
    else:
        # Slower implementation without scipy
        local_max = _maximum_filter_simple(response, window_size)
    
    # A point is a local maximum if it equals the local maximum
    is_local_max = (response == local_max)
    
    # Apply threshold
    if threshold is not None:
        is_local_max = is_local_max & (response >= threshold)
    else:
        # Adaptive threshold: at least 10% of the maximum response
        threshold = 0.1 * response.max()
        is_local_max = is_local_max & (response >= threshold)
    
    return is_local_max


def _maximum_filter_simple(image: np.ndarray, size: int) -> np.ndarray:
    """Simple maximum filter fallback when scipy is not available."""
    H, W = image.shape
    half = size // 2
    padded = np.pad(image, half, mode='reflect')
    
    output = np.zeros_like(image)
    for i in range(size):
        for j in range(size):
            output = np.maximum(output, padded[i:i+H, j:j+W])
    
    return output


def select_grid_points(
    image: np.ndarray,
    grid_spacing: int = 10,
    min_corner_response: float = 1.0,
    sigma_deriv: float = 1.0,
    sigma_integrate: float = 1.5,
    border: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select tracking points on a regular grid, choosing the best point in each cell.
    
    This implements the grid-based point selection strategy:
    1. Divide image into grid cells
    2. In each cell, find the point with highest corner response
    3. Keep only points above the minimum threshold
    
    Args:
        image: Input image (H, W) or (H, W, C).
        grid_spacing: Spacing between grid cells (in pixels).
        min_corner_response: Minimum corner response for a valid point.
        sigma_deriv: Sigma for gradient computation.
        sigma_integrate: Sigma for local integration.
        border: Border margin to exclude points near image edges.
        
    Returns:
        Tuple of (points, responses) where:
            points: Array of shape (N, 2) with (x, y) coordinates.
            responses: Array of shape (N,) with corner response values.
    """
    H, W = image.shape[:2]
    
    # Compute corner response
    response = compute_corner_response(image, sigma_deriv, sigma_integrate)
    
    points = []
    responses = []
    
    # Create grid cells
    for y_start in range(border, H - border, grid_spacing):
        for x_start in range(border, W - border, grid_spacing):
            y_end = min(y_start + grid_spacing, H - border)
            x_end = min(x_start + grid_spacing, W - border)
            
            # Find maximum in this cell
            cell_response = response[y_start:y_end, x_start:x_end]
            
            if cell_response.size == 0:
                continue
            
            max_idx = np.argmax(cell_response)
            local_y, local_x = np.unravel_index(max_idx, cell_response.shape)
            
            # Global coordinates
            y = y_start + local_y
            x = x_start + local_x
            
            # Check threshold
            if response[y, x] >= min_corner_response:
                points.append([x, y])  # Note: (x, y) order for consistency
                responses.append(response[y, x])
    
    if len(points) == 0:
        return np.array([]).reshape(0, 2), np.array([])
    
    return np.array(points), np.array(responses)


def select_best_points(
    image: np.ndarray,
    num_points: int,
    min_distance: int = 5,
    sigma_deriv: float = 1.0,
    sigma_integrate: float = 1.5,
    border: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select the best N tracking points using adaptive non-maximum suppression.
    
    This implements a greedy selection:
    1. Compute corner response
    2. Find global maximum
    3. Suppress nearby points
    4. Repeat until N points selected
    
    Args:
        image: Input image (H, W) or (H, W, C).
        num_points: Number of points to select.
        min_distance: Minimum distance between selected points.
        sigma_deriv: Sigma for gradient computation.
        sigma_integrate: Sigma for local integration.
        border: Border margin to exclude points near image edges.
        
    Returns:
        Tuple of (points, responses) where:
            points: Array of shape (N, 2) with (x, y) coordinates.
            responses: Array of shape (N,) with corner response values.
    """
    H, W = image.shape[:2]
    
    # Compute corner response
    response = compute_corner_response(image, sigma_deriv, sigma_integrate)
    
    # Create mask for valid region
    valid_mask = np.ones_like(response, dtype=bool)
    valid_mask[:border, :] = False
    valid_mask[-border:, :] = False
    valid_mask[:, :border] = False
    valid_mask[:, -border:] = False
    
    # Apply mask to response
    masked_response = response.copy()
    masked_response[~valid_mask] = -np.inf
    
    points = []
    responses = []
    
    for _ in range(num_points):
        # Find global maximum
        idx = np.argmax(masked_response)
        y, x = np.unravel_index(idx, masked_response.shape)
        
        # Check if valid
        if masked_response[y, x] <= 0:
            break
        
        points.append([x, y])
        responses.append(response[y, x])
        
        # Suppress nearby points
        y_min = max(0, y - min_distance)
        y_max = min(H, y + min_distance + 1)
        x_min = max(0, x - min_distance)
        x_max = min(W, x + min_distance + 1)
        masked_response[y_min:y_max, x_min:x_max] = -np.inf
    
    if len(points) == 0:
        return np.array([]).reshape(0, 2), np.array([])
    
    return np.array(points), np.array(responses)


def detect_good_features_to_track(
    image: np.ndarray,
    max_corners: int = 1000,
    quality_level: float = 0.01,
    min_distance: int = 10,
    block_size: int = 7,
    use_harris: bool = False,
    k: float = 0.04
) -> np.ndarray:
    """
    Detect good features to track using OpenCV's goodFeaturesToTrack.
    
    This is a wrapper around OpenCV's implementation, which uses the
    Shi-Tomasi corner detector (or Harris if use_harris=True).
    
    Args:
        image: Input image (H, W) or (H, W, C).
        max_corners: Maximum number of corners to return.
        quality_level: Minimum accepted quality of corners (relative to best).
        min_distance: Minimum Euclidean distance between corners.
        block_size: Size of averaging block for computing derivative covariation.
        use_harris: If True, use Harris detector; otherwise Shi-Tomasi.
        k: Harris detector free parameter.
        
    Returns:
        Array of shape (N, 2) with (x, y) coordinates.
    """
    if not HAS_CV2:
        raise ImportError("OpenCV (cv2) is required for this function")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Ensure uint8
    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8) if gray.max() <= 1 else gray.astype(np.uint8)
    
    # Detect corners
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size,
        useHarrisDetector=use_harris,
        k=k
    )
    
    if corners is None:
        return np.array([]).reshape(0, 2)
    
    # Convert to (N, 2) format with (x, y) coordinates
    points = corners.reshape(-1, 2)
    
    return points


class PointSelector:
    """
    High-level interface for point selection.
    
    Provides multiple strategies for selecting tracking points.
    """
    
    def __init__(
        self,
        strategy: str = 'grid',
        grid_spacing: int = 10,
        min_corner_response: float = 1.0,
        min_distance: int = 5,
        border: int = 5,
        sigma_deriv: float = 1.0,
        sigma_integrate: float = 1.5
    ):
        """
        Initialize the point selector.
        
        Args:
            strategy: Selection strategy ('grid', 'best', 'opencv').
            grid_spacing: Spacing for grid-based selection.
            min_corner_response: Minimum corner response threshold.
            min_distance: Minimum distance between points.
            border: Border margin.
            sigma_deriv: Sigma for gradient computation.
            sigma_integrate: Sigma for local integration.
        """
        self.strategy = strategy
        self.grid_spacing = grid_spacing
        self.min_corner_response = min_corner_response
        self.min_distance = min_distance
        self.border = border
        self.sigma_deriv = sigma_deriv
        self.sigma_integrate = sigma_integrate
    
    def select(
        self,
        image: np.ndarray,
        num_points: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select tracking points from an image.
        
        Args:
            image: Input image (H, W) or (H, W, C).
            num_points: Number of points (for 'best' and 'opencv' strategies).
            
        Returns:
            Tuple of (points, responses) where:
                points: Array of shape (N, 2) with (x, y) coordinates.
                responses: Array of shape (N,) with corner response values.
        """
        if self.strategy == 'grid':
            return select_grid_points(
                image,
                grid_spacing=self.grid_spacing,
                min_corner_response=self.min_corner_response,
                sigma_deriv=self.sigma_deriv,
                sigma_integrate=self.sigma_integrate,
                border=self.border
            )
        elif self.strategy == 'best':
            if num_points is None:
                num_points = 1000
            return select_best_points(
                image,
                num_points=num_points,
                min_distance=self.min_distance,
                sigma_deriv=self.sigma_deriv,
                sigma_integrate=self.sigma_integrate,
                border=self.border
            )
        elif self.strategy == 'opencv':
            if not HAS_CV2:
                raise ImportError("OpenCV is required for 'opencv' strategy")
            if num_points is None:
                num_points = 1000
            points = detect_good_features_to_track(
                image,
                max_corners=num_points,
                min_distance=self.min_distance
            )
            # Compute responses for the selected points
            response = compute_corner_response(
                image,
                self.sigma_deriv,
                self.sigma_integrate
            )
            responses = response[points[:, 1].astype(int), points[:, 0].astype(int)]
            return points, responses
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def compute_response_map(self, image: np.ndarray) -> np.ndarray:
        """
        Compute the corner response map for visualization.
        
        Args:
            image: Input image (H, W) or (H, W, C).
            
        Returns:
            Corner response map.
        """
        return compute_corner_response(
            image,
            self.sigma_deriv,
            self.sigma_integrate
        )
