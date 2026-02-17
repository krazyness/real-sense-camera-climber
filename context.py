import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import argparse
import sys

def load_point_cloud(pickle_file):
    """
    Load point cloud from pickle file
    
    Parameters:
    -----------
    pickle_file : str
        Path to pickle file containing point cloud
    
    Returns:
    --------
    points : numpy array
        Point cloud data (Nx2 or Nx3)
    """
    try:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        
        # Handle different data formats
        if isinstance(data, np.ndarray):
            points = data
        elif isinstance(data, dict):
            # Try common keys for point cloud data
            pcd_obj = None
            for key in ['pcd', 'points', 'point_cloud', 'data', 'xyz', 'coords']:
                if key in data:
                    pcd_obj = data[key]
                    break
            else:
                raise ValueError(f"Could not find point cloud data in dict. Keys: {data.keys()}")
            
            # Handle Open3D point cloud objects or similar
            if isinstance(pcd_obj, dict):
                # Nested dictionary - look for point data inside
                print(f"Found nested dict with keys: {pcd_obj.keys()}")
                for nested_key in ['points', 'vertices', 'xyz', 'coords', 'data']:
                    if nested_key in pcd_obj:
                        points = np.asarray(pcd_obj[nested_key])
                        print(f"Using '{nested_key}' from nested dictionary")
                        break
                else:
                    raise ValueError(f"Could not find point data in nested dict. Keys: {pcd_obj.keys()}")
            elif hasattr(pcd_obj, 'points'):
                # Open3D PointCloud object
                points = np.asarray(pcd_obj.points)
            elif hasattr(pcd_obj, 'vertices'):
                points = np.asarray(pcd_obj.vertices)
            elif hasattr(pcd_obj, 'xyz'):
                points = np.asarray(pcd_obj.xyz)
            elif isinstance(pcd_obj, np.ndarray):
                points = pcd_obj
            elif isinstance(pcd_obj, list):
                points = np.array(pcd_obj)
            else:
                raise ValueError(f"Unsupported point cloud object type: {type(pcd_obj)}. "
                               f"Available attributes: {dir(pcd_obj) if hasattr(pcd_obj, '__dir__') else 'N/A'}")
        elif isinstance(data, list):
            points = np.array(data)
        else:
            raise ValueError(f"Unsupported data type in pickle: {type(data)}")
        
        # Ensure it's a numpy array
        points = np.array(points)
        
        # Check shape
        if len(points.shape) != 2:
            raise ValueError(f"Expected 2D array, got shape {points.shape}. "
                           f"Point cloud object type: {type(data.get('pcd', 'N/A') if isinstance(data, dict) else data)}")
        
        if points.shape[1] not in [2, 3]:
            raise ValueError(f"Expected Nx2 or Nx3 points, got shape {points.shape}")
        
        if points.shape[1] == 3:
            print(f"Loaded 3D point cloud with {len(points)} points")
        elif points.shape[1] == 2:
            print(f"Loaded 2D point cloud with {len(points)} points")
        
        return points
        
    except FileNotFoundError:
        print(f"Error: File '{pickle_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        sys.exit(1)

def project_3d_to_2d(points_3d, projection='xy'):
    """
    Project 3D points to 2D plane
    
    Parameters:
    -----------
    points_3d : numpy array (N, 3)
        3D points [x, y, z]
    projection : str
        Projection plane: 'xy', 'xz', 'zy'
    
    Returns:
    --------
    points_2d : numpy array (N, 2)
        Projected 2D points
    proj_indices : tuple
        Indices used for projection
    """
    projection_mapping = {
        'xy': (0, 1),
        'xz': (0, 2),
        'zy': (2, 1),
        'yz': (1, 2),
    }
    
    if projection not in projection_mapping:
        raise ValueError(f"Unknown projection: {projection}")
    
    proj_indices = projection_mapping[projection]
    points_2d = points_3d[:, list(proj_indices)]
    
    return points_2d, proj_indices

def ransac_vertical_line_2d(points_2d, n_iterations=1000, distance_threshold=0.1,
                            angle_tolerance=5, min_inliers=50):
    """
    Detect a single vertical line in 2D using RANSAC
    
    Parameters:
    -----------
    points_2d : numpy array (N, 2)
        2D points [x, y] where y is vertical
    n_iterations : int
        Number of RANSAC iterations
    distance_threshold : float
        Maximum distance for a point to be considered an inlier
    angle_tolerance : float
        Maximum deviation from vertical in degrees
    min_inliers : int
        Minimum number of inliers for a valid line
    
    Returns:
    --------
    line_params : dict or None
        Dictionary with line parameters: 'x', 'y_min', 'y_max', 'inliers', 'angle'
    """
    
    if len(points_2d) < 2:
        return None
    
    best_inliers = []
    best_x = None
    best_angle = None
    
    angle_tolerance_rad = np.deg2rad(angle_tolerance)
    
    for iteration in range(n_iterations):
        # Randomly sample 2 points
        idx = np.random.choice(len(points_2d), 2, replace=False)
        p1, p2 = points_2d[idx]
        
        # Calculate line parameters
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Calculate angle from vertical (vertical is when dx ≈ 0, dy > 0)
        if abs(dx) < 1e-10 and abs(dy) < 1e-10:
            continue
        
        # Angle of line from horizontal
        line_angle = np.arctan2(abs(dy), abs(dx))
        
        # Angle from vertical (90 degrees - line_angle from horizontal)
        angle_from_vertical = abs(np.pi/2 - line_angle)
        
        # Check if close to vertical
        if angle_from_vertical > angle_tolerance_rad:
            continue
        
        # For a vertical line, the x-coordinate should be approximately constant
        # Line equation: x = x0 (vertical line)
        x_line = (p1[0] + p2[0]) / 2
        
        # Find inliers: points whose x-coordinate is close to x_line
        distances = np.abs(points_2d[:, 0] - x_line)
        inlier_mask = distances < distance_threshold
        n_inliers = np.sum(inlier_mask)
        
        # Update best model if this is better
        if n_inliers > len(best_inliers):
            best_inliers = inlier_mask
            best_x = x_line
            best_angle = angle_from_vertical
    
    # Check if we found a valid line
    if len(best_inliers) == 0 or np.sum(best_inliers) < min_inliers:
        return None
    
    # Refine the line using all inliers
    inlier_points = points_2d[best_inliers]
    refined_x = np.mean(inlier_points[:, 0])
    y_min = np.min(inlier_points[:, 1])
    y_max = np.max(inlier_points[:, 1])
    
    return {
        'x': refined_x,
        'y_min': y_min,
        'y_max': y_max,
        'inliers': best_inliers,
        'n_inliers': np.sum(best_inliers),
        'angle_from_vertical': np.rad2deg(best_angle)
    }

def detect_multiple_vertical_lines_2d(points_2d, max_lines=10, n_iterations=1000,
                                     distance_threshold=0.1, angle_tolerance=5,
                                     min_inliers=50, removal_threshold=0.15):
    """
    Detect multiple vertical lines in 2D using sequential RANSAC
    
    Parameters:
    -----------
    points_2d : numpy array (N, 2)
        2D points [x, y] where y is vertical
    max_lines : int
        Maximum number of lines to detect
    n_iterations : int
        Number of RANSAC iterations per line
    distance_threshold : float
        Maximum distance for a point to be considered an inlier
    angle_tolerance : float
        Maximum deviation from vertical in degrees
    min_inliers : int
        Minimum number of inliers for a valid line
    removal_threshold : float
        Distance within which to remove points after detecting a line
    
    Returns:
    --------
    lines : list of dict
        List of detected lines with their parameters
    """
    
    remaining_points = points_2d.copy()
    remaining_indices = np.arange(len(points_2d))
    detected_lines = []
    
    for line_num in range(max_lines):
        if len(remaining_points) < min_inliers:
            break
        
        # Detect one line
        line = ransac_vertical_line_2d(
            remaining_points,
            n_iterations=n_iterations,
            distance_threshold=distance_threshold,
            angle_tolerance=angle_tolerance,
            min_inliers=min_inliers
        )
        
        if line is None:
            break
        
        # Map inliers back to original indices
        original_inliers = remaining_indices[line['inliers']]
        line['original_inliers'] = original_inliers
        
        detected_lines.append(line)
        
        print(f"  Line {line_num + 1}: x={line['x']:.3f}, "
              f"y=[{line['y_min']:.3f}, {line['y_max']:.3f}], "
              f"{line['n_inliers']} inliers, "
              f"angle from vertical: {line['angle_from_vertical']:.2f}°")
        
        # Remove inliers from remaining points
        # Use a slightly larger threshold to ensure we remove nearby points
        distances = np.abs(remaining_points[:, 0] - line['x'])
        to_remove = distances < removal_threshold
        
        remaining_points = remaining_points[~to_remove]
        remaining_indices = remaining_indices[~to_remove]
        
        if len(remaining_points) == 0:
            break
    
    return detected_lines

def detect_vertical_lines_ransac_multi_projection(points_3d, projections=['xy', 'zy'],
                                                  max_lines=10, n_iterations=1000,
                                                  distance_threshold=0.1, angle_tolerance=5,
                                                  min_inliers=50):
    """
    Detect vertical lines using RANSAC on multiple 2D projections of 3D data
    
    Parameters:
    -----------
    points_3d : numpy array (N, 3)
        3D points [x, y, z] where y is vertical
    projections : list of str
        List of projections to use: 'xy', 'zy', 'xz', 'yz'
    (other parameters same as detect_multiple_vertical_lines_2d)
    
    Returns:
    --------
    results : dict
        Results for each projection
    """
    
    results = {}
    
    for proj in projections:
        print(f"\n--- Processing {proj.upper()} projection ---")
        
        # Project to 2D
        points_2d, proj_indices = project_3d_to_2d(points_3d, proj)
        
        # Detect lines
        lines = detect_multiple_vertical_lines_2d(
            points_2d,
            max_lines=max_lines,
            n_iterations=n_iterations,
            distance_threshold=distance_threshold,
            angle_tolerance=angle_tolerance,
            min_inliers=min_inliers
        )
        
        results[proj] = {
            'lines': lines,
            'proj_indices': proj_indices,
            'points_2d': points_2d
        }
    
    return results

def visualize_results_2d(points_2d, lines, title="2D Projection"):
    """Visualize 2D points with detected lines"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all points
    ax.scatter(points_2d[:, 0], points_2d[:, 1], alpha=0.3, s=1, c='lightblue', label='All points')
    
    # Plot each detected line
    colors = plt.cm.rainbow(np.linspace(0, 1, len(lines)))
    
    for i, (line, color) in enumerate(zip(lines, colors)):
        # Plot inliers
        inlier_points = points_2d[line['original_inliers']]
        ax.scatter(inlier_points[:, 0], inlier_points[:, 1], 
                  alpha=0.6, s=3, c=[color], label=f"Line {i+1}")
        
        # Plot the fitted line
        ax.axvline(x=line['x'], color=color, linewidth=2, linestyle='--', alpha=0.7)
        
        # Draw vertical extent
        ax.plot([line['x'], line['x']], [line['y_min'], line['y_max']],
               color=color, linewidth=3, alpha=0.9)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y (Vertical)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.invert_yaxis()
    
    return fig, ax

def visualize_results_3d(points_3d, results_dict):
    """Visualize 3D point cloud with detected lines from multiple projections"""
    
    num_projections = len(results_dict)
    fig = plt.figure(figsize=(18, 6))
    
    # 3D plot
    ax1 = fig.add_subplot(1, num_projections + 1, 1, projection='3d')
    ax1.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
               alpha=0.2, s=1, c='lightblue', label='All points')
    
    # Draw detected lines in 3D
    colors = plt.cm.rainbow(np.linspace(0, 1, 10))
    color_idx = 0
    
    for proj, result in results_dict.items():
        proj_indices = result['proj_indices']
        
        for line in result['lines']:
            color = colors[color_idx % len(colors)]
            color_idx += 1
            
            # Get inlier points in 3D
            inlier_points_3d = points_3d[line['original_inliers']]
            ax1.scatter(inlier_points_3d[:, 0], inlier_points_3d[:, 1], inlier_points_3d[:, 2],
                       alpha=0.6, s=3, c=[color])
            
            # Draw vertical line in 3D
            if proj_indices == (0, 1):  # xy projection
                x_pos = line['x']
                z_pos = np.mean(inlier_points_3d[:, 2])
                ax1.plot([x_pos, x_pos], [line['y_min'], line['y_max']], [z_pos, z_pos],
                        color=color, linewidth=3, alpha=0.9)
            elif proj_indices == (2, 1):  # zy projection
                z_pos = line['x']
                x_pos = np.mean(inlier_points_3d[:, 0])
                ax1.plot([x_pos, x_pos], [line['y_min'], line['y_max']], [z_pos, z_pos],
                        color=color, linewidth=3, alpha=0.9)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y (Vertical)')
    ax1.set_zlabel('Z')
    ax1.set_title(f'3D Point Cloud with Detected Vertical Lines')
    
    # 2D projection plots
    plot_idx = 2
    for proj, result in results_dict.items():
        ax = fig.add_subplot(1, num_projections + 1, plot_idx)
        plot_idx += 1
        
        points_2d = result['points_2d']
        lines = result['lines']
        
        # Plot all points
        ax.scatter(points_2d[:, 0], points_2d[:, 1], alpha=0.3, s=1, c='lightblue')
        
        # Plot detected lines
        color_idx_local = 0
        for line in lines:
            color = colors[color_idx_local % len(colors)]
            color_idx_local += 1
            
            inlier_points = points_2d[line['original_inliers']]
            ax.scatter(inlier_points[:, 0], inlier_points[:, 1],
                      alpha=0.6, s=3, c=[color])
            
            ax.plot([line['x'], line['x']], [line['y_min'], line['y_max']],
                   color=color, linewidth=3, alpha=0.9)
        
        proj_indices = result['proj_indices']
        axis_names = ['X', 'Y', 'Z']
        ax.set_xlabel(axis_names[proj_indices[0]])
        ax.set_ylabel(axis_names[proj_indices[1]])
        ax.set_title(f"{proj.upper()} Projection ({len(lines)} lines)")
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Detect vertical lines using 2D RANSAC on projections'
    )
    parser.add_argument('pickle_file', type=str,
                       help='Path to pickle file containing point cloud')
    parser.add_argument('--projections', type=str, default='xy,zy',
                       help='Comma-separated projections (xy,zy for Y-vertical) (default: xy,zy)')
    parser.add_argument('--max-lines', type=int, default=10,
                       help='Maximum number of lines to detect (default: 10)')
    parser.add_argument('--n-iterations', type=int, default=1000,
                       help='RANSAC iterations per line (default: 1000)')
    parser.add_argument('--distance-threshold', type=float, default=0.1,
                       help='Distance threshold for inliers (default: 0.1)')
    parser.add_argument('--angle-tolerance', type=float, default=5,
                       help='Max angle deviation from vertical in degrees (default: 5)')
    parser.add_argument('--min-inliers', type=int, default=50,
                       help='Minimum inliers per line (default: 50)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip visualization')
    
    args = parser.parse_args()
    
    # Load point cloud
    points = load_point_cloud(args.pickle_file)
    
    print(f"\nPoint cloud statistics:")
    if points.shape[1] == 3:
        print(f"  Number of points: {len(points)}")
        print(f"  X range: [{points[:,0].min():.2f}, {points[:,0].max():.2f}]")
        print(f"  Y range: [{points[:,1].min():.2f}, {points[:,1].max():.2f}]")
        print(f"  Z range: [{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
    else:
        print(f"  Number of points: {len(points)}")
        print(f"  X range: [{points[:,0].min():.2f}, {points[:,0].max():.2f}]")
        print(f"  Y range: [{points[:,1].min():.2f}, {points[:,1].max():.2f}]")
    
    # Parse projections
    projections = [p.strip() for p in args.projections.split(',')]
    
    # Detect vertical lines
    if points.shape[1] == 3:
        results = detect_vertical_lines_ransac_multi_projection(
            points,
            projections=projections,
            max_lines=args.max_lines,
            n_iterations=args.n_iterations,
            distance_threshold=args.distance_threshold,
            angle_tolerance=args.angle_tolerance,
            min_inliers=args.min_inliers
        )
        
        # Summary
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        for proj, result in results.items():
            print(f"{proj.upper()} projection: {len(result['lines'])} lines detected")
        
        # Visualize
        if not args.no_plot:
            visualize_results_3d(points, results)
    
    else:
        # 2D point cloud
        print("\nProcessing 2D point cloud...")
        lines = detect_multiple_vertical_lines_2d(
            points,
            max_lines=args.max_lines,
            n_iterations=args.n_iterations,
            distance_threshold=args.distance_threshold,
            angle_tolerance=args.angle_tolerance,
            min_inliers=args.min_inliers
        )
        
        print(f"\nDetected {len(lines)} vertical lines")
        
        if not args.no_plot and len(lines) > 0:
            visualize_results_2d(points, lines, "2D Point Cloud with Detected Lines")
            plt.show()
    
    print(f"\nDetection complete!")