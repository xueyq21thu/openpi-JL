import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
import pathlib
import tyro
import dataclasses
from typing import List, Optional, Dict
import re # For parsing filenames
from collections import defaultdict

# (Re-using the VisualizationArgs and load_and_extract_trajectory from previous script,
# but VisualizationArgs will be slightly different for this batch processing script)

@dataclasses.dataclass
class BatchVisualizationArgs:
    """Arguments for batch visualizing robot trajectories."""
    input_npy_directory: pathlib.Path = dataclasses.field(
        metadata=dict(tyro_conf=tyro.conf.arg(name="--input-dir"))
    )
    """Directory containing the .npy episode files (e.g., /workspace/openpi-JL/data/libero/npy/)."""

    output_image_directory: pathlib.Path = dataclasses.field(
        metadata=dict(tyro_conf=tyro.conf.arg(name="--output-dir"))
    )
    """Directory to save the generated trajectory images (e.g., /workspace/openpi-JL/data/libero/image)."""

    filename_pattern: str = r"Episode_(\d+)_(\d+)\.npy"
    """Regex pattern to match and extract Task ID (X) and Episode ID (Y) from filenames."""

    plot_title_prefix: str = "Task"
    """Prefix for the plot titles (e.g., 'Task X Trajectories')."""

    highlight_start_end: bool = True
    """Whether to highlight the start and end points of each trajectory."""

    view_elevation: float = 30.0
    """Elevation angle for the 3D plot view."""
    view_azimuth: float = -60.0
    """Azimuth angle for the 3D plot view."""

    max_trajectories_per_plot: Optional[int] = None # Set to a number to limit, e.g., 10
    """Maximum number of trajectories to plot on a single task image (plots the first N found)."""


def load_and_extract_trajectory(npy_path: pathlib.Path) -> Optional[np.ndarray]:
    """
    Loads an episode from an .npy file and extracts the end-effector trajectory.
    (Identical to the function in the previous answer)
    """
    if not npy_path.exists():
        print(f"Error: File not found at {npy_path}")
        return None
    
    try:
        episode_data = np.load(npy_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading {npy_path}: {e}")
        return None

    if not isinstance(episode_data, (list, np.ndarray)) or len(episode_data) == 0:
        # print(f"Warning: No data or unexpected data format in {npy_path}")
        return None # Less verbose for batch processing

    trajectory_points = []
    for i, step_data in enumerate(episode_data):
        if not isinstance(step_data, dict):
            # print(f"Warning: Step {i} in {npy_path.name} is not a dictionary. Skipping.")
            continue
        
        try:
            state_vector = step_data['observation']['state']
            eef_pos = state_vector[:3]  # x, y, z
            trajectory_points.append(eef_pos)
        except KeyError: # Removed 'as e' for less verbosity
            # print(f"Warning: Missing key in step data for {npy_path.name} (step {i}). Skipping step.")
            continue
        except IndexError:
            # print(f"Warning: State vector too short in {npy_path.name} (step {i}). Skipping step.")
            continue
        except TypeError:
            # print(f"Warning: 'observation' or 'state' is not subscriptable in {npy_path.name} (step {i}). Skipping step.")
            continue

    if not trajectory_points:
        # print(f"Warning: No valid trajectory points extracted from {npy_path.name}")
        return None
        
    return np.array(trajectory_points)


def batch_visualize_trajectories(args: BatchVisualizationArgs):
    """
    Scans a directory for .npy files, groups them by Task ID (X),
    and plots all trajectories for a given X on a single image.
    """
    input_dir = args.input_npy_directory
    output_dir = args.output_image_directory
    
    if not input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' does not exist or is not a directory.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Searching for .npy files in: {input_dir}")
    print(f"Saving images to: {output_dir}")

    # Group files by task ID (X)
    task_files: Dict[str, List[Dict[str, any]]] = defaultdict(list)
    file_pattern_re = re.compile(args.filename_pattern)

    found_files_count = 0
    for npy_file in sorted(input_dir.rglob("*.npy")): # rglob for recursive, or glob for current dir
        match = file_pattern_re.match(npy_file.name)
        if match:
            found_files_count += 1
            task_id_x = match.group(1)
            episode_id_y = match.group(2)
            task_files[task_id_x].append({"path": npy_file, "episode_id": episode_id_y})
        else:
            print(f"Skipping file (does not match pattern '{args.filename_pattern}'): {npy_file.name}")

    if not task_files:
        print(f"No .npy files matching the pattern '{args.filename_pattern}' found in '{input_dir}'.")
        return
    
    print(f"Found {found_files_count} files matching pattern, grouped into {len(task_files)} tasks.")

    for task_id_x, episode_infos in task_files.items():
        print(f"\nProcessing Task ID (X): {task_id_x} ({len(episode_infos)} episodes)")

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        num_episodes_in_task = len(episode_infos)
        # Sort episodes by episode_id_y to make colors consistent if some are missing
        sorted_episode_infos = sorted(episode_infos, key=lambda info: int(info["episode_id"]))

        if args.max_trajectories_per_plot is not None:
            sorted_episode_infos = sorted_episode_infos[:args.max_trajectories_per_plot]
            print(f"  Plotting first {len(sorted_episode_infos)} trajectories due to max_trajectories_per_plot limit.")

        # Use a colormap for distinct colors for different episodes (Y) within the same task (X)
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(sorted_episode_infos))) # 0.9 to avoid very light yellow

        plotted_something_for_task = False
        all_points_for_this_task_plot = []

        for i, episode_info in enumerate(sorted_episode_infos):
            npy_path = episode_info["path"]
            episode_id_y = episode_info["episode_id"]
            # print(f"  Loading trajectory for Episode Y={episode_id_y} from {npy_path.name}")
            
            trajectory = load_and_extract_trajectory(npy_path)

            if trajectory is not None and trajectory.shape[0] > 1:
                plotted_something_for_task = True
                xs, ys, zs = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
                all_points_for_this_task_plot.extend(trajectory) # Store for axis limits

                label = f"Ep. {episode_id_y}"
                ax.plot(xs, ys, zs, label=label, color=colors[i], linewidth=1.5, alpha=0.8)

                if args.highlight_start_end:
                    ax.scatter(xs[0], ys[0], zs[0], color=colors[i], marker='o', s=60) # Start
                    ax.scatter(xs[-1], ys[-1], zs[-1], color=colors[i], marker='X', s=60) # End
            
            elif trajectory is not None and trajectory.shape[0] == 1:
                plotted_something_for_task = True
                all_points_for_this_task_plot.append(trajectory[0])
                # print(f"  Warning: Trajectory in {npy_path.name} has only one point. Plotting as a single point.")
                ax.scatter(trajectory[0,0], trajectory[0,1], trajectory[0,2], label=f"Ep. {episode_id_y} (point)", color=colors[i], marker='s', s=60)
            # else:
                # print(f"  Skipping {npy_path.name} for Task {task_id_x} (no valid trajectory data).")

        if not plotted_something_for_task:
            print(f"  No valid trajectories found to plot for Task ID {task_id_x}.")
            plt.close(fig)  # Close the empty figure
            continue

        ax.set_xlabel("X position (m)")
        ax.set_ylabel("Y position (m)")
        ax.set_zlabel("Z position (m)")
        ax.set_title(f"{args.plot_title_prefix} {task_id_x} Trajectories ({len(sorted_episode_infos)} shown)")
        ax.view_init(elev=args.view_elevation, azim=args.view_azimuth)

        # Set axis limits based on all points plotted in this specific figure
        if all_points_for_this_task_plot:
            all_points_np = np.array(all_points_for_this_task_plot)
            if all_points_np.size > 0:
                min_vals = all_points_np.min(axis=0)
                max_vals = all_points_np.max(axis=0)
                mid_vals = (min_vals + max_vals) / 2
                plot_range_val = (max_vals - min_vals).max() / 2 
                plot_range_val = max(plot_range_val, 0.1) # ensure a minimum range

                ax.set_xlim(mid_vals[0] - plot_range_val, mid_vals[0] + plot_range_val)
                ax.set_ylim(mid_vals[1] - plot_range_val, mid_vals[1] + plot_range_val)
                ax.set_zlim(mid_vals[2] - plot_range_val, mid_vals[2] + plot_range_val)

        if len(sorted_episode_infos) > 0: # Only show legend if there are items
            # Adjust legend box placement if it's too crowded
            legend_max_items = 15 # Heuristic
            if len(sorted_episode_infos) > legend_max_items:
                 ax.legend(title="Episodes", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')
            else:
                 ax.legend(title="Episodes", fontsize='small')


        plt.tight_layout(rect=[0, 0, 0.9, 1] if len(sorted_episode_infos) > legend_max_items else None) # Adjust for external legend

        output_filename = output_dir / f"task_{task_id_x}_trajectories.png"
        plt.savefig(output_filename, dpi=200)
        print(f"  Saved plot to {output_filename}")
        plt.close(fig) # Close the figure to free memory

    print("\nBatch visualization complete.")


if __name__ == "__main__":
    # Example usage:
    # python your_script_name.py --input-dir /workspace/openpi-JL/data/libero/npy/ --output-dir /workspace/openpi-JL/data/libero/image
    
    # You might need to adjust the rglob in `batch_visualize_trajectories`
    # if your npy files are directly in `input_dir` and not in subfolders.
    # If they are in subfolders like `/workspace/openpi-JL/data/libero/npy/libero_spatial_no_noops/Episode_0_21.npy`
    # then `input_dir.rglob("*.npy")` is correct.
    # If they are directly like `/workspace/openpi-JL/data/libero/npy/Episode_0_21.npy`
    # then `input_dir.glob("*.npy")` would be sufficient (though rglob also works).
    
    tyro.cli(batch_visualize_trajectories)