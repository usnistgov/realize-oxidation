import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List
import os
from realize.environment.dissociation_checker import DEFAULT_DISSOCIATION_THRESHOLD as DISSOCIATION_THRESHOLD
import logging
_LOG = logging.getLogger(__name__)


class TrajectoryPlotter:
    """Handles trajectory and evolution plotting."""
    
    def plot_o2_trajectory(self, data: Dict, episode: Optional[int] = None, 
                          save_path: Optional[str] = None) -> None:
        """Plot O2 molecule trajectory in 3D."""
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            if episode is not None and episode in data:
                episodes = [episode]
            else:
                episodes = list(data.keys())[:5]

            colors = plt.cm.tab10(np.linspace(0, 1, len(episodes)))

            for ep, color in zip(episodes, colors):
                if ep not in data:
                    continue

                steps = sorted(data[ep].keys())
                distances = []
                valid_steps = []

                for step in steps:
                    dist = data[ep][step].get('o2_distance')
                    if dist is not None:
                        distances.append(dist)
                        valid_steps.append(step)

                if distances:
                    plt.plot(valid_steps, distances,
                            'o-', color=color, label=f'Episode {ep}')

                ax.scatter(centers[0, 0], centers[0, 1], centers[0, 2],
                          c='green', s=100, marker='^')
                ax.scatter(centers[-1, 0], centers[-1, 1], centers[-1, 2],
                          c='red', s=100, marker='v')

            ax.set_xlabel('X (A)')
            ax.set_ylabel('Y (A)')
            ax.set_zlabel('Z (A)')
            ax.set_title('O2 Molecule Trajectory')
            ax.legend()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                from realize.config.paths import get_results_dir
                filename = f"o2_trajectory_ep{episode}.png" if episode else "o2_trajectory.png"
                save_path = os.path.join(get_results_dir(), filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.close()
            
        except Exception as e:
            _LOG.info(f"Error plotting trajectory: {e}")
            plt.close()
    
    def plot_distance_evolution(self, data: Dict, episode: Optional[int] = None,
                               save_path: Optional[str] = None) -> None:
        """Plot O-O distance evolution."""
        try:
            plt.figure(figsize=(10, 6))

            if episode is not None and episode in data:
                episodes = [episode]
            else:
                episodes = list(data.keys())[:5]

            colors = plt.cm.tab10(np.linspace(0, 1, len(episodes)))

            for ep, color in zip(episodes, colors):
                if ep not in data:
                    continue

                steps = sorted(data[ep].keys())
                distances = []

                for step in steps:
                    dist = data[ep][step].get('o2_distance')
                    if dist is not None:
                        distances.append(dist)

                if distances:
                    plt.plot(steps[:len(distances)], distances,
                            'o-', color=color, label=f'Episode {ep}')

            plt.axhline(y=DISSOCIATION_THRESHOLD, color='red', linestyle='--', alpha=0.7,
                       label='Dissociation Threshold')

            plt.xlabel('Step')
            plt.ylabel('O-O Distance (A)')
            plt.title('O2 Bond Distance Evolution')
            plt.legend()
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                from realize.config.paths import get_results_dir
                filename = f"o2_distance_ep{episode}.png" if episode else "o2_distance.png"
                save_path = os.path.join(get_results_dir(), filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.close()
            
        except Exception as e:
            _LOG.info(f"Error plotting distance evolution: {e}")
            plt.close()
