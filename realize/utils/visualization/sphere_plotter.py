"""Plots spherical-harmonic distributions on the sphere."""
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from e3nn import o3
from e3nn.o3 import Irreps, SphericalHarmonics
from typing import Optional, Union
import logging
_LOG = logging.getLogger(__name__)


class SpherePlotter:
    """Handles spherical harmonics visualization on sphere."""
    
    def __init__(self, lmax: int = 2):
        self.lmax = lmax
        self.irreps_sh = Irreps.spherical_harmonics(lmax)
        self.sh_fn = SphericalHarmonics(self.irreps_sh, normalize=True, normalization='component')
    
    def generate_sphere_points(self, num_pts: int = 5000) -> np.ndarray:
        """Generate evenly distributed points on unit sphere."""
        indices = np.arange(0, num_pts, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / num_pts)
        theta = np.pi * (1 + 5**0.5) * indices
        
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        
        return np.stack([x, y, z], axis=1)
    
    def plot_distribution(self, sh_coeffs: np.ndarray, 
                         temperature: float = 1.0,
                         episode: int = 0, 
                         step: int = 0,
                         save_path: Optional[str] = None,
                         resolution: int = 5000) -> None:
        """Plot the SH distribution on the sphere over `resolution` points.

        sh_coeffs is (sh_dim,) for a single channel or (num_channels, sh_dim).
        """
        try:
            if isinstance(sh_coeffs, torch.Tensor):
                sh_coeffs = sh_coeffs.detach().cpu().numpy()

            sphere_pts = self.generate_sphere_points(resolution)
            sphere_pts_tensor = torch.tensor(sphere_pts, dtype=torch.float32)

            sh_values = self.sh_fn(sphere_pts_tensor).numpy()

            if sh_coeffs.ndim == 1:
                logits = np.dot(sh_values, sh_coeffs) / temperature
                probs = np.exp(logits - np.max(logits))
                probs = probs / np.sum(probs)
                title_suffix = "Single Channel"

            elif sh_coeffs.ndim == 2:
                num_channels = sh_coeffs.shape[0]
                channel_logits = np.zeros((resolution, num_channels))
                
                for ch in range(num_channels):
                    channel_logits[:, ch] = np.dot(sh_values, sh_coeffs[ch]) / temperature
                
                # Log-sum-exp for numerical stability
                max_logits = np.max(channel_logits, axis=1, keepdims=True)
                exp_logits = np.exp(channel_logits - max_logits)
                probs = np.sum(exp_logits, axis=1)
                probs = probs / np.sum(probs)
                title_suffix = f"{num_channels} Channels"
            
            else:
                raise ValueError(f"Invalid sh_coeffs shape for plotting: {sh_coeffs.shape}")

            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            colors = cm.hot(probs / np.max(probs))
            ax.scatter(sphere_pts[:, 0], sphere_pts[:, 1], sphere_pts[:, 2],
                      c=colors, s=1, alpha=0.8)

            self._add_coordinate_axes(ax)

            ax.set_title(f'SH Distribution ({title_suffix})\nEpisode {episode}, Step {step}, T={temperature:.3f}')
            ax.set_box_aspect([1, 1, 1])
            ax.grid(False)

            mappable = cm.ScalarMappable(cmap=cm.hot)
            mappable.set_array(probs)
            cbar = plt.colorbar(mappable, ax=ax, pad=0.1, shrink=0.8)
            cbar.set_label('Probability Density', rotation=270, labelpad=15)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.close()
            
        except Exception as e:
            _LOG.info(f"Error plotting distribution: {e}")
            plt.close()
    
    def _add_coordinate_axes(self, ax, length: float = 1.2):
        """Draw R/G/B x/y/z coordinate axes on the 3D plot."""
        ax.quiver(0, 0, 0, length, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=2)
        ax.quiver(0, 0, 0, 0, length, 0, color='g', arrow_length_ratio=0.1, linewidth=2)
        ax.quiver(0, 0, 0, 0, 0, length, color='b', arrow_length_ratio=0.1, linewidth=2)

        ax.text(length + 0.1, 0, 0, 'X', color='r', fontsize=12)
        ax.text(0, length + 0.1, 0, 'Y', color='g', fontsize=12)
        ax.text(0, 0, length + 0.1, 'Z', color='b', fontsize=12)
