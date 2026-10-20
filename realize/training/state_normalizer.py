"""Running-statistics state normalization."""
import torch
import numpy as np
from typing import Optional


class RunningMeanStd:
    """
    Running mean/std calculator using Welford's online algorithm.
    
    Numerically stable incremental computation of mean and variance.
    """
    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()):
        """
        Initialize running statistics.
        
        Args:
            epsilon: Small initial count for stability
            shape: Shape of the data (empty for scalars, (D,) for vectors)
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray):
        """
        Update running statistics with new batch.
        
        Args:
            x: New data (can be single sample or batch)
        """
        if x.ndim == 1:
            # Single sample - add batch dimension
            batch_mean = x
            batch_var = np.zeros_like(x)
            batch_count = 1
        else:
            # Batch of samples
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]
        
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """Combine batch statistics with running statistics (parallel algorithm)."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        # Update mean
        new_mean = self.mean + delta * batch_count / total_count
        
        # Update variance (parallel algorithm)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = M2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
    
    @property
    def std(self):
        """Get standard deviation."""
        return np.sqrt(self.var + 1e-8)


class StateNormalizer:
    """
    State normalizer with shape normalization AND statistical normalization.
    
    Dual functionality:
    1. Shape normalization: Ensures consistent [1, feature_dim] or [B, feature_dim]
    2. Statistical normalization: Normalizes by running mean/std (z-score)
    
    Critical for dual backbone:
    - Actor normalizer: Updates statistics as backbone adapts
    - Critic normalizer: Frozen statistics (stable value targets)
    """
    
    def __init__(self, device: torch.device, moving_mean: bool = True, epsilon: float = 1e-8):
        """
        Initialize state normalizer.
        
        Args:
            device: torch.device for tensor operations
            moving_mean: Whether to use running mean/std normalization
            epsilon: Small constant for numerical stability
        """
        self.device = device
        self.moving_mean = moving_mean
        self.epsilon = epsilon
        
        # Running statistics (initialized on first normalize() call)
        self.running_stats = None
        self._frozen = False
        
        # Track initialization
        self._initialized = False
    
    def freeze(self):
        """
        Freeze normalizer - prevent statistics updates.
        
        Critical for dual backbone architecture:
        - Actor normalizer updates as actor backbone adapts
        - Critic normalizer stays frozen (stable value targets)
        
        Without this, critic learns on moving target (drifting normalization).
        """
        self._frozen = True
        print(f"[StateNormalizer] Normalizer frozen (statistics will not update)", flush=True)
    
    def is_frozen(self):
        """Check if normalizer is frozen."""
        return self._frozen
    
    def normalize(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize state using running statistics."""
        if state is None:
            raise RuntimeError(
                "STRICT MODE: Cannot normalize None state.\n"
                "State must be a valid tensor or array.\n"
                "Check that embedding extraction succeeded before normalization."
            )
        
        # Convert to numpy for statistics
        if torch.is_tensor(state):
            state_np = state.detach().cpu().numpy()
        else:
            state_np = np.array(state)
        
        # Update statistics ONLY if not frozen AND not temp frozen
        if not self._frozen and not getattr(self, '_temp_frozen', False):
            self._update_statistics(state_np)
        
        
        # ===== STEP 1: Convert to tensor =====
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, requires_grad=False)
        else:
            # Detach from computation graph (DON'T lose gradients if they exist!)
            # Just ensure we don't track normalization operations
            pass
        
        # Move to correct device
        original_device = state.device
        state = state.to(self.device)
        
        # ===== STEP 2: Shape normalization =====
        state = self._normalize_shape(state)
        
        # ===== STEP 3: Statistical normalization =====
        if self.moving_mean:
            state = self._normalize_statistics(state)
        
        return state
    
    def _normalize_shape(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize state to consistent [1, feature_dim] shape.
        
        Args:
            state: Input state tensor
            
        Returns:
            Reshaped state [1, feature_dim]
        """
        # Handle various input shapes
        if state.dim() == 0:
            # Scalar - shouldn't happen but handle gracefully
            state = state.unsqueeze(0).unsqueeze(0)
        
        elif state.dim() == 1:
            # [feature_dim] -> [1, feature_dim]
            state = state.unsqueeze(0)
        
        elif state.dim() == 2:
            # Already [batch, feature_dim]
            if state.shape[0] != 1:
                # Multiple samples - take first
                state = state[:1]
        
        elif state.dim() == 3:
            # [batch, 1, feature_dim] or [1, channels, feature_dim]
            if state.shape[1] == 1:
                # Squeeze out singleton middle dimension
                state = state.squeeze(1)
            else:
                # Flatten extra dimensions
                state = state.reshape(1, -1)
        
        else:
            # Higher dimensional - flatten all but batch
            state = state.reshape(1, -1)
        
        # Final shape check
        if state.dim() != 2:
            raise ValueError(f"State must be 2D after normalization, got shape: {state.shape}")
        if state.shape[0] != 1:
            raise ValueError(f"Batch size must be 1, got shape: {state.shape}")
        
        return state
    
    def _normalize_statistics(self, state: torch.Tensor) -> torch.Tensor:
        """
        Irreps-aware normalization for E(3)-equivariant embeddings.
        
        Standard z-score (x - mean) / std destroys directional information
        in l>=1 components because mean subtraction removes the consistent
        directional bias that encodes "where the substrate is."
        
        Strategy:
          l=0 (scalars, dims 0:128):   Standard z-score (mean + std)
          l>=1 (vectors/tensors, dims 128:480): Scale by running RMS only (NO mean subtraction)
          
        This preserves the directional structure of l=1 vectors and l=2 tensors
        while keeping all features on comparable scales.
        
        Args:
            state: Shape-normalized state [1, feature_dim]
        Returns:
            Statistically normalized state
        """
        # Convert to numpy for statistics computation
        state_np = state.detach().cpu().numpy()

        # Running stats should already be initialized by _update_statistics
        if self.running_stats is None:
            raise RuntimeError(
                "Running stats not initialized. This should not happen - "
                "_update_statistics() should initialize on first call."
            )

        feature_dim = state_np.shape[-1]
        state_normalized = np.empty_like(state_np)

        mean = self.running_stats.mean
        std = self.running_stats.std

        # Irreps-aware normalization: scale only, no mean subtraction
        # Mean subtraction destroys cross-block correlations that encode
        # site quality (probe showed AUC drops from 1.0 to 0.44 after normalization)
        if feature_dim == 480:
            # Global RMS scaling: preserve all relative structure
            # Just bring features to unit-ish scale without removing any signal
            rms = np.sqrt(np.mean(mean ** 2 + std ** 2) + self.epsilon)
            state_normalized = state_np / rms
            state_tensor = torch.tensor(state_normalized, dtype=torch.float32, device=self.device)
            return state_tensor

        if False and feature_dim == 480:
            l0_end = 128       # 128 × 1 = 128
            l1_end = 320       # 128 + 64 × 3 = 320
            # l2_end = 480     # 320 + 32 × 5 = 480

            # l=0 block: standard z-score (scalars, no directional content)
            state_normalized[..., :l0_end] = (
                (state_np[..., :l0_end] - mean[:l0_end]) / (std[:l0_end] + self.epsilon)
            )

            # l>=1 blocks: scale by RMS only, NO mean subtraction
            # This preserves the directional information (e.g., "substrate is below")
            # RMS = sqrt(mean² + var) = sqrt(mean² + std²)
            for block_start, block_end, vec_dim, n_vecs in [
                (l0_end, l1_end, 3, 64),   # l=1: 64 vectors × 3
                (l1_end, 480, 5, 32),       # l=2: 32 tensors × 5
            ]:
                block = state_np[..., block_start:block_end]
                block_mean = mean[block_start:block_end]
                block_std = std[block_start:block_end]

                # Per-component RMS: sqrt(mean² + std²)
                # This captures both the consistent signal (mean) and fluctuations (std)
                rms = np.sqrt(block_mean ** 2 + block_std ** 2 + self.epsilon)

                state_normalized[..., block_start:block_end] = block / rms
        else:
            # Non-standard feature dim: fall back to uniform z-score
            state_normalized = (state_np - mean) / (std + self.epsilon)

        # Convert back to tensor
        state_tensor = torch.tensor(state_normalized, dtype=torch.float32, device=self.device)
        return state_tensor

    def _update_statistics(self, state_np: np.ndarray):
        """
        Update running statistics with new state data.

        This is called by normalize() when the normalizer is not frozen.

        Args:
            state_np: State as numpy array [batch, features] or [features]
        """
        # Initialize running stats on first call if needed
        if self.running_stats is None:
            # Determine feature dimension
            if state_np.ndim == 1:
                feature_dim = len(state_np)
            else:
                feature_dim = state_np.shape[-1]

            self.running_stats = RunningMeanStd(shape=(feature_dim,))
            self._initialized = True
            print(f"[StateNormalizer] Initialized running stats with feature_dim={feature_dim}", flush=True)

        # Flatten to [samples, features] if needed
        if state_np.ndim == 1:
            state_np = state_np.reshape(1, -1)
        elif state_np.ndim > 2:
            state_np = state_np.reshape(-1, state_np.shape[-1])

        # Update running statistics
        self.running_stats.update(state_np)

    def validate_batch(self, states: torch.Tensor) -> torch.Tensor:
        """
        Validate and normalize batch of states.
        
        Args:
            states: Batch of states [B, ...] 
            
        Returns:
            Normalized batch [B, feature_dim]
        """
        # Shape normalization for batch
        if states.dim() == 3 and states.shape[1] == 1:
            # [B, 1, F] -> [B, F]
            states = states.squeeze(1)
        
        if states.dim() != 2:
            raise ValueError(f"Batch must be 2D, got shape: {states.shape}")
        
        # Statistical normalization for batch
        if self.moving_mean and self.running_stats is not None:
            states_np = states.detach().cpu().numpy()
            
            # Update statistics (skip if frozen)
            if not self._frozen:
                self.running_stats.update(states_np)
            
            # Normalize
            mean = self.running_stats.mean
            std = self.running_stats.std
            
            states_normalized = (states_np - mean) / (std + self.epsilon)
            states = torch.tensor(states_normalized, dtype=torch.float32, device=self.device)
        
        return states
    
    def get_statistics(self):
        """
        Get current normalization statistics.
        
        Returns:
            dict with mean, std, count (or None if not initialized)
        """
        if self.running_stats is None:
            return None
        
        return {
            'mean': self.running_stats.mean.copy(),
            'std': self.running_stats.std.copy(),
            'var': self.running_stats.var.copy(),
            'count': float(self.running_stats.count),
            'frozen': self._frozen
        }
    
    def set_statistics(self, mean: np.ndarray, var: np.ndarray, count: float):
        """
        Manually set normalization statistics.
        
        Used to initialize critic normalizer from actor normalizer.
        
        Args:
            mean: Mean values
            var: Variance values
            count: Sample count
        """
        if self.running_stats is None:
            feature_dim = len(mean)
            self.running_stats = RunningMeanStd(shape=(feature_dim,))
        
        self.running_stats.mean = np.copy(mean)
        self.running_stats.var = np.copy(var)
        self.running_stats.count = count
        
        print(f"[StateNormalizer] Statistics manually set (count={count:.0f})", flush=True)

    def set_frozen_temporarily(self, frozen: bool):
        """
        Temporarily freeze/unfreeze normalizer without losing frozen state.
        
        Used during PPO updates to prevent statistics drift.
        """
        self._temp_frozen = frozen
    
