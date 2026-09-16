import torch
from typing import Tuple

class NoveltyTracker:
    """
    Trajectory diversity tracker via embedding novelty detection.
    
    For equivariant embeddings, normalizes each irrep type independently
    to ensure equal influence in similarity computation.
    """
    
    def __init__(self, threshold: float = 0.85, use_equivariant: bool = False):
        """
        Initialize novelty tracker.
        
        Args:
            threshold: Cosine similarity threshold for novelty [0, 1]
            use_equivariant: Whether input uses equivariant structure
        """
        self.threshold = threshold
        self.use_equivariant = use_equivariant
        self.embedding_dim = None  
        self.seen_states = []

    def check_novelty(self, state: torch.Tensor) -> Tuple[bool, float]:
        """
        Check if state is novel compared to previously seen states.
        
        Args:
            state: Input state tensor (will be flattened and normalized)
            
        Returns:
            (is_novel, max_similarity) where max_similarity  in  [0, 1]
        """
        # Process embedding for comparison
        processed = self._process_embedding(state)
        
        # Compute max similarity with all seen states
        max_similarity = 0.0
        if self.seen_states:
            # Vectorized comparison: [1, D] vs [N, D]
            seen_tensor = torch.stack(self.seen_states)  # [N, D]
            similarities = torch.nn.functional.cosine_similarity(
                processed.unsqueeze(0), seen_tensor, dim=1
            )
            max_similarity = similarities.max().item()
        
        # Determine novelty
        is_novel = max_similarity < self.threshold
        
        # Store if novel
        if is_novel:
            self.seen_states.append(processed)
            
            # Prune if grows unbounded
            if len(self.seen_states) > 2000:
                self._prune()
        
        return is_novel, max_similarity

    def _process_embedding(self, state: torch.Tensor) -> torch.Tensor:
        """
        Process embedding into normalized feature vector.
        
        For equivariant structure, normalize each irrep block (0e, 1e, 2e)
        independently by its own L2 norm. This ensures each irrep type 
        has equal influence in novelty detection, regardless of typical 
        magnitude scale.
        
        Args:
            state: Input tensor, shape [1, F] or [F]
            
        Returns:
            Normalized embedding on CPU, shape [D], unit L2 norm
        """
        flat = state.flatten()

        if self.use_equivariant:
            # Structure 128x0e + 64x1e + 32x2e = 128 + 192 + 160 = 480 dims.
            if len(flat) == 480:
                scalars = flat[:128]           # 0e: scalars
                vectors = flat[128:320]        # 1e: vectors (64x3)
                tensors = flat[320:]           # 2e: tensors (32x5)
                
                # Normalize each irrep block independently
                # This gives equal voice to 0e, 1e, 2e in similarity computation
                eps = 1e-8
                s_norm = scalars / (torch.norm(scalars) + eps)
                v_norm = vectors / (torch.norm(vectors) + eps)
                t_norm = tensors / (torch.norm(tensors) + eps)
                
                # Concatenate normalized blocks
                processed = torch.cat([s_norm, v_norm, t_norm])
            else:
                # Unexpected dimension; fall back to flat normalization
                processed = flat / (torch.norm(flat) + 1e-8)
        else:
            # Non-equivariant: just normalize flat
            processed = flat / (torch.norm(flat) + 1e-8)
        
        # Final L2 normalization for stable cosine similarity
        processed = processed / (torch.norm(processed) + 1e-8)
        
        return processed.cpu().detach()

    def _prune(self):
        """Prune old embeddings to prevent unbounded memory growth."""
        # Keep every other to maintain trajectory diversity
        self.seen_states = self.seen_states[::2]
        
        # If still too many, trim to recent 1000
        if len(self.seen_states) > 1000:
            self.seen_states = self.seen_states[-1000:]
