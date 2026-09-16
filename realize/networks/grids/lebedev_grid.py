import torch
import math
from typing import Tuple

class LebedevGrid:
    """Lebedev quadrature grids for spherical integration"""
    
    @staticmethod
    def get_grid(degree: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get Lebedev grid for given degree
        
        Args:
            degree: 5, 9, or 11 (for 50, 110, or 170 points)
            
        Returns:
            points: Grid points on unit sphere [n, 3]
            weights: Integration weights [n]
        """
        if degree <= 5:
            return LebedevGrid._grid_050()
        elif degree <= 9:
            return LebedevGrid._grid_110()
        else:
            return LebedevGrid._grid_170()

    @staticmethod
    def get_grid_with_log_weights(degree: int, device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get Lebedev grid with both weights and log-weights
        
        Args:
            degree: 5, 9, or 11 (for 50, 110, or 170 points)
            device: Target device for tensors
            
        Returns:
            points: Grid points on unit sphere [n, 3]
            weights: Integration weights [n]
            log_weights: Log of integration weights [n]
        """
        points, weights = LebedevGrid.get_grid(degree)

        if device is not None:
            points = points.to(device)
            weights = weights.to(device)

        # clamp keeps log finite for zero/near-zero weights.
        log_weights = torch.log(weights.clamp(min=1e-10))
        
        return points, weights, log_weights

    @staticmethod
    def _grid_050() -> Tuple[torch.Tensor, torch.Tensor]:
        """50-point Lebedev grid (exact for l<=5)"""
        points = []
        weights = []
        
        # Type 1: (+/-1,0,0) and permutations
        w1 = 0.0126984126984127
        for i in range(3):
            for s in [1, -1]:
                p = [0, 0, 0]
                p[i] = s
                points.append(p)
                weights.append(w1)
        
        # Type 2: (+/-a,+/-a,0) and permutations, a = 1/sqrt2
        a2 = 1.0 / math.sqrt(2.0)
        w2 = 0.0225749559082176
        for i in range(3):
            for sx in [1, -1]:
                for sy in [1, -1]:
                    p = [0, 0, 0]
                    p[i] = 0
                    p[(i+1)%3] = sx * a2
                    p[(i+2)%3] = sy * a2
                    points.append(p)
                    weights.append(w2)
        
        # Type 3: (+/-a,+/-a,+/-a), a = 1/sqrt3
        a3 = 1.0 / math.sqrt(3.0)
        w3 = 0.0109417949847150
        for sx in [1, -1]:
            for sy in [1, -1]:
                for sz in [1, -1]:
                    points.append([sx*a3, sy*a3, sz*a3])
                    weights.append(w3)
        
        # Type 4: (+/-a,+/-b,0) and permutations
        a4, b4 = 0.3015113445777636, 0.9534625892455922
        w4 = 0.0189208064260222
        for vals in [[a4, b4], [b4, a4]]:
            for i in range(3):
                for sx in [1, -1]:
                    for sy in [1, -1]:
                        p = [0, 0, 0]
                        p[i] = 0
                        p[(i+1)%3] = sx * vals[0]
                        p[(i+2)%3] = sy * vals[1]
                        points.append(p)
                        weights.append(w4)
        
        points = torch.tensor(points, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights * (4 * torch.pi / weights.sum())  # weights sum to 4pi

        points = points / torch.norm(points, dim=1, keepdim=True)

        assert len(points) == 50, f"Expected 50 points, got {len(points)}"
        
        return points, weights

    @staticmethod
    def _grid_110() -> Tuple[torch.Tensor, torch.Tensor]:
        """110-point Lebedev grid (exact for l<=9)"""
        points = []
        weights = []
        
        # Type 1: (+/-1,0,0) and permutations
        w1 = 0.003828270494937162
        for i in range(3):
            for s in [1, -1]:
                p = [0, 0, 0]
                p[i] = s
                points.append(p)
                weights.append(w1)
        
        # Type 2: (+/-1/sqrt2, +/-1/sqrt2, 0) and permutations
        a2 = 1.0 / math.sqrt(2.0)
        w2 = 0.009793737512487512
        for i in range(3):
            for sx in [1, -1]:
                for sy in [1, -1]:
                    p = [0, 0, 0]
                    p[i] = 0
                    p[(i+1)%3] = sx * a2
                    p[(i+2)%3] = sy * a2
                    points.append(p)
                    weights.append(w2)
        
        # Type 3: (+/-1/sqrt3, +/-1/sqrt3, +/-1/sqrt3)
        a3 = 1.0 / math.sqrt(3.0)
        w3 = 0.008211737283191111
        for sx in [1, -1]:
            for sy in [1, -1]:
                for sz in [1, -1]:
                    points.append([sx*a3, sy*a3, sz*a3])
                    weights.append(w3)
        
        # Type 4: (+/-a4, +/-b4, 0) and permutations
        a4, b4 = 0.1846922114288918, 0.9829028488139443
        w4 = 0.007074905586561684
        for vals in [[a4, b4], [b4, a4]]:
            for i in range(3):
                for sx in [1, -1]:
                    for sy in [1, -1]:
                        p = [0, 0, 0]
                        p[i] = 0
                        p[(i+1)%3] = sx * vals[0]
                        p[(i+2)%3] = sy * vals[1]
                        points.append(p)
                        weights.append(w4)
        
        # Type 5: (+/-a5, +/-a5, +/-b5), 3 cyclic perms x 8 signs = 24 points
        a5, b5 = 0.5701670798443118, 0.8215439992318403
        w5 = 0.01116977719877011
        base_vals = [a5, a5, b5]
        for sx in [1, -1]:
            for sy in [1, -1]:
                for sz in [1, -1]:
                    points.append([sx*a5, sy*a5, sz*b5])  # (a, a, b)
                    weights.append(w5)
                    points.append([sx*a5, sy*b5, sz*a5])  # (a, b, a)
                    weights.append(w5)
                    points.append([sx*b5, sy*a5, sz*a5])  # (b, a, a)
                    weights.append(w5)

        # Type 6: (+/-a6, +/-b6, +/-c6), 6 perms x 8 signs = 48 points
        a6, b6, c6 = 0.2326020255423193, 0.4830618478381067, 0.8447285305722087
        w6 = 0.01118728150267822
        perms = [
            [a6, b6, c6], [a6, c6, b6],
            [b6, a6, c6], [b6, c6, a6],
            [c6, a6, b6], [c6, b6, a6]
        ]
        for perm in perms:
            for sx in [1, -1]:
                for sy in [1, -1]:
                    for sz in [1, -1]:
                        points.append([sx*perm[0], sy*perm[1], sz*perm[2]])
                        weights.append(w6)

        points = torch.tensor(points, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights * (4 * torch.pi / weights.sum())  # weights sum to 4pi

        points = points / torch.norm(points, dim=1, keepdim=True)

        assert len(points) == 110, f"Expected 110 points, got {len(points)}"
        
        return points, weights
    
    @staticmethod
    def _grid_170() -> Tuple[torch.Tensor, torch.Tensor]:
        """170-point Lebedev grid (exact for l<=11)"""
        points = []
        weights = []
        
        # Type 1: (+/-1,0,0) and permutations, 6 points
        w1 = 0.003333333333333333
        for i in range(3):
            for s in [1, -1]:
                p = [0, 0, 0]
                p[i] = s
                points.append(p)
                weights.append(w1)
        
        # Type 2: (+/-1/sqrt2, +/-1/sqrt2, 0) and permutations, 12 points
        a2 = 1.0 / math.sqrt(2.0)
        w2 = 0.003198565161784104
        for i in range(3):
            for sx in [1, -1]:
                for sy in [1, -1]:
                    p = [0, 0, 0]
                    p[i] = 0
                    p[(i+1)%3] = sx * a2
                    p[(i+2)%3] = sy * a2
                    points.append(p)
                    weights.append(w2)
        
        # Type 3: (+/-1/sqrt3, +/-1/sqrt3, +/-1/sqrt3), 8 points
        a3 = 1.0 / math.sqrt(3.0)
        w3 = 0.003609319754929666
        for sx in [1, -1]:
            for sy in [1, -1]:
                for sz in [1, -1]:
                    points.append([sx*a3, sy*a3, sz*a3])
                    weights.append(w3)
        
        # Type 4: (+/-a4, +/-b4, 0) and permutations, 24 points
        a4, b4 = 0.6356636053651498, 0.7719764691511746
        w4 = 0.006922891973680223
        for vals in [[a4, b4], [b4, a4]]:
            for i in range(3):
                for sx in [1, -1]:
                    for sy in [1, -1]:
                        p = [0, 0, 0]
                        p[i] = 0
                        p[(i+1)%3] = sx * vals[0]
                        p[(i+2)%3] = sy * vals[1]
                        points.append(p)
                        weights.append(w4)
        
        # Type 5: (+/-a5, +/-b5, 0) and permutations, 24 points
        a5, b5 = 0.3304677785504106, 0.9437893827156162
        w5 = 0.006412444741680419
        for vals in [[a5, b5], [b5, a5]]:
            for i in range(3):
                for sx in [1, -1]:
                    for sy in [1, -1]:
                        p = [0, 0, 0]
                        p[i] = 0
                        p[(i+1)%3] = sx * vals[0]
                        p[(i+2)%3] = sy * vals[1]
                        points.append(p)
                        weights.append(w5)
        
        # Type 6: (+/-a6, +/-a6, +/-b6) and permutations, 24 points
        a6, b6 = 0.4842439168601732, 0.8749283950470770
        w6 = 0.007386764762947244
        perms = [[0, 1, 2], [0, 2, 1], [2, 0, 1]]
        for perm in perms:
            for sx in [1, -1]:
                for sy in [1, -1]:
                    for sz in [1, -1]:
                        p = [0, 0, 0]
                        vals = [a6, a6, b6]
                        p[perm[0]] = sx * vals[0]
                        p[perm[1]] = sy * vals[1]
                        p[perm[2]] = sz * vals[2]
                        points.append(p)
                        weights.append(w6)
        
        # Type 7: (+/-a7, +/-a7, +/-b7) and permutations, 24 points
        a7, b7 = 0.2381436282654889, 0.9713960027029735
        w7 = 0.005105325832473484
        perms = [[0, 1, 2], [0, 2, 1], [2, 0, 1]]
        for perm in perms:
            for sx in [1, -1]:
                for sy in [1, -1]:
                    for sz in [1, -1]:
                        p = [0, 0, 0]
                        vals = [a7, a7, b7]
                        p[perm[0]] = sx * vals[0]
                        p[perm[1]] = sy * vals[1]
                        p[perm[2]] = sz * vals[2]
                        points.append(p)
                        weights.append(w7)
        
        # Type 8: (+/-a8, +/-b8, +/-c8), all 6 perms x 8 signs = 48 points
        a8, b8, c8 = 0.5082326932078424, 0.6805823924997263, 0.5282977936090615
        w8 = 0.007774356512558349
        import itertools
        for perm in itertools.permutations([a8, b8, c8]):
            for sx in [1, -1]:
                for sy in [1, -1]:
                    for sz in [1, -1]:
                        points.append([sx*perm[0], sy*perm[1], sz*perm[2]])
                        weights.append(w8)

        points = torch.tensor(points, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights * (4 * math.pi / weights.sum())  # weights sum to 4pi

        assert len(points) == 170, f"Expected 170 points, got {len(points)}"

        points = points / torch.norm(points, dim=1, keepdim=True)

        norms = torch.norm(points, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), "Points not on unit sphere"
        
        return points, weights        
