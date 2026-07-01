"""
hybrid_solver.py — A Neuro-Symbolic Hybrid ARC Solver.
"""
import os
import sys
from typing import Optional, Tuple, Dict, List
import random
import numpy as np
import torch
import torch.nn as nn

SEED = 42
N_COLORS = 10
OOB_INDEX = 10
NEURAL_EPOCHS = 100
NEURAL_LR = 0.01
NEURAL_EMB_DIM = 8

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

def set_deterministic(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SymbolicSolver:
    def __init__(self):
        self.geom_transforms = {
            "identity": lambda x: np.ascontiguousarray(x.copy()),
            "rot90_1": lambda x: np.ascontiguousarray(np.rot90(x, 1)),
            "rot90_2": lambda x: np.ascontiguousarray(np.rot90(x, 2)),
            "rot90_3": lambda x: np.ascontiguousarray(np.rot90(x, 3)),
            "flip_lr": lambda x: np.ascontiguousarray(np.fliplr(x)),
            "flip_ud": lambda x: np.ascontiguousarray(np.flipud(x)),
            "transpose": lambda x: np.ascontiguousarray(x.T),
            "anti_transpose": lambda x: np.ascontiguousarray(np.rot90(x, 2).T)
        }

    def find_consistent_geom_transform(self, train_pairs: List[dict]) -> Optional[str]:
        for name, func in self.geom_transforms.items():
            solved_all = True
            for pair in train_pairs:
                inp, out = np.array(pair["input"]), np.array(pair["output"])
                if not np.array_equal(func(inp), out):
                    solved_all = False; break
            if solved_all: return name
        return None

    def find_consistent_geom_plus_color(self, train_pairs: List[dict]) -> Optional[Tuple[str, Dict[int, int]]]:
        for geom_name, geom_func in self.geom_transforms.items():
            mapping, reverse_mapping = {}, {}
            valid = True
            for pair in train_pairs:
                inp, out = np.array(pair["input"]), np.array(pair["output"])
                transformed = geom_func(inp)
                if transformed.shape != out.shape:
                    valid = False; break
                for src, dst in zip(transformed.flat, out.flat):
                    src, dst = int(src), int(dst)
                    if (src in mapping and mapping[src] != dst) or (dst in reverse_mapping and reverse_mapping[dst] != src):
                        valid = False; break
                    mapping[src], reverse_mapping[dst] = dst, src
                if not valid: break
            if valid: return geom_name, mapping
        return None

    def find_consistent_color_mapping(self, train_pairs: List[dict]) -> Optional[Dict[int, int]]:
        mapping, reverse_mapping = {}, {}
        for pair in train_pairs:
            inp, out = np.array(pair["input"]), np.array(pair["output"])
            if inp.shape != out.shape: return None
            for src, dst in zip(inp.flat, out.flat):
                src, dst = int(src), int(dst)
                if (src in mapping and mapping[src] != dst) or (dst in reverse_mapping and reverse_mapping[dst] != src):
                    return None
                mapping[src], reverse_mapping[dst] = dst, src
        return mapping

def grid_to_tabular(inp_grid: np.ndarray, out_grid: np.ndarray = None):
    H, W = inp_grid.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    y_norm = (ys.flatten() / max(1, H - 1)) - 0.5
    x_norm = (xs.flatten() / max(1, W - 1)) - 0.5
    num_feats = np.stack([y_norm, x_norm], axis=-1)
    padded = np.pad(inp_grid, pad_width=1, mode='constant', constant_values=OOB_INDEX)
    cat_feats = []
    for r in range(H):
        for c in range(W):
            pr, pc = r + 1, c + 1
            pixel_neighborhood = [padded[pr, pc]]
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    pixel_neighborhood.append(padded[pr + dr, pc + dc])
            cat_feats.append(pixel_neighborhood)
    num_tensor = torch.tensor(num_feats, dtype=torch.float32)
    cat_tensor = torch.tensor(cat_feats, dtype=torch.long)
    targets_tensor = torch.tensor(out_grid.flatten(), dtype=torch.long) if out_grid is not None else None
    return num_tensor, cat_tensor, targets_tensor

class TabularCoordNet(nn.Module):
    def __init__(self, cat_cardinality: int = 11, emb_dim: int = 8):
        super().__init__()
        self.shared_emb = nn.Embedding(cat_cardinality, emb_dim)
        total_input_dim = 2 + (9 * emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, N_COLORS)
        )
    def forward(self, num_x, cat_x):
        embedded_cats = self.shared_emb(cat_x)
        embedded_flat = embedded_cats.view(embedded_cats.size(0), -1)
        x = torch.cat([num_x, embedded_flat], dim=-1)
        return self.mlp(x)

class HybridARCSolver:
    def __init__(self, verbose: bool = False, seed: int = SEED):
        self.verbose, self.seed = verbose, seed
        set_deterministic(seed)
        self.symbolic_solver = SymbolicSolver()

    def log(self, msg: str):
        if self.verbose: print(msg)

    def solve(self, task: dict) -> List[np.ndarray]:
        train_pairs = task["train"]
        test_inputs = [np.array(test["input"]) for test in task["test"]]
        predictions = []
        rule = self.symbolic_solver.find_consistent_geom_transform(train_pairs)
        if rule:
            self.log(f"Stage 1a: {rule}")
            return [self.symbolic_solver.geom_transforms[rule](ti) for ti in test_inputs]
        gc_rule = self.symbolic_solver.find_consistent_geom_plus_color(train_pairs)
        if gc_rule:
            rule, cmap = gc_rule
            self.log(f"Stage 1b: {rule} + {cmap}")
            for ti in test_inputs:
                tr = self.symbolic_solver.geom_transforms[rule](ti)
                out = tr.copy()
                for s, d in cmap.items(): out[tr == s] = d
                predictions.append(out)
            return predictions
        lookup, consistent = {}, True
        for pair in train_pairs:
            _, cat_t, tgt_t = grid_to_tabular(np.array(pair["input"]), np.array(pair["output"]))
            for crow, tgt in zip(cat_t.tolist(), tgt_t.tolist()):
                k = tuple(crow)
                if k in lookup and lookup[k] != tgt: consistent = False; break
                lookup[k] = tgt
            if not consistent: break
        if consistent:
            all_cov = True
            for ti in test_inputs:
                _, tcat, _ = grid_to_tabular(ti)
                if any(tuple(r) not in lookup for r in tcat.tolist()): all_cov = False; break
            if all_cov:
                self.log("Stage 2a")
                for ti in test_inputs:
                    _, tcat, _ = grid_to_tabular(ti)
                    predictions.append(np.array([lookup[tuple(r)] for r in tcat.tolist()]).reshape(ti.shape))
                return predictions
        self.log("Stage 2b")
        all_num, all_cat, all_tgt = [], [], []
        for pair in train_pairs:
            n, c, t = grid_to_tabular(np.array(pair["input"]), np.array(pair["output"]))
            all_num.append(n); all_cat.append(c); all_tgt.append(t)
        t_num, t_cat, t_tgt = torch.cat(all_num), torch.cat(all_cat), torch.cat(all_tgt)
        net = TabularCoordNet(emb_dim=NEURAL_EMB_DIM)
        opt = torch.optim.Adam(net.parameters(), lr=NEURAL_LR)
        crit = nn.CrossEntropyLoss()
        for _ in range(NEURAL_EPOCHS):
            opt.zero_grad(); crit(net(t_num, t_cat), t_tgt).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            for ti in test_inputs:
                n, c, _ = grid_to_tabular(ti)
                predictions.append(torch.argmax(net(n, c), -1).numpy().reshape(ti.shape))
        return predictions

if __name__ == "__main__":
    set_deterministic(SEED)
    solver = HybridARCSolver(verbose=True)
    context_task = {
        "train": [
            {"input": [[1, 2, 0], [0, 1, 0]], "output": [[4, 2, 0], [0, 4, 0]]},
            {"input": [[1, 0, 0], [0, 0, 2]], "output": [[1, 0, 0], [0, 0, 2]]}
        ],
        "test": [
            {"input": [[0, 1, 2], [1, 0, 0]]}
        ]
    }
    pred = solver.solve(context_task)[0]
    expected = np.array([[0, 4, 2], [1, 0, 0]])
    if np.array_equal(pred, expected): print("Test 3 Passed")
    else: print(f"Test 3 Failed. Pred:\n{pred}")
