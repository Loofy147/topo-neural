import json
import numpy as np
from autonomous_manifold_v4 import AutonomousManifoldV4

class ARCTransformers:
    @staticmethod
    def identity(grid): return grid.tolist()

    @staticmethod
    def rotate_90(grid): return np.rot90(grid, k=1).tolist()

    @staticmethod
    def rotate_180(grid): return np.rot90(grid, k=2).tolist()

    @staticmethod
    def rotate_270(grid): return np.rot90(grid, k=3).tolist()

    @staticmethod
    def flip_v(grid): return np.flipud(grid).tolist()

    @staticmethod
    def flip_h(grid): return np.fliplr(grid).tolist()

    @staticmethod
    def transpose(grid): return grid.T.tolist()

class ARCFeatureExtractorV3:
    @staticmethod
    def analyze_transformations(train_pairs):
        possible_transforms = {
            "identity": ARCTransformers.identity,
            "rotate_90": ARCTransformers.rotate_90,
            "rotate_180": ARCTransformers.rotate_180,
            "rotate_270": ARCTransformers.rotate_270,
            "flip_v": ARCTransformers.flip_v,
            "flip_h": ARCTransformers.flip_h,
            "transpose": ARCTransformers.transpose
        }
        scores = {k: 0 for k in possible_transforms}
        for pair in train_pairs:
            in_grid, out_grid = np.array(pair["input"]), np.array(pair["output"])
            for name, func in possible_transforms.items():
                try:
                    transformed = np.array(func(in_grid))
                    if transformed.shape == out_grid.shape and np.array_equal(transformed, out_grid):
                        scores[name] += 1
                except: continue
        return scores

    @staticmethod
    def extract(task_data):
        train_pairs = task_data.get("train", [])
        scores = ARCFeatureExtractorV3.analyze_transformations(train_pairs)
        num_pairs = len(train_pairs) if train_pairs else 1
        max_score = max(scores.values()) if scores else 0
        features = {
            "user_state": f"ARC_Reasoning",
            "pattern_type": "even" if max_score == num_pairs else "complex",
            "data_volume": len(train_pairs) * 100,
            "Creativity": 0.3,
            "Logic": max_score / num_pairs,
            "Density": np.mean([np.count_nonzero(p["output"]) / np.array(p["output"]).size for p in train_pairs]) if train_pairs else 0.5,
            "Realization": 0.1,
            "Entropy": 0.1 if max_score == num_pairs else 0.8
        }
        return features, scores

class ARCManifoldSolverV3:
    def __init__(self):
        self.manifold = AutonomousManifoldV4()

    def solve_task(self, task_id, task_data):
        features, scores = ARCFeatureExtractorV3.extract(task_data)
        result = self.manifold.process(features, iterations=5)

        sorted_transforms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best = sorted_transforms[0][0] if sorted_transforms and sorted_transforms[0][1] > 0 else "identity"
        second = sorted_transforms[1][0] if len(sorted_transforms) > 1 and sorted_transforms[1][1] > 0 else "flip_v"

        predictions = []
        for test_in in task_data.get("test", []):
            grid = np.array(test_in["input"])
            predictions.append({
                "attempt_1": getattr(ARCTransformers, best)(grid),
                "attempt_2": getattr(ARCTransformers, second)(grid)
            })
        return predictions

if __name__ == "__main__":
    solver = ARCManifoldSolverV3()
    sample = {"train": [{"input": [[1,1],[0,0]], "output": [[0,0],[1,1]]}], "test": [{"input": [[1,0],[1,0]]}]}
    print(solver.solve_task("sample", sample))
