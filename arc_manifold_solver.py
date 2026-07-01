import json
import numpy as np
from autonomous_manifold_v3 import AutonomousManifoldV3, ManifoldConfig

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

class ARCFeatureExtractorV2:
    @staticmethod
    def analyze_transformations(train_pairs):
        """Detects which standard transformations work for the training set."""
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
            in_grid = np.array(pair["input"])
            out_grid = np.array(pair["output"])

            for name, func in possible_transforms.items():
                try:
                    transformed = np.array(func(in_grid))
                    if transformed.shape == out_grid.shape and np.array_equal(transformed, out_grid):
                        scores[name] += 1
                except:
                    continue

        return scores

    @staticmethod
    def extract(task_data):
        train_pairs = task_data.get("train", [])
        transform_scores = ARCFeatureExtractorV2.analyze_transformations(train_pairs)

        num_pairs = len(train_pairs) if train_pairs else 1
        max_score = max(transform_scores.values()) if transform_scores else 0

        features = {
            "Creativity": 0.3,
            "Logic": max_score / num_pairs,
            "Density": np.mean([np.count_nonzero(p["output"]) / np.array(p["output"]).size for p in train_pairs]) if train_pairs else 0.5,
            "Realization": 0.1,
            "Entropy": 0.1 if max_score == num_pairs else 0.8
        }

        return features, transform_scores

class ARCManifoldSolverV2:
    def __init__(self):
        self.manifold = AutonomousManifoldV3()

    def solve_task(self, task_id, task_data):
        features, transform_scores = ARCFeatureExtractorV2.extract(task_data)

        result = self.manifold.process(
            {"user_state": f"Solve ARC task {task_id}", "external_input": features},
            iterations=5,
            training_mode=True
        )

        sorted_transforms = sorted(transform_scores.items(), key=lambda x: x[1], reverse=True)
        best_transform_name = sorted_transforms[0][0] if sorted_transforms and sorted_transforms[0][1] > 0 else "identity"
        second_best_name = sorted_transforms[1][0] if len(sorted_transforms) > 1 and sorted_transforms[1][1] > 0 else "flip_v"

        test_inputs = task_data.get("test", [])
        predictions = []

        for test_in in test_inputs:
            grid = np.array(test_in["input"])
            attempt_1 = getattr(ARCTransformers, best_transform_name)(grid)
            attempt_2 = getattr(ARCTransformers, second_best_name)(grid)
            predictions.append({"attempt_1": attempt_1, "attempt_2": attempt_2})

        return predictions

def solve_arc_tasks(dataset_path, limit=100):
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
    with open(dataset_path, "r") as f:
        challenges = json.load(f)
    solver = ARCManifoldSolverV2()
    submission = {}
    task_ids = list(challenges.keys())[:limit]
    for task_id in task_ids:
        print(f"Solving: {task_id}")
        submission[task_id] = solver.solve_task(task_id, challenges[task_id])
    for task_id in challenges:
        if task_id not in submission:
            test_inputs = challenges[task_id].get("test", [])
            submission[task_id] = [{"attempt_1": t["input"], "attempt_2": t["input"]} for t in test_inputs]
    with open("submission.json", "w") as f:
        json.dump(submission, f, indent=4)
    print(f"\n[SUCCESS] ARC Manifold Solver V2 generated submission.json for {len(submission)} tasks.")

if __name__ == "__main__":
    # solve_arc_tasks("/home/ubuntu/arc-agi-dataset/arc-agi_training_challenges.json")
    pass
