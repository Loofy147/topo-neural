import json
import torch
from torch.utils.data import Dataset, DataLoader

class StratosCoTDataset(Dataset):
    def __init__(self, file_path):
        self.samples = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                instruction = data['instruction']
                response = data['response']

                # Extract input for prediction (last line of instruction)
                input_str = instruction.split(':')[-1].strip()

                # Extract target from boxed answer
                if '\\boxed{' in response:
                    target_str = response.split('\\boxed{')[-1].split('}')[0].strip()
                else:
                    continue

                if len(input_str) == 8 and len(target_str) == 8:
                    input_bits = [int(b) for b in input_str]
                    target_bits = [int(b) for b in target_str]
                    self.samples.append((
                        torch.tensor(input_bits, dtype=torch.float32),
                        torch.tensor(target_bits, dtype=torch.float32)
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def get_dataloader(file_path, batch_size=32):
    dataset = StratosCoTDataset(file_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    loader = get_dataloader('./kaggle_data/stratoscot/augmented_train.jsonl')
    x, y = next(iter(loader))
    print(f"Batch X shape: {x.shape}")
    print(f"Batch Y shape: {y.shape}")
