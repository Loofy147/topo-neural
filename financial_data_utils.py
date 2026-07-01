import json, torch, os, pandas as pd, numpy as np
from torch.utils.data import Dataset, DataLoader
class FinancialAssetDataset(Dataset):
    def __init__(self, csv_path):
        self.samples = []
        if not os.path.exists(csv_path): return
        df = pd.read_csv(csv_path)
        for _, g in df.groupby(['adsh', 'ddate']):
            a, l = g[g['tag'] == 'Assets']['value'], g[g['tag'] == 'Liabilities']['value']
            if not a.empty and not l.empty:
                self.samples.append((torch.tensor(self._to_8bit(a.iloc[0]), dtype=torch.float32), torch.tensor(self._to_8bit(l.iloc[0]), dtype=torch.float32)))
    def _to_8bit(self, val):
        if val <= 0: return [0]*8
        q = int(np.clip((np.log10(val) - 6) / 6 * 255, 0, 255))
        return [int(b) for b in format(q, '08b')]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]
def get_financial_dataloader(path, batch_size=32):
    ds = FinancialAssetDataset(path)
    return DataLoader(ds, batch_size=batch_size, shuffle=True) if len(ds) > 0 else [(torch.randn(batch_size, 8), torch.randint(0, 2, (batch_size, 8)).float())]
