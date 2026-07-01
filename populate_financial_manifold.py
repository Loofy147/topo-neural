import os, numpy as np, pandas as pd
from autonomous_manifold_v3 import AutonomousManifoldV3
def project_to_1024(vector, masses):
    base = np.array(list(vector.values()) + list(masses.values()))
    expanded = np.tile(base, (1024 // len(base)) + 1)[:1024]
    return (expanded + 0.1 * np.sin(np.linspace(0, 10, 1024) * expanded.sum())).reshape(32, 32)
def main():
    if not os.path.exists('extracted_assets_liabilities.csv'): return
    df = pd.read_csv('extracted_assets_liabilities.csv')
    df['ddate'] = pd.to_datetime(df['ddate'])
    pivoted = df.loc[df.groupby(['adsh', 'tag'])['ddate'].idxmax()].pivot(index='adsh', columns='tag', values='value').dropna(subset=['Assets', 'Liabilities'], how='all')
    pivoted['ratio'] = pivoted['Assets'] / pivoted['Liabilities'].replace(0, 1)
    out_dir = 'kaggle_data/financial_manifold'
    os.makedirs(out_dir, exist_ok=True)
    manifold = AutonomousManifoldV3(persistence_file="financial_manifold_state.json")
    for count, (adsh, row) in enumerate(pivoted.sort_values(by='Assets', ascending=False).head(100).iterrows()):
        ext = {"Logic": min(1.0, row['Assets'] / 1e12), "Realization": min(1.0, row['ratio'] / 5.0), "Density": min(1.0, row.get('AssetsCurrent', 0) / (row['Assets'] + 1e-6))}
        res = manifold.process({"user_state": f"Financial evolution", "external_input": ext}, iterations=5)
        np.save(os.path.join(out_dir, f'weight_{count}.npy'), project_to_1024(res['state_vector'], res['agent_masses']))
        lib = np.tile(np.array(list(res['agent_masses'].values())), (1024 // len(res['agent_masses'])) + 1)[:1024].reshape(32, 32)
        np.save(os.path.join(out_dir, f'lib_{count}.npy'), lib)
if __name__ == "__main__": main()
