import json, os, pandas as pd
from io import StringIO
def extract_from_json(json_path):
    with open(json_path, 'r') as f: data = json.load(f)
    num_df = pd.read_csv(StringIO(data['num.txt']), sep='\t')
    sub_df = pd.read_csv(StringIO(data['sub.txt']), sep='\t')
    target_tags = ['Assets', 'Liabilities', 'AssetsCurrent', 'LiabilitiesCurrent']
    merged = pd.merge(num_df[num_df['tag'].isin(target_tags)], sub_df[['adsh', 'name', 'cik']], on='adsh')
    return merged
def main():
    all_data = []
    base_dir = 'kaggle_data/sec_financials'
    if not os.path.exists(base_dir): return
    for f in [f for f in os.listdir(base_dir) if f.endswith('.json')]:
        try: all_data.append(extract_from_json(os.path.join(base_dir, f)))
        except: pass
    if all_data: pd.concat(all_data, ignore_index=True).to_csv('extracted_assets_liabilities.csv', index=False)
if __name__ == "__main__": main()
