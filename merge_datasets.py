import pandas as pd
import os

# 1. Configuration
data_dir = 'data'
files = ['COMED_hourly.csv', 'DAYTON_hourly.csv', 'DEOK_hourly.csv']
output_file = 'data/unified_training_data.csv'

all_frames = []

print(" Starting Data Merger...")

for f in files:
    path = os.path.join(data_dir, f)
    if os.path.exists(path):
        df = pd.read_csv(path)
        
        # Standardize columns: We only need [Datetime, MW]
        # PJM files usually have the name of the region in the MW column
        df.columns = ['datetime', 'mw'] 
        
        print(f" Loaded {f} - Rows: {len(df)}")
        all_frames.append(df)
    else:
        print(f" Skipping {f} (Not found)")

# 2. Combine and Sort
final_df = pd.concat(all_frames, axis=0)
final_df['datetime'] = pd.to_datetime(final_df['datetime'])
final_df = final_df.sort_values('datetime')

# 3. Save Unified Version
final_df.to_csv(output_file, index=False)
print(f"\n Success! Unified dataset saved to {output_file}")
print(f"Total Records: {len(final_df)}")