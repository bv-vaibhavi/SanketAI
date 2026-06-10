import os
import pandas as pd

print("📂 Merging all CSVs from both members...")

# Read all CSVs from dataset folder
all_data = []
for letter in os.listdir('dataset'):
    folder_path = f'dataset/{letter}'
    if os.path.isdir(folder_path):
        for csv_file in os.listdir(folder_path):
            if csv_file.endswith('.csv'):
                df = pd.read_csv(f'{folder_path}/{csv_file}')
                all_data.append(df)
                print(f'✅ Loaded {csv_file} ({len(df)} samples)')

# Combine all
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f'\n📊 Total samples: {len(combined_df)}')
    print(f'📋 Classes: {sorted(combined_df["label"].unique())}')
    
    # Save
    combined_df.to_csv('dataset_combined.csv', index=False)
    print('✅ Saved to dataset_combined.csv')
else:
    print("❌ No CSV files found!")