import pandas as pd 
import os
from pathlib import Path
import sys

# Loop through all files in the folder
def class_analysis(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, "r") as f:  # open the file to check if it was already processed
                content = f.read()
                if "# Summary" in content:
                    print(f"Skipping {filename} (already processed)")
                    continue

            df = pd.read_csv(file_path, skiprows=1) if filename.startswith('results_') else pd.read_csv(file_path)

            if 'label' not in df.columns or 'is_correct' not in df.columns:
                print(f"Skipping {filename} (missing required columns)")
                continue

            # Group by class label and calculate mean accuracy for each class
            grouped = df.groupby('label')
            mean_acc = grouped['is_correct'].mean()

            summary = pd.DataFrame({
                'class_label': mean_acc.index,
                'accuracy': mean_acc
            })

            with open(file_path, "a") as f:  # add the summary section
                f.write("\n# Summary (accuracy by class)\n")
            summary.to_csv(file_path, mode='a', index=False)
            print(f"Processed {filename} - class-wise accuracy summary added.")

if __name__ == "__main__":
    folder_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    class_analysis(folder_path)
