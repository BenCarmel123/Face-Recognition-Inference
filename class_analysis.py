import pandas as pd 
import os
from pathlib import Path

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

            df = pd.read_csv(file_path, skiprows=1)  # skip comment line

            if 'class_type' not in df.columns or 'l2_correct' not in df.columns:
                print(f"Skipping {filename} (missing required columns)")
                continue

            grouped = df.groupby('class_type')
            mean_l2 = grouped['l2_correct'].mean()

            summary = pd.DataFrame({
                'class_type': mean_l2.index,
                'l2_accuracy': mean_l2
            })

            with open(file_path, "a") as f:  # add the summary section
                f.write("\n# Summary\n")
            summary.to_csv(file_path, mode='a', index=False)

if __name__ == "__main__":
    folder_path = Path("C:/Users/benca/PsyTask1/FECNet/Analysis/Res4.0")
    class_analysis(folder_path)
