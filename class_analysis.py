import pandas as pd 
import os
from pathlib import Path

# Loop through all files in the folder
def class_analysis(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as f: # open the file to check if it was already processed
                content = f.read()
                if "# Summary" in content:
                    print(f"Skipping {filename} (already processed)")
                    continue
            df = pd.read_csv(file_path, skiprows=1)  # skip comment line
            grouped = df.groupby('class_type')
            mean_a = grouped['c_correct'].mean()
            mean_b = grouped['p_correct'].mean()
            summary = pd.DataFrame({
                'class_type': mean_a.index,
                'accuracy': (mean_a + mean_b) / 2
            })
            with open(file_path, "a") as f: # add the summary
                f.write("\n# Summary\n")
            summary.to_csv(file_path, mode='a', index=False)

if __name__ == "__main__":
    folder_path = Path("C:/Users/benca/PsyTask1/FECNet/Analysis/Res2.0")
    class_analysis(folder_path)
