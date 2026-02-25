
import pandas as pd
import numpy as np

file_path = r"C:\Users\feng\Desktop\新建文件夹 (2)\kaggle\consts\笔记\EnMAP_Spectral_Bands_update.xlsx"

try:
    # Read Excel. Usually headers are on row 0.
    # We look for a column named 'center_wavelength' or similar.
    df = pd.read_excel(file_path)
    print("Columns found:", df.columns.tolist())
    
    # Try to find wavelength column
    # Common names: 'Center_Wavelength', 'Wavelength', 'nm', 'center'
    possible_cols = [c for c in df.columns if 'center' in str(c).lower() or 'wl' in str(c).lower() or 'wavelength' in str(c).lower()]
    
    if possible_cols:
        target_col = possible_cols[0]
        print(f"Extracting strictly from column: {target_col}")
        wavs = df[target_col].dropna().values
        
        # Sort just in case
        wavs = np.sort(wavs)
        
        print(f"Found {len(wavs)} bands.")
        print("First 10:", wavs[:10])
        print("Last 10:", wavs[-10:])
        
        # Print full list formatted for Python list
        print("\nPY_LIST_START")
        print("[")
        for i, w in enumerate(wavs):
            end = ", " if (i+1)%10 != 0 else ",\n    "
            if i == len(wavs)-1: end = ""
            print(f"{w:.2f}", end=end)
        print("\n]")
        print("PY_LIST_END")
        
    else:
        print("Could not identify wavelength column. First few rows:")
        print(df.head())

except Exception as e:
    print(f"Error reading excel: {e}")
    # Fallback: maybe need openpyxl?
    print("Ensure openpyxl is installed.")
