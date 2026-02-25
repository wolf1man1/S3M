
import pandas as pd
import numpy as np

file_path = r"C:\Users\feng\Desktop\新建文件夹 (2)\kaggle\consts\笔记\EnMAP_Spectral_Bands_update.xlsx"

try:
    df = pd.read_excel(file_path)
    print("Columns found:", df.columns.tolist())
    
    # Try assuming Col 1 is Wavelength (0-based index)
    # Standard format: Band | Wavelength | FWHM
    target_col = df.columns[1]
    print(f"Extracting from Column Index 1: '{target_col}'")
    
    wavs = df.iloc[:, 1].dropna().values
    
    # Ensure they are numeric
    try:
        wavs = wavs.astype(float)
    except:
        print("Warning: Contains non-numeric? Cleaning...")
        # Replace non-numeric
        wavs = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna().values
    
    wavs = np.sort(wavs)
    
    print(f"Found {len(wavs)} bands.")
    print("First 10:", wavs[:10])
    
    # Write to file for clean reading
    with open("enmap_bands.txt", "w") as f:
        f.write("HYSPECNET_WAVELENGTHS = [\n    ")
        for i, w in enumerate(wavs):
            end = ", " if (i+1)%10 != 0 else ",\n    "
            if i == len(wavs)-1: end = ""
            f.write(f"{w:.2f}{end}")
        f.write("\n]")
    print("Successfully wrote to enmap_bands.txt")

except Exception as e:
    print(f"Error: {e}")
