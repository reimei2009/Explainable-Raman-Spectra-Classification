"""
Convert all .spc files from raw/Stone folder to csv/ folder
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import RamanDataLoader

def convert_stone_spc_to_csv():
    """Convert all .spc files from raw/Stone to csv/Stone"""
    
    raw_stone_dir = Path(__file__).parent / "Data" / "different potential" / "raw" / "Stone"
    csv_stone_dir = Path(__file__).parent / "Data" / "different potential" / "csv" / "Stone"
    
    if not raw_stone_dir.exists():
        print(f"❌ Raw Stone folder not found: {raw_stone_dir}")
        return False
    
    # Create CSV output folder if it doesn't exist
    csv_stone_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output folder: {csv_stone_dir}")
    
    loader = RamanDataLoader(verbose=False)
    
    total_files = 0
    total_errors = 0
    
    # Iterate through all sample folders
    for sample_folder in sorted(raw_stone_dir.iterdir()):
        if not sample_folder.is_dir():
            continue
        
        sample_name = sample_folder.name
        csv_sample_dir = csv_stone_dir / sample_name
        csv_sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all .spc files in this sample folder
        spc_files = sorted(sample_folder.glob("*.spc"))
        
        if not spc_files:
            print(f"⚠️  {sample_name}: no .spc files found")
            continue
        
        print(f"🔄 Processing {sample_name}... ({len(spc_files)} files)")
        
        for spc_file in spc_files:
            try:
                # Load spectrum
                spectrum = loader.load_spc(str(spc_file))
                wavenumber = spectrum["wavenumber"]
                intensity = spectrum["intensity"]
                
                # Create CSV output filename (same name, different extension)
                csv_filename = spc_file.stem + ".csv"
                csv_filepath = csv_sample_dir / csv_filename
                
                # Create DataFrame and save
                df = pd.DataFrame({
                    "wavenumber_cm1": wavenumber,
                    "intensity": intensity
                })
                df.to_csv(csv_filepath, index=False)
                
                total_files += 1
                
            except Exception as e:
                print(f"  ❌ Error converting {spc_file.name}: {e}")
                total_errors += 1
    
    print("\n" + "="*70)
    print(f"✅ Conversion complete!")
    print(f"   Total files converted: {total_files}")
    print(f"   Total errors: {total_errors}")
    print(f"   Output folder: {csv_stone_dir}")
    print("="*70)
    
    return total_errors == 0


if __name__ == "__main__":
    success = convert_stone_spc_to_csv()
    sys.exit(0 if success else 1)
