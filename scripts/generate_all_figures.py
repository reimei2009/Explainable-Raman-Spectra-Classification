"""
generate_all_figures.py - Generate all missing visualizations for Stone dataset samples.

Run from project root:
    python scripts/generate_all_figures.py

This script:
1. Loads all 38 deduplicated samples (13 from original + 25 from Stone)
2. For each sample, creates a waterfall plot (all potentials)
3. Saves to outputs/figures/per_sample/<sample>_waterfall.png
4. Creates summary statistics table
"""

import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import logging
import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import RamanDataLoader
from src.utils import ensure_dir, get_project_root, list_samples, setup_logging
from src.visualization import RamanVisualizer

# ---------------------------------------------------------------------------

def main():
    setup_logging()
    log = logging.getLogger(__name__)
    
    # Locate CSV data (not raw, since we're working with CSV)
    root = get_project_root()
    csv_root = root / "Data" / "different potential" / "csv"
    
    if not csv_root.exists():
        log.error("CSV data directory not found: %s", csv_root)
        sys.exit(1)
    
    log.info("CSV root: %s", csv_root)
    
    # Get all sample folders
    all_samples = sorted([d.name for d in csv_root.iterdir() if d.is_dir()])
    log.info("Found %d samples", len(all_samples))
    
    # Ensure output directories
    figures_dir = root / "outputs" / "figures"
    per_sample_dir = figures_dir / "per_sample"
    ensure_dir(figures_dir)
    ensure_dir(per_sample_dir)
    
    loader = RamanDataLoader(verbose=False)
    viz = RamanVisualizer(output_dir=per_sample_dir)
    
    # Track progress
    success_count = 0
    skip_count = 0
    error_count = 0
    
    missing_samples = []
    existing_samples = set(f.stem.replace("_waterfall", "") for f in per_sample_dir.glob("*.png"))
    
    print("\n" + "="*70)
    print("  GENERATING WATERFALL PLOTS FOR ALL SAMPLES")
    print("="*70)
    print(f"\nTotal samples to process: {len(all_samples)}")
    print(f"Already existing: {len(existing_samples)}\n")
    
    # Generate figures for each sample
    for idx, sample_name in enumerate(all_samples, 1):
        sample_dir = csv_root / sample_name
        
        # Check if already exists
        figure_path = per_sample_dir / f"{sample_name}_waterfall.png"
        if figure_path.exists():
            print(f"[{idx:2d}/{len(all_samples)}] ✓ {sample_name:15s} (exists, skipping)")
            skip_count += 1
            continue
        
        try:
            # Load sample folder
            sample_data = loader.load_sample_folder(str(sample_dir))
            
            if not sample_data:
                log.warning("No data found for %s", sample_name)
                missing_samples.append(sample_name)
                error_count += 1
                print(f"[{idx:2d}/{len(all_samples)}] ✗ {sample_name:15s} (no data)")
                continue
            
            # Create waterfall plot
            fig = viz.plot_potential_series(sample_data, sample_name=sample_name)
            
            # Save figure
            saved_path = viz.save_figure(fig, f"{sample_name}_waterfall.png")
            plt.close(fig)
            
            print(f"[{idx:2d}/{len(all_samples)}] ✓ {sample_name:15s} ({len(sample_data)} potentials)")
            success_count += 1
            
        except Exception as e:
            log.error("Error processing %s: %s", sample_name, e)
            print(f"[{idx:2d}/{len(all_samples)}] ✗ {sample_name:15s} (error: {str(e)[:40]})")
            error_count += 1
    
    # Summary
    print("\n" + "="*70)
    print("  GENERATION SUMMARY")
    print("="*70)
    print(f"✓ Successfully created:   {success_count}")
    print(f"⊘ Already existing:       {skip_count}")
    print(f"✗ Errors:                 {error_count}")
    print(f"Total:                    {success_count + skip_count + error_count}/{len(all_samples)}")
    print("="*70)
    
    if missing_samples:
        print(f"\nMissing data samples: {', '.join(missing_samples)}")
    
    print(f"\nFigures saved to: {per_sample_dir}\n")


if __name__ == "__main__":
    main()
